from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.libero_dataset import LiberoVLADataset
from train import build_model, normalize_action_head_state_dict


def _build_dataset(cfg):
    return LiberoVLADataset(
        dataset_name=cfg.dataset.name,
        data_root=cfg.dataset.get("data_root", None),
        action_horizon=cfg.action_head.action_horizon,
        image_size=tuple(cfg.dataset.image_size),
        action_type=cfg.dataset.action_type,
        split="train",
        vlm_model_id=cfg.model.vlm_model_id,
        max_train_samples=max(int(cfg.dataset.get("max_train_samples", 0)), 16),
        subset_seed=cfg.dataset.get("subset_seed", 42),
        norm_stats_sample_size=cfg.dataset.get("norm_stats_sample_size", 2048),
        history_len=cfg.action_head.get("history_len", 0),
        tokenizer_padding_side=cfg.dataset.get("tokenizer_padding_side", "right"),
        tokenizer_max_length=cfg.dataset.get("tokenizer_max_length", None),
        prompt_style=cfg.dataset.get("prompt_style", "pi05_state_prompt"),
        state_prompt_bins=cfg.dataset.get("state_prompt_bins", cfg.action_head.get("state_prompt_bins", 256)),
        image_resize_mode=cfg.dataset.get("image_resize_mode", "pad"),
        empty_cameras=cfg.dataset.get("empty_cameras", 0),
    )


def _set_norm_stats(model, dataset):
    model.set_norm_stats(
        dataset.norm_stats["q01"],
        dataset.norm_stats["q99"],
        dataset.norm_stats.get("local_q01", dataset.norm_stats["q01"]),
        dataset.norm_stats.get("local_q99", dataset.norm_stats["q99"]),
        dataset.norm_stats.get("state_q01", None),
        dataset.norm_stats.get("state_q99", None),
    )


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description="Unit-style validation for qwen v3.2 Gemma bridge")
    parser.add_argument(
        "--config",
        type=str,
        default="config/libero_train_qwen_3_5_2b_gemma_bridge_v3_2_smoke1000.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    _assert(
        str(cfg.model.architecture) == "qwen_3.5_2b_gemma_bridge_v3_2",
        "Expected v3.2 bridge architecture config",
    )
    _assert(
        str(cfg.action_head.head_type) == "pi05_gemma_bridge",
        "Expected pi05_gemma_bridge head_type",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = _build_dataset(cfg)
    sample = dataset[0]

    model = build_model(cfg)
    _set_norm_stats(model, dataset)
    model = model.to(device)
    model.train()

    images = [sample["images"]]
    instructions = [sample["instruction"]]
    actions = sample["actions"].unsqueeze(0).to(device)
    state = sample["state"].unsqueeze(0).to(device)

    prompts = model._build_pi05_prompts(instructions, state)
    prefix_ctx = model.backbone.encode_prefix(images=images, prompts=prompts)
    prefix_memory = model.backbone.get_prefix_memory(
        prefix_ctx,
        tap_strategy=model.action_config.tap_strategy,
    )
    full_attention_indices = model.backbone.full_attention_layer_indices()

    _assert(full_attention_indices, "No Qwen full-attention tap indices were found")
    if str(cfg.action_head.tap_strategy) == "last":
        _assert(
            prefix_memory["tap_indices"] == [full_attention_indices[-1]],
            "tap_strategy=last must choose last full-attention layer",
        )
    else:
        _assert(
            prefix_memory["tap_indices"] == full_attention_indices,
            "multi-tap strategies must retain all full-attention layers",
        )
    _assert(prefix_memory["memory"].ndim == 3, "Prefix memory must be rank-3")
    _assert(prefix_memory["memory"].shape[0] == 1, "Expected batch size 1 for structure test")
    _assert(
        prefix_memory["memory"].shape[-1] == int(cfg.action_head.vlm_hidden_dim),
        "Tap memory width must remain Qwen hidden size before bridge",
    )
    _assert("memory_mask" in prefix_memory, "Prefix memory must expose memory_mask")
    _assert(
        prefix_memory["memory_mask"] is not None
        and prefix_memory["memory_mask"].shape == prefix_memory["attention_mask"].shape,
        "memory_mask must align with attention_mask",
    )
    _assert(
        bool(prefix_memory["memory_mask"].any().item()),
        "memory_mask unexpectedly contains no attended tokens",
    )
    _assert("tap_ids" in prefix_memory, "Prefix memory must expose tap_ids")
    _assert(
        prefix_memory["tap_ids"] is not None
        and prefix_memory["tap_ids"].shape == prefix_memory["attention_mask"].shape,
        "tap_ids must align with attention_mask",
    )
    _assert(
        "token_type_ids" in prefix_memory
        and prefix_memory["token_type_ids"] is not None
        and prefix_memory["token_type_ids"].shape == prefix_memory["attention_mask"].shape,
        "token_type_ids must align with attention_mask",
    )
    _assert(
        bool(torch.isfinite(prefix_memory["memory"]).all().item()),
        "Prefix memory contains NaN/Inf",
    )

    outputs = model(
        images=images,
        instructions=instructions,
        actions=actions,
        state=state,
    )
    loss = outputs["total_loss"]
    _assert(bool(torch.isfinite(loss).item()), "Forward loss is not finite")
    for key in (
        "memory_norm_ratio",
        "query_token_norm",
        "cross_attn_entropy",
        "mean_norm_saturation_fraction",
    ):
        _assert(key in outputs, f"v3.2 bridge forward must report {key}")
    ratio = float(outputs["memory_norm_ratio"].detach().cpu().item())
    saturation_fraction = float(outputs["mean_norm_saturation_fraction"].detach().cpu().item())
    _assert(np.isfinite(ratio), "memory_norm_ratio is not finite")
    _assert(np.isfinite(saturation_fraction), "mean_norm_saturation_fraction is not finite")

    tap_weight_entropy = None
    scalar_mix_gamma = None
    tap_weight_distribution = None
    if str(cfg.action_head.tap_strategy) == "scalar_mix":
        _assert("tap_weight_entropy" in outputs, "scalar_mix must report tap_weight_entropy")
        _assert("scalar_mix_gamma" in outputs, "scalar_mix must report scalar_mix_gamma")
        tap_weight_entropy = float(outputs["tap_weight_entropy"].detach().cpu().item())
        scalar_mix_gamma = float(outputs["scalar_mix_gamma"].detach().cpu().item())
        tap_weight_distribution = model.action_head.last_bridge_stats.get("tap_weight_distribution", None)
        _assert(isinstance(tap_weight_distribution, list), "scalar_mix must record tap_weight_distribution")
        _assert(
            abs(sum(tap_weight_distribution) - 1.0) < 1e-5,
            "scalar_mix tap weights must sum to 1",
        )

    loss.backward()

    action_grad_norm = 0.0
    for param in model.action_head.parameters():
        if param.grad is not None:
            action_grad_norm += float(param.grad.detach().float().pow(2).sum().item())
    action_grad_norm = float(action_grad_norm ** 0.5)
    _assert(action_grad_norm > 0.0, "Expected non-zero gradients on bridge/expert parameters")

    vlm_grads = []
    for name, param in model.vlm.named_parameters():
        if param.requires_grad and param.grad is not None:
            vlm_grads.append(name)
    _assert(not vlm_grads, f"Frozen Qwen backbone unexpectedly received gradients: {vlm_grads[:5]}")

    model.eval()
    head = model.action_head
    head_dtype = model._action_head_dtype()
    norm_actions = model.normalize_actions(actions).to(head_dtype)
    norm_state = model.normalize_states(state).to(head_dtype)
    padded_actions = head._pad_actions(norm_actions)
    timestep = torch.full((padded_actions.shape[0],), 0.5, device=device, dtype=head_dtype)
    vel_with_memory, _ = head._predict_velocity(
        prefix_memory=prefix_memory["memory"],
        prefix_attention_mask=prefix_memory["memory_mask"],
        tap_ids=prefix_memory.get("tap_ids"),
        token_type_ids=prefix_memory.get("token_type_ids"),
        noisy_actions=padded_actions,
        timestep=timestep,
        state=norm_state,
        use_memory_conditioning=False,
    )
    vel_with_zero_memory, _ = head._predict_velocity(
        prefix_memory=torch.zeros_like(prefix_memory["memory"]),
        prefix_attention_mask=prefix_memory["memory_mask"],
        tap_ids=prefix_memory.get("tap_ids"),
        token_type_ids=prefix_memory.get("token_type_ids"),
        noisy_actions=padded_actions,
        timestep=timestep,
        state=norm_state,
        use_memory_conditioning=False,
    )
    parity_error = float((vel_with_memory - vel_with_zero_memory).abs().max().detach().cpu().item())
    _assert(parity_error < 1e-5, f"Bridge-off parity failed: max error {parity_error}")

    with torch.no_grad():
        pred_base = model.predict_action(
            images=images,
            instructions=instructions,
            state=state,
            num_inference_steps=10,
            deterministic_seed=0,
        )
        pred_wrong_instruction = model.predict_action(
            images=images,
            instructions=["do something unrelated"],
            state=state,
            num_inference_steps=10,
            deterministic_seed=0,
        )
        pred_zero_state = model.predict_action(
            images=images,
            instructions=instructions,
            state=torch.zeros_like(state),
            num_inference_steps=10,
            deterministic_seed=0,
        )
    wrong_instruction_shift = float(np.linalg.norm(pred_base - pred_wrong_instruction))
    zero_state_shift = float(np.linalg.norm(pred_base - pred_zero_state))
    _assert(
        wrong_instruction_shift > 2e-2,
        f"Instruction conditioning is too weak: shift={wrong_instruction_shift}",
    )
    _assert(
        zero_state_shift > 1e-2,
        f"State conditioning is too weak: shift={zero_state_shift}",
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="qwen35_v32_bridge_"))
    try:
        torch.save(
            {
                "action_head": normalize_action_head_state_dict(model.action_head.state_dict()),
                "action_q01": model.action_q01.detach().cpu(),
                "action_q99": model.action_q99.detach().cpu(),
                "local_action_q01": model.local_action_q01.detach().cpu(),
                "local_action_q99": model.local_action_q99.detach().cpu(),
                "state_q01": model.state_q01.detach().cpu(),
                "state_q99": model.state_q99.detach().cpu(),
                "step": 0,
            },
            tmp_dir / "action_head.pt",
        )
        OmegaConf.save(cfg, tmp_dir / "config.yaml")

        reloaded = build_model(cfg)
        payload = torch.load(tmp_dir / "action_head.pt", map_location="cpu", weights_only=False)
        missing, unexpected = reloaded.action_head.load_state_dict(payload["action_head"], strict=False)
        _assert(
            not missing and not unexpected,
            f"Round-trip checkpoint mismatch: missing={missing} unexpected={unexpected}",
        )
        reloaded.set_norm_stats(
            payload["action_q01"],
            payload["action_q99"],
            payload.get("local_action_q01", payload["action_q01"]),
            payload.get("local_action_q99", payload["action_q99"]),
            payload.get("state_q01", None),
            payload.get("state_q99", None),
        )
        reloaded = reloaded.to(device)
        reloaded.eval()

        with torch.no_grad():
            pred_a = model.predict_action(
                images=images,
                instructions=instructions,
                state=state,
                num_inference_steps=10,
                deterministic_seed=0,
            )
            pred_b = reloaded.predict_action(
                images=images,
                instructions=instructions,
                state=state,
                num_inference_steps=10,
                deterministic_seed=0,
            )
        roundtrip_error = float(np.max(np.abs(pred_a - pred_b)))
        _assert(roundtrip_error < 5e-3, f"Checkpoint round-trip prediction drift too large: {roundtrip_error}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("v32_bridge_unit_tests_ok", True)
    print("tap_strategy", cfg.action_head.tap_strategy)
    print("memory_norm_ratio", ratio)
    print("mean_norm_saturation_fraction", saturation_fraction)
    if ratio > float(cfg.action_head.get("memory_norm_ratio_limit", 8.0)):
        print(
            "memory_norm_ratio_warning",
            f"ratio {ratio:.4f} exceeded configured limit {float(cfg.action_head.get('memory_norm_ratio_limit', 8.0)):.4f}",
        )
    print("action_grad_norm", action_grad_norm)
    print("bridge_off_parity_max_error", parity_error)
    print("wrong_instruction_shift_l2", wrong_instruction_shift)
    print("zero_state_shift_l2", zero_state_shift)
    if tap_weight_entropy is not None:
        print("tap_weight_entropy", tap_weight_entropy)
        print("scalar_mix_gamma", scalar_mix_gamma)
        print("tap_weight_distribution", tap_weight_distribution)


if __name__ == "__main__":
    main()
