from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from train import build_dataloader, build_model, load_lora_adapters, normalize_action_head_state_dict


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Open-loop train-set eval for Qwen3.5-2B VLA")
    parser.add_argument("--config", type=str, required=True, help="Training YAML config")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=32, help="Eval batch size override")
    parser.add_argument("--num_workers", type=int, default=8, help="Eval dataloader workers")
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full train set")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_checkpoint_for_eval(model, checkpoint_dir: Path) -> int:
    payload = torch.load(checkpoint_dir / "action_head.pt", map_location="cpu", weights_only=False)
    normalized = normalize_action_head_state_dict(payload["action_head"])
    missing, unexpected = model.action_head.load_state_dict(normalized, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Action head checkpoint mismatch for {checkpoint_dir}: "
            f"missing={missing} unexpected={unexpected}"
        )

    if "action_q01" in payload and "action_q99" in payload:
        model.set_norm_stats(
            payload["action_q01"],
            payload["action_q99"],
            payload.get("local_action_q01", payload["action_q01"]),
            payload.get("local_action_q99", payload["action_q99"]),
            payload.get("state_q01", None),
            payload.get("state_q99", None),
        )

    lora_dir = checkpoint_dir / "lora_adapters"
    if lora_dir.exists() and hasattr(model.vlm, "model"):
        load_lora_adapters(model.vlm.model, lora_dir)

    return int(payload.get("step", 0))


def _predict_norm_actions(model, batch, num_inference_steps: int | None, deterministic_seed: int | None):
    actions = batch["actions"]
    state = batch["state"]
    history = batch.get("history", None)

    if model.action_head_type in {"layerwise_fm_v3", "layerwise_fm_v4", "pi05_qwen", "pi05_gemma_bridge"} and "images" in batch and "instructions" in batch:
        pred_raw = torch.from_numpy(
            model.predict_action(
                batch["images"],
                batch["instructions"],
                state=state,
                history=history,
                num_inference_steps=num_inference_steps,
                deterministic_seed=deterministic_seed,
            )
        ).to(actions.device)
        return model.normalize_actions(pred_raw)

    if "vlm_inputs" not in batch:
        pred_raw = torch.from_numpy(
            model.predict_action(
                batch["images"],
                batch["instructions"],
                state=state,
                history=history,
                num_inference_steps=num_inference_steps,
                deterministic_seed=deterministic_seed,
            )
        ).to(actions.device)
        return model.normalize_actions(pred_raw)

    vlm_inputs = batch["vlm_inputs"]
    if model.action_head_type == "pi05_qwen":
        action_head_dtype = model._action_head_dtype()
        norm_state = model.normalize_states(state).to(action_head_dtype)
        norm_history = model.normalize_history(history).to(action_head_dtype) if history is not None else None
        prefix_inputs = model.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
        return model.action_head.predict_action(
            language_model=model._language_model(),
            prefix_embeds=prefix_inputs["prefix_embeds"],
            prefix_attention_mask=prefix_inputs["attention_mask"],
            prefix_position_ids=prefix_inputs["position_ids"],
            state=norm_state,
            history=norm_history,
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
        )
    if model.action_head_type == "pi05_gemma_bridge":
        prompts = model._build_pi05_prompts(batch["instructions"], state)
        prefix_ctx = model.backbone.encode_prefix(
            images=batch.get("images"),
            prompts=prompts,
            vlm_inputs=vlm_inputs,
        )
        prefix_memory = model.backbone.get_prefix_memory(
            prefix_ctx,
            tap_strategy=model.action_config.tap_strategy,
        )
        norm_state = model.normalize_states(state).to(model._action_head_dtype()) if state is not None else None
        return model.action_head.predict_action(
            prefix_memory=prefix_memory["memory"],
            prefix_attention_mask=prefix_memory["memory_mask"],
            tap_ids=prefix_memory.get("tap_ids"),
            token_type_ids=prefix_memory.get("token_type_ids"),
            state=norm_state,
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
        )

    vlm_outputs = model.vlm(output_hidden_states=True, **vlm_inputs)
    hidden_states, layerwise_hidden_states = model._collect_vlm_hidden_states(vlm_outputs)
    text_embedding = model.vlm.extract_text_pooled(hidden_states, vlm_inputs)
    attention_mask = vlm_inputs.get("attention_mask", None)
    action_head_dtype = model._action_head_dtype()
    state = state.to(action_head_dtype)
    if history is not None:
        history = history.to(action_head_dtype)
    text_embedding = text_embedding.to(action_head_dtype)

    if model.action_head_type == "layerwise_fm":
        layerwise_hidden_states = [hs.to(action_head_dtype) for hs in layerwise_hidden_states]
        norm_actions = model.action_head.predict_action(
            vlm_hidden_states_list=layerwise_hidden_states,
            state=state,
            attention_mask=attention_mask,
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
        )
    elif model.action_head_type in {"layerwise_fm_v3", "layerwise_fm_v4"}:
        layerwise_hidden_states = [hs.to(action_head_dtype) for hs in layerwise_hidden_states]
        norm_actions = model.action_head.predict_action(
            vlm_hidden_states_list=layerwise_hidden_states,
            state=state,
            history=history,
            text_embedding=text_embedding,
            attention_mask=attention_mask,
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
        )
    else:
        hidden_states = hidden_states.to(action_head_dtype)
        if model.action_head_type == "pi05_qwen":
            prefix_inputs = model.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
            norm_state = model.normalize_states(state).to(action_head_dtype)
            norm_history = model.normalize_history(history).to(action_head_dtype) if history is not None else None
            norm_actions = model.action_head.predict_action(
                language_model=model._language_model(),
                prefix_embeds=prefix_inputs["prefix_embeds"],
                prefix_attention_mask=prefix_inputs["attention_mask"],
                prefix_position_ids=prefix_inputs["position_ids"],
                state=norm_state,
                history=norm_history,
                num_steps=num_inference_steps,
                deterministic_seed=deterministic_seed,
            )
        else:
            norm_actions = model.action_head.predict_action(
                vlm_hidden_states=hidden_states,
                state=state,
                attention_mask=attention_mask,
                num_steps=num_inference_steps,
                deterministic_seed=deterministic_seed,
            )
    return norm_actions


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.batch_size = int(args.batch_size)
    cfg.dataset.num_workers = int(args.num_workers)
    cfg.dataset.shuffle = False
    cfg.dataset.drop_last = False

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building model from %s", args.config)
    model = build_model(cfg)
    step = load_checkpoint_for_eval(model, checkpoint_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logger.info("Building dataloader")
    dataloader, dataset = build_dataloader(cfg)
    total_batches = len(dataloader)
    if args.max_batches and args.max_batches > 0:
        total_batches = min(total_batches, int(args.max_batches))

    logger.info(
        "Running open-loop eval | checkpoint_step=%d | windows=%d | batches=%d | batch_size=%d",
        step,
        len(dataset),
        total_batches,
        args.batch_size,
    )

    total_examples = 0
    total_elements = 0
    total_mae = 0.0
    total_mse = 0.0
    total_first_mae = 0.0
    total_first_mse = 0.0
    total_non_grip_mae = 0.0
    total_non_grip_mse = 0.0
    total_gripper_correct = 0.0
    total_gripper_count = 0
    per_dim_abs = None
    per_dim_sq = None

    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= total_batches:
            break

        batch = {
            key: ({k: v.to(device) for k, v in value.items()} if key == "vlm_inputs" else value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

        gt_raw = batch["actions"]
        with torch.no_grad():
            pred_norm = _predict_norm_actions(
                model,
                batch,
                num_inference_steps=args.num_inference_steps,
                deterministic_seed=args.deterministic_seed,
            )
        if (
            model.action_head_type == "pi05_qwen"
            and hasattr(model.action_head, "_uses_binary_gripper")
            and model.action_head._uses_binary_gripper()
        ):
            motion_norm = pred_norm[..., :6]
            gripper = pred_norm[..., 6:7]
            low = model.action_q01.to(device=motion_norm.device, dtype=motion_norm.dtype)
            high = model.action_q99.to(device=motion_norm.device, dtype=motion_norm.dtype)
            motion_raw = (motion_norm + 1.0) / 2.0 * (high[:6] - low[:6]) + low[:6]
            pred_raw = torch.cat([motion_raw, gripper], dim=-1)
        else:
            pred_raw = model.unnormalize_actions(pred_norm)

        # Compare gripper in execution semantics, preserving the signed
        # {-1, +1} convention used by LIBERO control.
        gt_exec = gt_raw.clone()
        if gt_exec.shape[-1] >= 7:
            gt_exec[..., 6] = torch.where(gt_exec[..., 6] > 0.0, 1.0, -1.0)

        diff = pred_raw - gt_exec
        abs_diff = diff.abs()
        sq_diff = diff.square()

        batch_examples = gt_raw.shape[0]
        batch_elements = gt_raw.numel()
        total_examples += batch_examples
        total_elements += batch_elements
        total_mae += abs_diff.sum().item()
        total_mse += sq_diff.sum().item()

        first_abs = (pred_raw[:, 0] - gt_exec[:, 0]).abs()
        first_sq = (pred_raw[:, 0] - gt_exec[:, 0]).square()
        total_first_mae += first_abs.sum().item()
        total_first_mse += first_sq.sum().item()

        non_grip_abs = abs_diff[..., :6]
        non_grip_sq = sq_diff[..., :6]
        total_non_grip_mae += non_grip_abs.sum().item()
        total_non_grip_mse += non_grip_sq.sum().item()

        pred_grip = torch.where(pred_raw[..., 6] > 0.0, 1.0, -1.0)
        gt_grip = torch.where(gt_exec[..., 6] > 0.0, 1.0, -1.0)
        total_gripper_correct += (pred_grip == gt_grip).float().sum().item()
        total_gripper_count += pred_grip.numel()

        dim_abs = abs_diff.sum(dim=(0, 1)).detach().cpu()
        dim_sq = sq_diff.sum(dim=(0, 1)).detach().cpu()
        if per_dim_abs is None:
            per_dim_abs = dim_abs
            per_dim_sq = dim_sq
        else:
            per_dim_abs += dim_abs
            per_dim_sq += dim_sq

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            sec_per_batch = elapsed / (batch_idx + 1)
            eta_sec = (total_batches - (batch_idx + 1)) * sec_per_batch
            logger.info(
                "batch %d/%d | running_mae=%.6f | running_mse=%.6f | eta=%.1f min",
                batch_idx + 1,
                total_batches,
                total_mae / total_elements,
                total_mse / total_elements,
                eta_sec / 60.0,
            )

    if total_elements == 0:
        raise RuntimeError("No evaluation batches were processed.")

    num_action_dims = int(gt_raw.shape[-1])
    num_chunk_dims = int(gt_raw[:, 0].numel() / gt_raw.shape[0])
    per_dim_mae = (per_dim_abs / (total_examples * gt_raw.shape[1])).tolist()
    per_dim_mse = (per_dim_sq / (total_examples * gt_raw.shape[1])).tolist()

    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": step,
        "config": str(args.config),
        "dataset_name": str(cfg.dataset.name),
        "num_windows": len(dataset),
        "batch_size": int(args.batch_size),
        "evaluated_batches": int(total_batches),
        "evaluated_examples": int(total_examples),
        "num_inference_steps": int(args.num_inference_steps) if args.num_inference_steps is not None else int(cfg.action_head.num_inference_steps),
        "deterministic_seed": int(args.deterministic_seed) if args.deterministic_seed is not None else None,
        "raw_action_mae": total_mae / total_elements,
        "raw_action_mse": total_mse / total_elements,
        "first_step_mae": total_first_mae / (total_examples * num_action_dims),
        "first_step_mse": total_first_mse / (total_examples * num_action_dims),
        "non_gripper_mae": total_non_grip_mae / (total_examples * gt_raw.shape[1] * 6),
        "non_gripper_mse": total_non_grip_mse / (total_examples * gt_raw.shape[1] * 6),
        "gripper_accuracy": total_gripper_correct / max(total_gripper_count, 1),
        "per_dim_mae": per_dim_mae,
        "per_dim_mse": per_dim_mse,
        "elapsed_min": (time.time() - start_time) / 60.0,
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
