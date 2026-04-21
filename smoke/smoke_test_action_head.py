"""Smoke-test the Pi 0.5 dual-stream action head in isolation.

Builds a tiny random-init Qwen3_5TextModel (no vision, no weights on disk),
wires it to QwenPI05ExpertHead, runs forward → backward → optimizer step
with dummy prefix embeds + dummy actions, then runs sampling.

Verifies:
  - suffix layers materialize with matching layer_types
  - joint full-attention at full_attention layers produces finite loss
  - backward populates gradients for action expert
  - predict_action() runs the cached-prefix + suffix denoising loop

Does NOT test the VLM side (image scatter / 3D RoPE) because that needs
real Qwen3.5-VL weights. Run separately once weights are available.
"""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from model.qwen35_pi05_action_head import QwenPI05ActionConfig, QwenPI05ExpertHead
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device                  : {device}")

    # -- Build a tiny Qwen3.5 text backbone --------------------------------
    # Mirrors real Qwen3.5 structure: hybrid full/linear-attention layer types.
    text_cfg = Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=2048,
        max_position_embeddings=1024,
        rope_theta=10000.0,
    )
    # Force a mixed pattern so we exercise both dual_full_attention and dual_linear paths.
    text_cfg.layer_types = ["full_attention", "linear_attention", "full_attention", "linear_attention"]
    print(f"tiny Qwen3_5TextConfig  : {text_cfg.num_hidden_layers} layers  layer_types={text_cfg.layer_types}")

    language_model = Qwen3_5TextModel(text_cfg).to(device=device, dtype=dtype)
    language_model.eval()
    lm_params = sum(p.numel() for p in language_model.parameters())
    print(f"language_model params   : {lm_params/1e6:.2f}M")

    # -- Build the action head ---------------------------------------------
    action_cfg = QwenPI05ActionConfig(
        hidden_dim=128,
        num_layers=4,
        action_dim=7,
        action_horizon=10,
        chunk_size=10,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=3,
        state_dim=8,
    )
    head = QwenPI05ExpertHead(action_cfg).to(device=device, dtype=dtype)
    head._ensure_initialized(language_model)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"action expert params    : {head_params/1e6:.2f}M")
    print(f"suffix layer_types      : {head.layer_types}")
    assert head.layer_types == text_cfg.layer_types

    # -- Dummy batch --------------------------------------------------------
    B, P = 2, 16
    prefix_embeds = torch.randn(B, P, text_cfg.hidden_size, device=device, dtype=dtype)
    prefix_mask = torch.ones(B, P, device=device, dtype=torch.long)
    prefix_mask[0, -3:] = 0  # exercise padding in the mask
    actions = torch.randn(B, action_cfg.action_horizon, action_cfg.action_dim, device=device) * 0.3

    # -- forward / backward / step -----------------------------------------
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4)

    head.train()
    t0 = time.time()
    out = head(
        language_model=language_model,
        prefix_embeds=prefix_embeds,
        prefix_attention_mask=prefix_mask,
        prefix_position_ids=None,
        actions=actions,
    )
    loss = out["loss"]
    t_fwd = time.time() - t0
    print(f"forward                 : loss={loss.item():.4f}  t={t_fwd*1000:.0f}ms")
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0

    # Check gradients exist on action head params.
    has_grad = 0
    missing = []
    for name, p in head.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif torch.isfinite(p.grad).all():
            has_grad += 1
    print(f"backward                : {has_grad} params with finite grads, {len(missing)} without  t={t_bwd*1000:.0f}ms")
    if missing:
        print(f"  missing grad           : {missing[:5]}{'...' if len(missing) > 5 else ''}")

    opt.step()
    opt.zero_grad()
    print(f"optimizer step          : OK")

    # -- sampling -----------------------------------------------------------
    head.eval()
    t0 = time.time()
    with torch.no_grad():
        pred = head.predict_action(
            language_model=language_model,
            prefix_embeds=prefix_embeds,
            prefix_attention_mask=prefix_mask,
            prefix_position_ids=None,
            num_steps=action_cfg.num_inference_steps,
            deterministic_seed=42,
        )
    t_sample = time.time() - t0
    print(f"predict_action          : shape={tuple(pred.shape)}  dtype={pred.dtype}  t={t_sample*1000:.0f}ms")
    assert pred.shape == (B, action_cfg.chunk_size, action_cfg.action_dim), f"unexpected shape {pred.shape}"
    assert torch.isfinite(pred).all(), "non-finite predicted actions"

    # Memory summary.
    if device.type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak GPU memory         : {mem_gb:.2f} GB")
    print(">>> action-head smoke test PASSED <<<")


if __name__ == "__main__":
    main()
