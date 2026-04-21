"""Smoke-test the full training path with real Qwen3.5-2B weights.

16GB VRAM budget: full-FT Qwen3.5-2B (bf16) + gemma_300m action expert + optimizer
states is tight. Enable gradient checkpointing. batch_size=1, action_horizon=4.
"""
import sys, time, os; sys.path.insert(0, ".")
import numpy as np, torch
from PIL import Image

from model import Qwen35PI05VLA, QwenPI05ActionConfig

MODEL_PATH = "/home/labuser/models/Qwen3.5-2B"

def main():
    torch.manual_seed(0)

    cfg = QwenPI05ActionConfig(
        hidden_dim=1024, num_layers=18,
        action_dim=7, action_horizon=4, chunk_size=4,
        max_action_dim=32, state_dim=8,
        empty_cameras=1, tokenizer_max_length=320,
        num_inference_steps=2,  # fewer sampler steps to save time
    )
    print("[1/5] building Qwen35PI05VLA with real Qwen3.5-2B ...")
    t0 = time.time()
    model = Qwen35PI05VLA(
        vlm_model_id=MODEL_PATH,
        action_config=cfg,
        freeze_vision_encoder=True,
        freeze_vlm=True,     # 5080 has 16GB; full-FT 2B+AdamW needs ~18 GB. Freeze for debug.
        attn_implementation="sdpa",
    ).cuda()
    # Grad checkpointing: essential for 16GB card.
    if hasattr(model.vlm.model, "gradient_checkpointing_enable"):
        model.vlm.model.gradient_checkpointing_enable()
        if hasattr(model.vlm.model, "config"):
            model.vlm.model.config.use_cache = False
        print("  gradient checkpointing enabled")
    print(f"  build time: {time.time()-t0:.1f}s")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params : {total/1e9:.2f}B")
    print(f"  trainable    : {trainable/1e6:.0f}M")
    print(f"  after load, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model.set_norm_stats(-torch.ones(7), torch.ones(7), -torch.ones(8), torch.ones(8))

    print("\n[2/5] dummy batch ...")
    rng = np.random.default_rng(0)
    images = [[Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]]
    instructions = ["pick up the red block"]
    actions = torch.randn(1, cfg.action_horizon, cfg.action_dim, device="cuda") * 0.3

    print("\n[3/5] forward ...")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=1e-10, eps=1e-8)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(images=images, instructions=instructions, actions=actions)
    loss = out["loss"]
    print(f"  loss: {loss.item():.4f}  t={time.time()-t0:.1f}s")
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    print("\n[4/5] backward + step ...")
    t0 = time.time()
    loss.backward()
    print(f"  backward: {time.time()-t0:.1f}s  peak mem after bwd: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    n_with = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_nan = sum(1 for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
    n_none = sum(1 for p in model.parameters() if p.grad is None)
    print(f"  params: finite={n_with}  nan/inf={n_nan}  no-grad={n_none}")
    t0 = time.time()
    opt.step()
    opt.zero_grad()
    print(f"  optimizer step: {time.time()-t0:.1f}s  peak mem after step: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print("\n[5/5] predict_action (eval) ...")
    model.eval()
    t0 = time.time()
    pred = model.predict_action(
        images=images, instructions=instructions,
        num_inference_steps=cfg.num_inference_steps, deterministic_seed=42,
    )
    print(f"  predicted: shape={pred.shape} dtype={pred.dtype} t={time.time()-t0:.1f}s")
    assert pred.shape == (1, cfg.chunk_size, cfg.action_dim)
    assert np.all(np.isfinite(pred))

    print(f"\npeak GPU memory total: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(">>> REAL Qwen3.5-2B smoke test PASSED <<<")

if __name__ == "__main__":
    main()
