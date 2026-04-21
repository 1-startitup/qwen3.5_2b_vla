"""Identify which params have no grad after a forward+backward.
Expected: lm_head and possibly unused vision tower bits (when vlm_loss_weight=0).
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, torch
from PIL import Image
from model import Qwen35PI05VLA, QwenPI05ActionConfig

cfg = QwenPI05ActionConfig(hidden_dim=128, num_layers=4, action_dim=7, action_horizon=10,
                           chunk_size=10, state_dim=8, empty_cameras=1, tokenizer_max_length=320)
model = Qwen35PI05VLA(vlm_model_id="/tmp/tiny_qwen35_vl_random", action_config=cfg,
                     attn_implementation="eager").cuda()
model.set_norm_stats(-torch.ones(7), torch.ones(7), -torch.ones(8), torch.ones(8))

rng = np.random.default_rng(0)
images = [[Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)) for _ in range(2)]]
out = model(images=images, instructions=["pick up"],
            actions=torch.randn(1, 10, 7, device="cuda") * 0.3)
out["loss"].backward()

print("=== params with no grad ===")
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"  {name:70s}  shape={tuple(p.shape)}")
print()
print("=== params with all-zero grad ===")
for name, p in model.named_parameters():
    if p.grad is not None and not p.grad.any():
        print(f"  {name:70s}  shape={tuple(p.shape)}")
