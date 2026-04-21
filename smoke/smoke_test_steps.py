"""Run 5 training steps on a fixed dummy batch — loss should decrease, time_mlp grads
should become non-zero after step 1."""
import sys; sys.path.insert(0, ".")
import numpy as np, torch
from PIL import Image
from model import Qwen35PI05VLA, QwenPI05ActionConfig

cfg = QwenPI05ActionConfig(hidden_dim=128, num_layers=4, action_dim=7, action_horizon=10,
                           chunk_size=10, state_dim=8, empty_cameras=1, tokenizer_max_length=320)
model = Qwen35PI05VLA(vlm_model_id="/tmp/tiny_qwen35_vl_random", action_config=cfg,
                     attn_implementation="eager").cuda()
model.set_norm_stats(-torch.ones(7), torch.ones(7), -torch.ones(8), torch.ones(8))

rng = np.random.default_rng(0)
images = [[Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)) for _ in range(2)] for _ in range(2)]
instructions = ["pick up the red block", "place the cup on the plate"]
actions = torch.randn(2, 10, 7, device="cuda") * 0.3

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # bigger LR to see movement fast
torch.manual_seed(1)  # fix time noise

print(f"{'step':>4} {'loss':>8} {'time_mlp_in.grad_nrm':>22} {'adarms0.dense.grad_nrm':>25}")
for step in range(6):
    out = model(images=images, instructions=instructions, actions=actions)
    loss = out["loss"]
    opt.zero_grad()
    loss.backward()
    g_time = model.action_head.time_mlp_in.weight.grad
    g_adarms = model.action_head.input_adarms[0].dense.weight.grad
    print(f"{step:>4} {loss.item():>8.4f} {g_time.norm().item() if g_time is not None else 'None':>22} "
          f"{g_adarms.norm().item() if g_adarms is not None else 'None':>25}")
    opt.step()
