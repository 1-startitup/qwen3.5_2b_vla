"""Run the actual train.py main loop for ~5 steps with a synthetic dataloader.

Goal: prove the full training loop (build_model → build_optimizer → scheduler →
Accelerator.prepare → forward/backward/step → log → save_checkpoint) runs without
errors. Uses batch_size=1 on the 5080 (16GB); freezes VLM since full-FT 2.74B +
AdamW optim states does not fit per-GPU.

Monkeys-patch `train.build_dataloader` to return a synthetic dataset that
satisfies the batch contract train.py expects:
  { "images": List[List[PIL]], "instructions": List[str], "actions": Tensor(B,H,Da),
    "state": Tensor(B,Ds) }
and an object carrying `.norm_stats` (used once at startup).
"""
import sys, os
# Run from the repo root (`python smoke/smoke_train_loop.py`) OR from smoke/ —
# either way, make the repo root importable so `import train` and `import model` work.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
        if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "smoke" \
        else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# --- synthetic dataset ------------------------------------------------------
class SyntheticLiberoDataset(Dataset):
    def __init__(self, n: int, action_horizon: int, action_dim: int, state_dim: int):
        self.n = int(n)
        self.H = int(action_horizon)
        self.Da = int(action_dim)
        self.Ds = int(state_dim)
        rng = np.random.default_rng(0)
        self._img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        # norm_stats dict mirrors libero_dataset.norm_stats output
        self.norm_stats = {
            "q01": -torch.ones(self.Da),
            "q99": torch.ones(self.Da),
            "local_q01": -torch.ones(self.Da),
            "local_q99": torch.ones(self.Da),
            "state_q01": -torch.ones(self.Ds),
            "state_q99": torch.ones(self.Ds),
        }

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "images": [Image.fromarray(self._img), Image.fromarray(self._img)],
            "instructions": "pick up the red block",
            "actions": torch.randn(self.H, self.Da) * 0.3,
            "state": torch.zeros(self.Ds),
        }


def collate(batch):
    return {
        "images":       [b["images"]       for b in batch],
        "instructions": [b["instructions"] for b in batch],
        "actions":      torch.stack([b["actions"] for b in batch]),
        "state":        torch.stack([b["state"]   for b in batch]),
    }


def fake_build_dataloader(cfg):
    ds = SyntheticLiberoDataset(
        n=256,
        action_horizon=int(cfg.action_head.action_horizon),
        action_dim=int(cfg.action_head.action_dim),
        state_dim=int(cfg.action_head.state_dim),
    )
    loader = DataLoader(
        ds, batch_size=int(cfg.dataset.batch_size), shuffle=False,
        num_workers=0, collate_fn=collate, drop_last=True,
    )
    return loader, ds


# --- monkey-patch train.build_dataloader and invoke main ---------------------
import train
train.build_dataloader = fake_build_dataloader

if __name__ == "__main__":
    train.main()
