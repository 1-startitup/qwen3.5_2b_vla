from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    ds = LeRobotDataset(repo_id="lerobot/libero_spatial_image", root="/home/frankkkz/datasets")
    sample = ds[0]
    print("num_frames", len(ds))
    print("sample_keys", sorted(sample.keys()))
    print("task", sample.get("task"))
    print("action_shape", np.asarray(sample["action"]).shape)
    print("state_shape", np.asarray(sample["observation.state"]).shape)
    print("action0", np.asarray(sample["action"]).tolist())
    print("state0", np.asarray(sample["observation.state"]).tolist())

    if hasattr(ds, "meta"):
        meta = ds.meta
        print("meta_total_episodes", getattr(meta, "total_episodes", None))
        episodes = getattr(meta, "episodes", None)
        if episodes:
            print("episode0", json.dumps(episodes[0], indent=2))


if __name__ == "__main__":
    main()
