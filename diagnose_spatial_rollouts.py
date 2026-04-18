from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def summarize_checkpoint(checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    keys = list(payload["action_head"].keys())
    print("=== CHECKPOINT ===")
    print("path:", checkpoint)
    print("num_action_head_keys:", len(keys))
    print("has_orig_mod_prefix:", any(k.startswith("_orig_mod.") for k in keys))
    print("first_12_keys:")
    for key in keys[:12]:
        print(" ", key)


def summarize_rollouts(rollout_base: Path) -> None:
    print("=== ROLLOUTS ===")
    for task_dir in sorted(rollout_base.iterdir()):
        traj_path = task_dir / "episode_00" / "trajectory.json"
        if not traj_path.exists():
            continue
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        actions = traj["actions"]
        eef_pos = traj["eef_pos"]
        dims = len(actions[0]) if actions else 0
        mean_abs = [
            sum(abs(action[i]) for action in actions) / len(actions)
            for i in range(dims)
        ] if actions else []
        max_abs = [
            max(abs(action[i]) for action in actions)
            for i in range(dims)
        ] if actions else []
        grip_delta_sum = sum(
            abs(actions[i][-1] - actions[i - 1][-1]) for i in range(1, len(actions))
        ) if actions else 0.0
        first_action = actions[0] if actions else []
        last_action = actions[-1] if actions else []
        print(task_dir.name)
        print(" success:", traj["success"], "num_actions:", traj["num_actions"])
        print(" first_action:", [round(x, 4) for x in first_action])
        print(" last_action :", [round(x, 4) for x in last_action])
        print(" mean_abs    :", [round(x, 4) for x in mean_abs])
        print(" max_abs     :", [round(x, 4) for x in max_abs])
        print(" grip_delta_sum:", round(grip_delta_sum, 4))
        print(" start_eef:", [round(x, 4) for x in eef_pos[0]])
        print(" end_eef  :", [round(x, 4) for x in eef_pos[-1]])
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-base", type=Path, required=True)
    args = parser.parse_args()

    summarize_checkpoint(args.checkpoint)
    summarize_rollouts(args.rollout_base)


if __name__ == "__main__":
    main()
