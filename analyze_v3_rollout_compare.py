import argparse
import json
from pathlib import Path

import numpy as np


def summarize_rollout(ep_dir: Path) -> dict:
    actions = np.load(ep_dir / "actions.npy")
    eef = np.load(ep_dir / "eef_positions.npy")
    traj = json.loads((ep_dir / "trajectory.json").read_text(encoding="utf-8"))

    diffs = np.diff(eef, axis=0) if len(eef) > 1 else np.zeros((0, 3), dtype=np.float32)
    path_length = float(np.linalg.norm(diffs, axis=-1).sum()) if len(diffs) else 0.0
    displacement = float(np.linalg.norm(eef[-1] - eef[0])) if len(eef) else 0.0
    z_drop = float(eef[0, 2] - eef[:, 2].min()) if len(eef) else 0.0

    motion = actions[:, :6]
    gripper = actions[:, 6]
    action_norms = np.linalg.norm(motion, axis=-1)
    gripper_sign = np.where(gripper > 0, 1.0, -1.0)
    switch_count = int(np.sum(gripper_sign[1:] != gripper_sign[:-1])) if len(gripper_sign) > 1 else 0

    close_steps = np.where(gripper_sign > 0)[0]
    first_close_step = int(close_steps[0]) if len(close_steps) else None
    close_fraction = float((gripper_sign > 0).mean()) if len(gripper_sign) else 0.0

    longest_close_run = 0
    current_run = 0
    for val in gripper_sign:
        if val > 0:
            current_run += 1
            longest_close_run = max(longest_close_run, current_run)
        else:
            current_run = 0

    step_rewards = []
    success = False
    if isinstance(traj, dict):
        success = bool(traj.get("success", False))
        for step in traj.get("steps", []):
            if "reward" in step:
                step_rewards.append(float(step["reward"]))

    return {
        "episode_dir": str(ep_dir),
        "success": success,
        "num_steps": int(actions.shape[0]),
        "path_length": path_length,
        "displacement": displacement,
        "z_drop": z_drop,
        "first_action_norm": float(action_norms[0]) if len(action_norms) else 0.0,
        "mean_action_norm": float(action_norms.mean()) if len(action_norms) else 0.0,
        "max_action_norm": float(action_norms.max()) if len(action_norms) else 0.0,
        "switch_count": switch_count,
        "first_close_step": first_close_step,
        "close_fraction": close_fraction,
        "longest_close_run": int(longest_close_run),
        "reward_max": max(step_rewards) if step_rewards else 0.0,
        "start_pos": eef[0].tolist() if len(eef) else None,
        "end_pos": eef[-1].tolist() if len(eef) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_root", required=True)
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = {
        "current_root": args.current_root,
        "baseline_root": args.baseline_root,
        "tasks": {},
    }

    for task_dir_name in args.tasks:
        current_ep = Path(args.current_root) / task_dir_name / "episode_00"
        baseline_ep = Path(args.baseline_root) / task_dir_name / "episode_00"
        current = summarize_rollout(current_ep)
        baseline = summarize_rollout(baseline_ep)
        delta = {}
        for key in [
            "path_length",
            "displacement",
            "z_drop",
            "first_action_norm",
            "mean_action_norm",
            "max_action_norm",
            "switch_count",
            "close_fraction",
            "longest_close_run",
            "reward_max",
        ]:
            delta[key] = current[key] - baseline[key]
        out["tasks"][task_dir_name] = {
            "current": current,
            "baseline": baseline,
            "delta": delta,
        }

    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
