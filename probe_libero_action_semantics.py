from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_libero import get_task_suites, make_libero_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def arr(obs: dict, key: str, default) -> np.ndarray:
    return np.asarray(obs.get(key, default), dtype=np.float32)


def collect_obs(obs: dict) -> dict:
    return {
        "eef_pos": arr(obs, "robot0_eef_pos", np.zeros(3)).tolist(),
        "eef_quat": arr(obs, "robot0_eef_quat", np.zeros(4)).tolist(),
        "gripper_qpos": arr(obs, "robot0_gripper_qpos", np.zeros(2)).tolist(),
        "joint_pos": arr(obs, "robot0_joint_pos", np.zeros(7)).tolist(),
    }


def diff_obs(before: dict, after: dict) -> dict:
    out = {}
    for key in ["eef_pos", "gripper_qpos", "joint_pos"]:
        b = np.asarray(before[key], dtype=np.float32)
        a = np.asarray(after[key], dtype=np.float32)
        out[f"{key}_delta"] = (a - b).tolist()
    return out


def find_dataset_action(dataset_name: str, task_text: str, root: str) -> list[float] | None:
    ds = LeRobotDataset(repo_id=dataset_name, root=root)
    for i in range(len(ds)):
        sample = ds[i]
        if sample.get("task", "").strip().lower() == task_text.strip().lower():
            return np.asarray(sample["action"], dtype=np.float32).tolist()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="lerobot/libero_spatial_image")
    parser.add_argument("--data_root", type=str, default="/home/frankkkz/datasets")
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    suite = get_task_suites([args.suite])[args.suite]
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env = make_libero_env(task)
    env.reset()
    env.set_init_state(init_states[args.episode])

    obs0 = env.env._get_observations()
    action_spec = None
    try:
        low, high = env.action_spec
        action_spec = {
            "low": np.asarray(low, dtype=np.float32).tolist(),
            "high": np.asarray(high, dtype=np.float32).tolist(),
        }
    except Exception:
        pass

    probes = []
    zero = np.zeros(7, dtype=np.float32)
    dataset_action = find_dataset_action(args.dataset_name, task.language, args.data_root)
    if dataset_action is not None:
        probes.append(("dataset_sample", np.asarray(dataset_action, dtype=np.float32)))

    probes.append(("zero", zero.copy()))
    for dim in range(7):
        plus = zero.copy()
        minus = zero.copy()
        plus[dim] = args.amplitude
        minus[dim] = -args.amplitude
        probes.append((f"dim{dim}_plus", plus))
        probes.append((f"dim{dim}_minus", minus))

    results = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": task.language,
        "episode": args.episode,
        "action_spec": action_spec,
        "initial_obs": collect_obs(obs0),
        "probes": [],
    }

    for name, action in probes:
        env.reset()
        env.set_init_state(init_states[args.episode])
        before = collect_obs(env.env._get_observations())
        obs1, reward, done, info = env.step(action)
        after = collect_obs(obs1)
        probe = {
            "name": name,
            "action": action.tolist(),
            "before": before,
            "after": after,
            "delta": diff_obs(before, after),
            "reward": float(reward),
            "done": bool(done),
            "success": bool(info.get("success", False)),
        }
        results["probes"].append(probe)

    env.close()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
