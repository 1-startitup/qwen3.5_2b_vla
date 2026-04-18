from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from data.libero_dataset import LiberoVLADataset
from eval_libero import get_task_suites


def quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quat of shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    x, y, z, w = q
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.asarray([x, y, z], dtype=np.float64) / s
    return (axis * angle).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/home/frankkkz/datasets")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    suite = get_task_suites([args.suite])[args.suite]
    task = suite.get_task(args.task_id)
    target_task = task.language.strip().lower()

    dataset = LiberoVLADataset(
        dataset_name=cfg.dataset.name,
        data_root=args.data_root,
        action_horizon=cfg.action_head.action_horizon,
        image_size=tuple(cfg.dataset.image_size),
        action_type=cfg.dataset.action_type,
        vlm_model_id=cfg.model.vlm_model_id,
        norm_stats_sample_size=cfg.dataset.get("norm_stats_sample_size", 20000),
    )

    matched_episode = None
    for ep in dataset.dataset.meta.episodes:
        tasks = ep.get("tasks", [])
        if tasks and tasks[0].strip().lower() == target_task:
            matched_episode = ep
            break
    if matched_episode is None:
        raise RuntimeError(f"No matching training episode found for task: {task.language}")

    episode_start = int(matched_episode["dataset_from_index"])
    episode_end = int(matched_episode["dataset_to_index"])
    expert_states = []
    expert_actions = []
    for idx in range(episode_start, episode_end):
        frame = dataset.dataset[idx]
        expert_states.append(np.asarray(frame["observation.state"], dtype=np.float32))
        expert_actions.append(np.asarray(frame["action"], dtype=np.float32))
    expert_states = np.stack(expert_states, axis=0)
    expert_actions = np.stack(expert_actions, axis=0)

    expert_gripper = expert_actions[:, 6]
    expert_switches = []
    prev = float(expert_gripper[0])
    for off, val in enumerate(expert_gripper[1:], start=1):
        val = float(val)
        if val != prev:
            expert_switches.append({"offset": off, "from": prev, "to": val})
            prev = val

    traj = json.loads(Path(args.trajectory).read_text())
    rollout_actions = np.asarray(traj["actions"], dtype=np.float32)
    rollout_eef_pos = np.asarray(traj["eef_pos"], dtype=np.float32)
    rollout_eef_quat = np.asarray(traj["eef_quat"], dtype=np.float32)
    rollout_states = []
    for pos, quat in zip(rollout_eef_pos[:-1], rollout_eef_quat[:-1]):
        axis_angle = quat_xyzw_to_axis_angle(quat)
        rollout_states.append(np.concatenate([pos, axis_angle], axis=0))
    rollout_states = np.stack(rollout_states, axis=0)
    expert_pose6 = expert_states[:, :6]

    rollout_gripper = rollout_actions[:, 6]
    rollout_switches = []
    prev = float(rollout_gripper[0])
    for step, val in enumerate(rollout_gripper[1:], start=1):
        val = float(val)
        if val != prev:
            pose6 = rollout_states[step]
            dists = np.linalg.norm(expert_pose6 - pose6[None, :], axis=1)
            nearest_off = int(np.argmin(dists))
            rollout_switches.append(
                {
                    "step": step,
                    "from": prev,
                    "to": val,
                    "rollout_pose6": pose6.tolist(),
                    "nearest_expert_offset": nearest_off,
                    "nearest_expert_dist_l2": float(dists[nearest_off]),
                    "nearest_expert_gripper": float(expert_gripper[nearest_off]),
                    "nearest_expert_action": expert_actions[nearest_off].tolist(),
                    "expert_grasp_offset": next(
                        (
                            sw["offset"]
                            for sw in expert_switches
                            if sw["from"] < 0 and sw["to"] > 0
                        ),
                        None,
                    ),
                    "dist_to_expert_grasp_pose": float(
                        np.linalg.norm(expert_pose6[next(sw["offset"] for sw in expert_switches if sw["from"] < 0 and sw["to"] > 0)] - pose6)
                    )
                    if any(sw["from"] < 0 and sw["to"] > 0 for sw in expert_switches)
                    else None,
                }
            )
            prev = val

    result = {
        "task_name": task.name,
        "instruction": task.language,
        "matched_episode_index": int(matched_episode["episode_index"]),
        "expert_episode_length": int(len(expert_actions)),
        "expert_switches": expert_switches,
        "rollout_num_steps": int(len(rollout_actions)),
        "rollout_switches": rollout_switches,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
