from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_libero import Qwen35VLALiberoPolicy, get_task_suites, make_libero_env, build_state


def arr(obs: dict, key: str, default) -> np.ndarray:
    return np.asarray(obs.get(key, default), dtype=np.float32)


def snapshot(obs: dict) -> dict:
    return {
        "eef_pos": arr(obs, "robot0_eef_pos", np.zeros(3)).tolist(),
        "eef_quat": arr(obs, "robot0_eef_quat", np.zeros(4)).tolist(),
        "gripper_qpos": arr(obs, "robot0_gripper_qpos", np.zeros(2)).tolist(),
        "joint_pos": arr(obs, "robot0_joint_pos", np.zeros(7)).tolist(),
        "state_vec": build_state(obs).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max_trace_steps", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    suite = get_task_suites([args.suite])[args.suite]
    task = suite.get_task(args.task_id)
    init_state = suite.get_task_init_states(args.task_id)[args.episode]

    policy = Qwen35VLALiberoPolicy(
        args.checkpoint_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        deterministic_seed=args.deterministic_seed,
    )

    env = make_libero_env(task)
    env.reset()
    env.set_init_state(init_state)
    policy.reset()

    obs = env.env._get_observations()
    records = []
    for step in range(args.max_trace_steps):
        before = snapshot(obs)
        obs_dict = {
            "image": obs["agentview_image"],
            "wrist_image": obs.get("robot0_eye_in_hand_image", obs["agentview_image"]),
            "state": build_state(obs),
        }
        action = policy.get_action(obs_dict, task.language)
        obs, reward, done, info = env.step(action)
        after = snapshot(obs)

        eef_before = np.asarray(before["eef_pos"], dtype=np.float32)
        eef_after = np.asarray(after["eef_pos"], dtype=np.float32)
        grip_before = np.asarray(before["gripper_qpos"], dtype=np.float32)
        grip_after = np.asarray(after["gripper_qpos"], dtype=np.float32)

        records.append(
            {
                "step": step,
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "eef_before": before["eef_pos"],
                "eef_after": after["eef_pos"],
                "eef_delta": (eef_after - eef_before).tolist(),
                "eef_delta_norm": float(np.linalg.norm(eef_after - eef_before)),
                "gripper_before": before["gripper_qpos"],
                "gripper_after": after["gripper_qpos"],
                "gripper_delta": (grip_after - grip_before).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(info.get("success", False)),
            }
        )
        if done or info.get("success", False):
            break

    env.close()

    result = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": task.language,
        "episode": args.episode,
        "max_trace_steps": args.max_trace_steps,
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
