"""Evaluate a Qwen3.5-VL + Pi 0.5 checkpoint on LIBERO suites.

Pi 0.5 libero exec strategy: predict a 10-step action chunk, execute the first
`replan_horizon` steps, then re-predict. No temporal-ensemble smoothing, no
gripper hysteresis, no instruction rephrasing — strict Pi 0.5 contract.

Usage:
    python eval_libero.py --checkpoint_dir checkpoints/.../checkpoint-30000 \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --n_episodes 10 --max_steps 300 --replan_horizon 5
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from train import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------- LIBERO env helpers ----------------------------------------------

def make_libero_env(task, camera_height: int = 256, camera_width: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_height, camera_widths=camera_width)
    env.seed(0)
    return env


def get_task_suites(suite_names):
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    return {name: benchmark_dict[name]() for name in suite_names}


def quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    sin_half = float(np.linalg.norm(quat[:3]))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, float(quat[3]))
    return ((quat[:3] / sin_half) * angle).astype(np.float32)


def build_state(obs: dict) -> np.ndarray:
    gripper = np.asarray(obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]
    eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(obs.get("robot0_eef_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    return np.concatenate([eef_pos, quat_xyzw_to_axis_angle(eef_quat), gripper]).astype(np.float32)


# ---------- policy -----------------------------------------------------------

class PI05Policy:
    """Closed-loop Pi 0.5 policy: predict chunk, execute replan_horizon steps, replan."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        num_inference_steps: int = 10,
        replan_horizon: int = 5,
        deterministic_seed: int = 0,
    ):
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_inference_steps = int(num_inference_steps)
        self.replan_horizon = int(replan_horizon)
        self.deterministic_seed = deterministic_seed

        cfg_path = self.checkpoint_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing config.yaml in {self.checkpoint_dir}")
        self.cfg = OmegaConf.load(cfg_path)

        self.model = build_model(self.cfg)
        payload = torch.load(self.checkpoint_dir / "action_head.pt", map_location="cpu", weights_only=False)
        self.model.action_head.load_state_dict(payload["action_head"], strict=False)
        self.model.set_norm_stats(
            payload["action_q01"], payload["action_q99"],
            payload.get("state_q01"), payload.get("state_q99"),
        )

        # Prefer EMA weights at eval (Pi 0.5 libero uses ema_decay=0.999, and
        # ema_params replace params at eval time; see openpi checkpoints.py:146).
        ema_path = self.checkpoint_dir / "ema.pt"
        if ema_path.exists():
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
            loaded, skipped = 0, 0
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in ema_state:
                        p.data.copy_(ema_state[name].to(p.device, p.dtype))
                        loaded += 1
                    else:
                        skipped += 1
            logger.info(f"Loaded EMA weights from {ema_path}: {loaded} params overwritten, {skipped} kept from base ckpt")

        self.model = self.model.to(self.device).eval()

        self._chunk: np.ndarray | None = None
        self._idx = 0

    def reset(self):
        self._chunk = None
        self._idx = 0

    @torch.no_grad()
    def _predict_chunk(self, obs: dict, instruction: str) -> np.ndarray:
        main = Image.fromarray(obs["image"]).convert("RGB")
        wrist = Image.fromarray(obs.get("wrist_image", obs["image"])).convert("RGB")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = self.model.predict_action(
                images=[[main, wrist]],
                instructions=[instruction],
                num_inference_steps=self.num_inference_steps,
                deterministic_seed=self.deterministic_seed,
            )
        return np.asarray(chunk, dtype=np.float32)[0]

    def get_action(self, obs: dict, instruction: str) -> np.ndarray:
        if self._chunk is None or self._idx >= self.replan_horizon:
            self._chunk = self._predict_chunk(obs, instruction)
            self._idx = 0
        action = self._chunk[self._idx]
        self._idx += 1
        return action


# ---------- rollout ----------------------------------------------------------

def evaluate_suite(policy: PI05Policy, suite_name: str, suite, n_episodes: int, max_steps: int,
                   camera_key: str = "agentview_image", wrist_key: str = "robot0_eye_in_hand_image"):
    per_task = []
    results = {"suite": suite_name, "tasks": {}}

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        instruction = task.language
        env = make_libero_env(task)
        init_states = suite.get_task_init_states(task_id)
        episodes = min(n_episodes, len(init_states))
        successes = 0

        for ep in range(episodes):
            policy.reset()
            env.reset()
            env.set_init_state(init_states[ep])
            obs = env.env._get_observations()
            success = False
            for step in range(max_steps):
                obs_dict = {
                    "image": obs[camera_key],
                    "wrist_image": obs.get(wrist_key, obs[camera_key]),
                    "state": build_state(obs),
                }
                action = policy.get_action(obs_dict, instruction)
                obs, _reward, done, info = env.step(action)
                if done or info.get("success", False):
                    success = bool(done or info.get("success", False))
                    break
            successes += int(success)
            logger.info("[%s] task %02d ep %d: %s (%d steps)",
                        suite_name, task_id, ep, "SUCCESS" if success else "FAIL", step + 1)
        env.close()

        sr = successes / max(episodes, 1)
        results["tasks"][task.name] = {"task_id": task_id, "success_rate": sr, "n": episodes}
        per_task.append(sr)
        logger.info("[%s] task %02d success: %.1f%% (%d/%d)", suite_name, task_id, 100 * sr, successes, episodes)

    results["average_success_rate"] = float(np.mean(per_task)) if per_task else 0.0
    logger.info("%s average: %.1f%%", suite_name, 100 * results["average_success_rate"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--suites", type=str, nargs="+",
                        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_horizon", type=int, default=5,
                        help="Pi 0.5 libero exec strategy: predict 10-step chunk, exec 5, replan.")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    policy = PI05Policy(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        replan_horizon=args.replan_horizon,
        deterministic_seed=args.deterministic_seed,
    )

    suites = get_task_suites(args.suites)
    all_results = {
        name: evaluate_suite(policy, name, suite, n_episodes=args.n_episodes, max_steps=args.max_steps)
        for name, suite in suites.items()
    }

    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print("\n" + "=" * 60)
    print("Pi 0.5 LIBERO eval")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name:25s} avg {100 * res['average_success_rate']:.1f}%")
    print(f"results saved to {args.output}")


if __name__ == "__main__":
    main()
