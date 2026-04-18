"""
Evaluate Qwen3.5-2B VLA checkpoints on LIBERO suites.

This loader matches the current checkpoint structure:
  - checkpoint-XXXX/action_head.pt
  - checkpoint-XXXX/lora_adapters/
  - checkpoint-XXXX/config.yaml
"""

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from train import (
    build_model,
    filter_optional_action_head_load_issues,
    load_lora_adapters,
    normalize_action_head_state_dict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


INSTRUCTION_REPHRASINGS = {
    "pick up": ["grab", "lift", "take", "get"],
    "put": ["place", "set", "move", "position"],
    "open": ["pull open", "unfasten", "unlatch"],
    "close": ["shut", "push closed", "seal"],
    "turn on": ["switch on", "activate", "power on"],
    "turn off": ["switch off", "deactivate", "power off"],
    "push": ["shove", "slide", "nudge"],
    "the": ["a", "that", "this"],
}

HIGH_SCORE_REPLAN_SUBSTRINGS = {
    # These layouts benefited from more frequent replanning in our deterministic evals.
    "next to the ramekin": 10,
    "on the cookie box": 10,
    "next to the plate": 10,
}


def rephrase_instruction(instruction: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    result = instruction.lower()
    for original, replacements in INSTRUCTION_REPHRASINGS.items():
        if original in result:
            result = result.replace(original, rng.choice(replacements), 1)
    return result


class Qwen35VLALiberoPolicy:
    """Checkpoint-aware wrapper for closed-loop LIBERO evaluation."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        chunk_size: int | None = None,
        num_inference_steps: int | None = None,
        deterministic_seed: int | None = None,
        temporal_ensemble_decay: float | None = None,
        gripper_hysteresis_steps: int = 0,
        gripper_min_hold_steps: int = 0,
        replan_horizon: int | None = None,
        adaptive_highscore_replan: bool = False,
        action_ema_alpha: float = 0.0,
    ):
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_inference_steps = num_inference_steps
        self.deterministic_seed = deterministic_seed
        self.temporal_ensemble_decay = temporal_ensemble_decay
        self.gripper_hysteresis_steps = int(gripper_hysteresis_steps)
        self.gripper_min_hold_steps = int(gripper_min_hold_steps)
        self.replan_horizon = None if replan_horizon is None else int(replan_horizon)
        self.adaptive_highscore_replan = bool(adaptive_highscore_replan)
        self.action_ema_alpha = float(action_ema_alpha)

        cfg_path = self.checkpoint_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.yaml in checkpoint dir: {self.checkpoint_dir}")
        self.cfg = OmegaConf.load(cfg_path)

        logger.info(
            "Building Qwen VLA model for evaluation with backbone: %s",
            self.cfg.model.vlm_model_id,
        )
        self.model = build_model(self.cfg)

        payload = torch.load(
            self.checkpoint_dir / "action_head.pt",
            map_location="cpu",
            weights_only=False,
        )
        normalized_action_head = normalize_action_head_state_dict(payload["action_head"])
        missing, unexpected = self.model.action_head.load_state_dict(
            normalized_action_head, strict=False
        )
        missing, unexpected = filter_optional_action_head_load_issues(missing, unexpected)
        if missing or unexpected:
            raise RuntimeError(
                f"Action head checkpoint mismatch for {self.checkpoint_dir}: "
                f"missing={missing} unexpected={unexpected}"
            )

        if "action_q01" in payload and "action_q99" in payload:
            self.model.set_norm_stats(
                payload["action_q01"],
                payload["action_q99"],
                payload.get("local_action_q01", payload["action_q01"]),
                payload.get("local_action_q99", payload["action_q99"]),
                payload.get("state_q01", None),
                payload.get("state_q99", None),
            )

        lora_dir = self.checkpoint_dir / "lora_adapters"
        if lora_dir.exists() and hasattr(self.model.vlm, "model"):
            load_lora_adapters(self.model.vlm.model, lora_dir)

        self.model = self.model.to(self.device)
        self.model.eval()

        self._action_buffer = None
        self._buffer_idx = 0
        self._step_index = 0
        self._recent_action_chunks: list[tuple[int, np.ndarray]] = []
        self._gripper_state: float | None = None
        self._gripper_candidate: float | None = None
        self._gripper_candidate_count = 0
        self._gripper_hold_remaining = 0
        self._prev_motion: np.ndarray | None = None
        self._history_len = int(self.cfg.action_head.get("history_len", 0))
        self._history_feature_dim = int(self.cfg.action_head.state_dim) + int(self.cfg.action_head.action_dim) + 1
        self._history_entries: list[np.ndarray] = []
        self._replan_steps_executed = 0
        self._resolved_replan_horizon: int | None = None
        default_chunk_size = int(
            self.cfg.action_head.get("n_action_steps", self.cfg.action_head.action_horizon)
        )
        self._chunk_size = default_chunk_size if chunk_size is None else int(chunk_size)

        logger.info(
            "Checkpoint loaded from %s at step %s",
            self.checkpoint_dir,
            payload.get("step", "unknown"),
        )
        logger.info(
            "Eval policy settings | chunk_size=%d | num_inference_steps=%s | deterministic_seed=%s | temporal_ensemble_decay=%s | gripper_hysteresis_steps=%d | gripper_min_hold_steps=%d | replan_horizon=%s | adaptive_highscore_replan=%s | action_ema_alpha=%.2f",
            self._chunk_size,
            self.num_inference_steps,
            self.deterministic_seed,
            self.temporal_ensemble_decay,
            self.gripper_hysteresis_steps,
            self.gripper_min_hold_steps,
            self.replan_horizon,
            self.adaptive_highscore_replan,
            self.action_ema_alpha,
        )

    def reset(self):
        self._action_buffer = None
        self._buffer_idx = 0
        self._step_index = 0
        self._recent_action_chunks = []
        self._gripper_state = None
        self._gripper_candidate = None
        self._gripper_candidate_count = 0
        self._gripper_hold_remaining = 0
        self._history_entries = []
        self._prev_motion = None
        self._replan_steps_executed = 0
        self._resolved_replan_horizon = None

    def _build_images(self, obs: dict):
        main = Image.fromarray(obs["image"]).convert("RGB")
        wrist_np = obs.get("wrist_image", obs["image"])
        wrist = Image.fromarray(wrist_np).convert("RGB")
        return [main, wrist]

    @torch.no_grad()
    def _predict_chunk(self, obs: dict, instruction: str) -> np.ndarray:
        state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device)
        history = None
        if self._history_len > 0:
            history_np = np.zeros((self._history_len, self._history_feature_dim), dtype=np.float32)
            recent_entries = self._history_entries[-self._history_len :]
            if recent_entries:
                history_np[-len(recent_entries) :] = np.stack(recent_entries, axis=0)
            history = torch.from_numpy(history_np).unsqueeze(0).to(self.device)
        images = [self._build_images(obs)]
        instructions = [instruction]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_chunk = self.model.predict_action(
                images=images,
                instructions=instructions,
                state=state,
                history=history,
                num_inference_steps=self.num_inference_steps,
                deterministic_seed=self.deterministic_seed,
            )

        return np.asarray(action_chunk, dtype=np.float32)[0]

    def _apply_gripper_hysteresis(self, action: np.ndarray) -> np.ndarray:
        if action.shape[-1] < 7:
            return action

        desired = 1.0 if float(action[6]) > 0.0 else -1.0
        if self._gripper_state is None:
            self._gripper_state = desired
            action[6] = self._gripper_state
            return action

        if self.gripper_hysteresis_steps <= 1 and self.gripper_min_hold_steps <= 0:
            self._gripper_state = desired
            action[6] = self._gripper_state
            return action

        if self._gripper_hold_remaining > 0:
            self._gripper_hold_remaining -= 1
            action[6] = self._gripper_state
            return action

        if desired == self._gripper_state:
            self._gripper_candidate = None
            self._gripper_candidate_count = 0
            action[6] = self._gripper_state
            return action

        if desired != self._gripper_candidate:
            self._gripper_candidate = desired
            self._gripper_candidate_count = 1
        else:
            self._gripper_candidate_count += 1

        if self._gripper_candidate_count >= max(self.gripper_hysteresis_steps, 1):
            self._gripper_state = desired
            self._gripper_candidate = None
            self._gripper_candidate_count = 0
            self._gripper_hold_remaining = max(self.gripper_min_hold_steps - 1, 0)

        action[6] = self._gripper_state
        return action

    def _fuse_current_step(self, action_chunk: np.ndarray) -> np.ndarray:
        if not self.temporal_ensemble_decay or self.temporal_ensemble_decay <= 0.0:
            self._action_buffer = action_chunk[: self._chunk_size]
            self._buffer_idx = 1
            return np.asarray(self._action_buffer[0], dtype=np.float32)

        self._recent_action_chunks.append((self._step_index, action_chunk))
        current_step = self._step_index
        candidates = []
        weights = []

        for start_step, chunk in self._recent_action_chunks:
            rel_idx = current_step - start_step
            if 0 <= rel_idx < len(chunk):
                age = current_step - start_step
                weight = float(self.temporal_ensemble_decay) ** age
                candidates.append(chunk[rel_idx])
                weights.append(weight)

        self._recent_action_chunks = [
            (start_step, chunk)
            for start_step, chunk in self._recent_action_chunks
            if start_step + len(chunk) > current_step + 1
        ]

        if not candidates:
            return np.asarray(action_chunk[0], dtype=np.float32)

        fused = np.average(np.stack(candidates, axis=0), axis=0, weights=np.asarray(weights))
        return np.asarray(fused, dtype=np.float32)

    def _resolve_replan_horizon(self, instruction: str) -> int | None:
        if self.replan_horizon is not None:
            return self.replan_horizon
        if not self.adaptive_highscore_replan:
            return None

        instruction_lower = instruction.lower()
        for pattern, horizon in HIGH_SCORE_REPLAN_SUBSTRINGS.items():
            if pattern in instruction_lower:
                return int(horizon)
        return None

    def _get_action_replan(self, obs: dict, instruction: str, horizon: int) -> np.ndarray:
        need_new_chunk = (
            self._action_buffer is None
            or self._buffer_idx >= len(self._action_buffer)
            or self._replan_steps_executed >= horizon
        )
        if need_new_chunk:
            action_chunk = self._predict_chunk(obs, instruction)
            self._action_buffer = action_chunk[: self._chunk_size]
            self._buffer_idx = 0
            self._replan_steps_executed = 0

        action = np.asarray(self._action_buffer[self._buffer_idx], dtype=np.float32)
        self._buffer_idx += 1
        self._replan_steps_executed += 1
        return action

    @torch.no_grad()
    def get_action(self, obs: dict, instruction: str) -> np.ndarray:
        if self._resolved_replan_horizon is None:
            self._resolved_replan_horizon = self._resolve_replan_horizon(instruction)

        horizon = self._resolved_replan_horizon
        if horizon is not None and 0 < horizon < self._chunk_size:
            action = self._get_action_replan(obs, instruction, horizon)
        elif (
            (not self.temporal_ensemble_decay or self.temporal_ensemble_decay <= 0.0)
            and self._action_buffer is not None
            and self._buffer_idx < len(self._action_buffer)
        ):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            action = np.asarray(action, dtype=np.float32)
        else:
            action_chunk = self._predict_chunk(obs, instruction)
            action = self._fuse_current_step(action_chunk)

        if self.action_ema_alpha > 0.0 and action.shape[-1] >= 7:
            motion = action[:6]
            if self._prev_motion is not None:
                alpha = self.action_ema_alpha
                motion = alpha * motion + (1.0 - alpha) * self._prev_motion
            self._prev_motion = motion.copy()
            action[:6] = motion

        action = self._apply_gripper_hysteresis(action)
        if self._history_len > 0:
            entry = np.zeros((self._history_feature_dim,), dtype=np.float32)
            state_dim = int(self.cfg.action_head.state_dim)
            action_dim = int(self.cfg.action_head.action_dim)
            entry[:state_dim] = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[:state_dim]
            entry[state_dim:state_dim + action_dim] = np.asarray(action, dtype=np.float32).reshape(-1)[:action_dim]
            entry[-1] = 1.0
            self._history_entries.append(entry)
            if len(self._history_entries) > self._history_len:
                self._history_entries = self._history_entries[-self._history_len :]
        self._step_index += 1
        return action


def make_libero_env(task, camera_height=256, camera_width=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    env.seed(0)
    return env


def get_task_suites(suite_names: list[str]):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    return {name: benchmark_dict[name]() for name in suite_names}


def perturb_init_state(init_state: np.ndarray, displacement: float = 0.1, seed: int = 0):
    rng = np.random.RandomState(seed)
    perturbed = init_state.copy()

    robot_state_dim = 32
    idx = min(robot_state_dim, len(perturbed) - 7)
    while idx + 7 <= len(perturbed):
        perturbed[idx] += rng.uniform(-displacement, displacement)
        perturbed[idx + 1] += rng.uniform(-displacement, displacement)
        idx += 7
    return perturbed


def quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {quat.shape}")

    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    quat = quat / norm
    xyz = quat[:3]
    w = float(quat[3])
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(sin_half, w)
    axis = xyz / sin_half
    return (axis * angle).astype(np.float32)


def build_state(obs: dict) -> np.ndarray:
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
    if np.ndim(gripper) > 0:
        gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)[:2]
    else:
        gripper = np.asarray([gripper, -gripper], dtype=np.float32)

    quat = np.asarray(
        obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0])),
        dtype=np.float32,
    )
    axis_angle = quat_xyzw_to_axis_angle(quat)

    return np.concatenate(
        [
            np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32),
            axis_angle,
            gripper.astype(np.float32),
        ]
    ).astype(np.float32)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "task"


def extract_eef_pose(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32)
    return pos, quat


def record_observation_step(record: dict, obs: dict):
    pos, quat = extract_eef_pose(obs)
    record["eef_pos"].append(pos.tolist())
    record["eef_quat"].append(quat.tolist())


def save_rollout_artifacts(
    episode_dir: Path,
    record: dict,
    save_traj_plot: bool = True,
):
    episode_dir.mkdir(parents=True, exist_ok=True)

    eef_pos = np.asarray(record["eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(record["eef_quat"], dtype=np.float32)
    actions = np.asarray(record["actions"], dtype=np.float32)

    np.save(episode_dir / "eef_positions.npy", eef_pos)
    np.save(episode_dir / "eef_quaternions.npy", eef_quat)
    np.save(episode_dir / "actions.npy", actions)

    summary = {
        "task_name": record["task_name"],
        "instruction": record["instruction"],
        "eval_level": record["eval_level"],
        "episode_index": record["episode_index"],
        "success": bool(record["success"]),
        "num_actions": int(len(record["actions"])),
        "num_states": int(len(record["eef_pos"])),
        "eef_pos": record["eef_pos"],
        "eef_quat": record["eef_quat"],
        "actions": record["actions"],
    }

    with open(episode_dir / "trajectory.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if save_traj_plot and len(eef_pos) >= 2:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2], linewidth=2.0)
            ax.scatter(eef_pos[0, 0], eef_pos[0, 1], eef_pos[0, 2], s=60, label="start")
            ax.scatter(eef_pos[-1, 0], eef_pos[-1, 1], eef_pos[-1, 2], s=60, label="end")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"EEF Trajectory | success={record['success']}")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(episode_dir / "eef_trajectory.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to save trajectory plot to %s: %s", episode_dir, exc)


def evaluate_suite(
    policy: Qwen35VLALiberoPolicy,
    suite_name: str,
    suite,
    n_episodes: int = 10,
    max_steps: int = 300,
    eval_level: str = "standard",
    displacement: float = 0.2,
    camera_key: str = "agentview_image",
    wrist_key: str = "robot0_eye_in_hand_image",
    rollout_dir: str | None = None,
    record_tasks: int = 0,
    record_episodes: int = 0,
    save_frames: bool = False,
    save_traj_plot: bool = True,
    task_ids: list[int] | None = None,
):
    num_tasks = suite.n_tasks
    selected_task_ids = list(range(num_tasks)) if task_ids is None else [tid for tid in task_ids if 0 <= tid < num_tasks]
    results = {
        "suite": suite_name,
        "eval_level": eval_level,
        "tasks": {},
        "per_task_success": [],
    }

    logger.info("%s", "=" * 60)
    logger.info(
        "Evaluating %s (%s selected / %s total tasks) | level=%s",
        suite_name,
        len(selected_task_ids),
        num_tasks,
        eval_level,
    )
    logger.info("%s", "=" * 60)

    for task_id in selected_task_ids:
        task = suite.get_task(task_id)
        task_name = task.name
        task_description = task.language

        instruction = (
            rephrase_instruction(task_description, seed=task_id)
            if eval_level == "instruction"
            else task_description
        )

        env = make_libero_env(task)
        init_states = suite.get_task_init_states(task_id)
        episodes = min(n_episodes, len(init_states))
        successes = 0

        for ep in range(episodes):
            policy.reset()
            env.reset()

            init_state = init_states[ep]
            if eval_level == "spatial":
                init_state = perturb_init_state(
                    init_state,
                    displacement=displacement,
                    seed=task_id * 1000 + ep,
                )
            env.set_init_state(init_state)

            obs = env.env._get_observations()
            success = False
            should_record = (
                rollout_dir is not None
                and task_id < max(record_tasks, 0)
                and ep < max(record_episodes, 0)
            )
            episode_dir = None
            episode_record = None
            frame_dir = None
            wrist_frame_dir = None

            if should_record:
                task_slug = slugify(task_name)
                suite_slug = slugify(suite_name)
                level_slug = slugify(eval_level)
                episode_dir = (
                    Path(rollout_dir)
                    / suite_slug
                    / level_slug
                    / f"task_{task_id:02d}_{task_slug}"
                    / f"episode_{ep:02d}"
                )
                episode_record = {
                    "task_name": task_name,
                    "instruction": instruction,
                    "eval_level": eval_level,
                    "episode_index": ep,
                    "eef_pos": [],
                    "eef_quat": [],
                    "actions": [],
                    "success": False,
                }
                record_observation_step(episode_record, obs)
                if save_frames:
                    frame_dir = episode_dir / "agentview_frames"
                    wrist_frame_dir = episode_dir / "wrist_frames"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    wrist_frame_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(obs[camera_key]).save(frame_dir / "frame_0000.png")
                    if wrist_key in obs:
                        Image.fromarray(obs[wrist_key]).save(wrist_frame_dir / "frame_0000.png")

            for step in range(max_steps):
                obs_dict = {
                    "image": obs[camera_key],
                    "wrist_image": obs.get(wrist_key, obs[camera_key]),
                    "state": build_state(obs),
                }
                action = policy.get_action(obs_dict, instruction)
                if episode_record is not None:
                    episode_record["actions"].append(np.asarray(action, dtype=np.float32).tolist())
                obs, reward, done, info = env.step(action)
                if episode_record is not None:
                    record_observation_step(episode_record, obs)
                if frame_dir is not None:
                    Image.fromarray(obs[camera_key]).save(frame_dir / f"frame_{step + 1:04d}.png")
                    if wrist_key in obs:
                        Image.fromarray(obs[wrist_key]).save(
                            wrist_frame_dir / f"frame_{step + 1:04d}.png"
                        )
                if done or info.get("success", False):
                    # LIBERO sets `done` from `_check_success()`, while `info["success"]`
                    # may remain unset for successful episodes.
                    success = bool(done or info.get("success", False))
                    break

            if episode_record is not None and episode_dir is not None:
                episode_record["success"] = bool(success)
                save_rollout_artifacts(
                    episode_dir=episode_dir,
                    record=episode_record,
                    save_traj_plot=save_traj_plot,
                )
                logger.info("Saved rollout artifacts to %s", episode_dir)

            successes += int(success)
            logger.info(
                "Task %02d [%s] ep %d: %s (%d steps)",
                task_id,
                task_name[:40],
                ep,
                "SUCCESS" if success else "FAIL",
                step + 1,
            )

        env.close()
        task_sr = successes / max(episodes, 1)
        results["tasks"][task_name] = {
            "task_id": task_id,
            "instruction": instruction,
            "original_instruction": task_description,
            "success_rate": task_sr,
            "successes": successes,
            "episodes": episodes,
        }
        results["per_task_success"].append(task_sr)
        logger.info(
            "Task %02d success rate: %.1f%% (%d/%d)",
            task_id,
            task_sr * 100.0,
            successes,
            episodes,
        )

    results["average_success_rate"] = float(np.mean(results["per_task_success"]))
    logger.info("%s average success rate: %.1f%%", suite_name, results["average_success_rate"] * 100.0)
    return results


def print_results_table(all_results: dict, output_path: str | None = None):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for suite_name, result in all_results.items():
        level = result.get("eval_level", "standard")
        avg = result.get("average_success_rate", 0.0)
        print(f"\n--- {suite_name} (level: {level}) | avg: {avg:.1%} ---")
        for task_name, task_result in result.get("tasks", {}).items():
            sr = task_result["success_rate"]
            successes = task_result["successes"]
            episodes = task_result["episodes"]
            bar = "█" * int(sr * 20) + "░" * (20 - int(sr * 20))
            print(f"  {task_name[:45]:45s} {bar} {sr:.0%} ({successes}/{episodes})")

    print("\n" + "=" * 70)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, default=str)
        logger.info("Results saved to %s", output_path)


def apply_eval_mode(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode != "official":
        return args

    violations = []

    def require_exact(name: str, actual, expected):
        if actual is None:
            return
        if actual != expected:
            violations.append(f"--{name}={actual} is incompatible with --mode official (expected {expected})")

    require_exact("chunk_size", args.chunk_size, 50)
    require_exact("num_inference_steps", args.num_inference_steps, 10)
    require_exact("temporal_ensemble_decay", args.temporal_ensemble_decay, 0.0)
    require_exact("gripper_hysteresis_steps", args.gripper_hysteresis_steps, 0)
    require_exact("gripper_min_hold_steps", args.gripper_min_hold_steps, 0)
    require_exact("action_ema_alpha", args.action_ema_alpha, 0.0)

    if args.replan_horizon is not None:
        violations.append(
            f"--replan_horizon={args.replan_horizon} is incompatible with --mode official (expected unset)"
        )
    if args.adaptive_highscore_replan:
        violations.append("--adaptive_highscore_replan is incompatible with --mode official")

    if violations:
        joined = "\n  - ".join(violations)
        raise ValueError(
            "Official mode rejects tuned/runtime overrides:\n"
            f"  - {joined}\n"
            "Use --mode tuned for diagnostic or model-specific evals."
        )

    args.chunk_size = 50
    args.num_inference_steps = 10
    args.temporal_ensemble_decay = 0.0
    args.gripper_hysteresis_steps = 0
    args.gripper_min_hold_steps = 0
    args.replan_horizon = None
    args.adaptive_highscore_replan = False
    args.action_ema_alpha = 0.0
    return args


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5-2B VLA on LIBERO")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument(
        "--eval_level",
        type=str,
        default="standard",
        choices=["standard", "instruction", "spatial", "all"],
    )
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--displacement", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="eval_results_qwen35.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mode",
        type=str,
        default="tuned",
        choices=["official", "tuned"],
        help="official locks benchmark-aligned runtime settings; tuned keeps diagnostic/model-specific knobs enabled.",
    )
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    parser.add_argument("--temporal_ensemble_decay", type=float, default=0.0)
    parser.add_argument("--gripper_hysteresis_steps", type=int, default=0)
    parser.add_argument("--gripper_min_hold_steps", type=int, default=0)
    parser.add_argument("--replan_horizon", type=int, default=None)
    parser.add_argument("--adaptive_highscore_replan", action="store_true")
    parser.add_argument("--action_ema_alpha", type=float, default=0.0,
                        help="EMA smoothing on 6D motion (0=disabled, 0.7-0.9 recommended)")
    parser.add_argument("--rollout_dir", type=str, default=None)
    parser.add_argument("--record_tasks", type=int, default=1)
    parser.add_argument("--record_episodes", type=int, default=1)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--no_traj_plot", action="store_true")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    args = apply_eval_mode(args)

    logger.info("Eval mode: %s", args.mode)

    policy = Qwen35VLALiberoPolicy(
        args.checkpoint_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        deterministic_seed=args.deterministic_seed,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        gripper_hysteresis_steps=args.gripper_hysteresis_steps,
        gripper_min_hold_steps=args.gripper_min_hold_steps,
        replan_horizon=args.replan_horizon,
        adaptive_highscore_replan=args.adaptive_highscore_replan,
        action_ema_alpha=args.action_ema_alpha,
    )
    suites = get_task_suites(args.suites)
    levels = ["standard", "instruction", "spatial"] if args.eval_level == "all" else [args.eval_level]

    all_results = {}
    for level in levels:
        for suite_name, suite in suites.items():
            key = suite_name if level == "standard" else f"{suite_name}_{level}"
            all_results[key] = evaluate_suite(
                policy,
                suite_name,
                suite,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                eval_level=level,
                displacement=args.displacement,
                rollout_dir=args.rollout_dir,
                record_tasks=args.record_tasks,
                record_episodes=args.record_episodes,
                save_frames=args.save_frames,
                save_traj_plot=not args.no_traj_plot,
                task_ids=args.task_ids,
            )

    print_results_table(all_results, output_path=args.output)


if __name__ == "__main__":
    main()
