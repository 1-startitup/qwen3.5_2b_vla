from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.libero_dataset import LiberoVLADataset
from eval_libero import get_task_suites
from train import build_model, load_lora_adapters, normalize_action_head_state_dict


def load_model(config_path: str, checkpoint_dir: Path, device: str):
    cfg = OmegaConf.load(config_path)
    model = build_model(cfg)
    payload = torch.load(checkpoint_dir / "action_head.pt", map_location="cpu", weights_only=False)
    normalized = normalize_action_head_state_dict(payload["action_head"])
    missing, unexpected = model.action_head.load_state_dict(normalized, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Action head checkpoint mismatch for {checkpoint_dir}: "
            f"missing={missing} unexpected={unexpected}"
        )
    if "action_q01" in payload and "action_q99" in payload:
        model.set_norm_stats(payload["action_q01"], payload["action_q99"])
    lora_dir = checkpoint_dir / "lora_adapters"
    if lora_dir.exists() and hasattr(model.vlm, "model"):
        load_lora_adapters(model.vlm.model, lora_dir)
    model = model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_action(
    model,
    images,
    instruction: str,
    state: torch.Tensor,
    num_steps: int,
    deterministic_seed: int,
):
    pred = model.predict_action(
        images=[images],
        instructions=[instruction],
        state=state,
        num_inference_steps=num_steps,
        deterministic_seed=deterministic_seed,
    )
    return np.asarray(pred[0, 0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="/home/frankkkz/datasets")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.config, checkpoint_dir, device)

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
    steps = min(args.max_steps, episode_end - episode_start)

    records = []
    for offset in range(steps):
        idx = episode_start + offset
        frame = dataset.dataset[idx]
        images = dataset._get_images(frame)
        instruction = dataset._get_instruction(frame)
        expert_action = np.asarray(frame["action"], dtype=np.float32)
        state = frame["observation.state"]
        if isinstance(state, torch.Tensor):
            state_t = state.float().unsqueeze(0).to(device)
        else:
            state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).float().unsqueeze(0).to(device)

        pred_action = predict_action(
            model,
            images,
            instruction,
            state_t,
            num_steps=args.num_inference_steps,
            deterministic_seed=args.deterministic_seed,
        )

        records.append(
            {
                "frame_index": int(idx),
                "episode_offset": int(offset),
                "expert_action": expert_action.tolist(),
                "expert_gripper": float(expert_action[6]),
                "pred_action": pred_action.tolist(),
                "pred_gripper": float(pred_action[6]),
                "abs_diff": np.abs(pred_action - expert_action).tolist(),
                "mean_abs_diff": float(np.mean(np.abs(pred_action - expert_action))),
            }
        )

    summary = {
        "task_id": int(args.task_id),
        "task_name": task.name,
        "instruction": task.language,
        "matched_episode_index": int(matched_episode["episode_index"]),
        "episode_frame_range": [episode_start, episode_end],
        "max_steps": steps,
        "mean_abs_diff": float(np.mean([record["mean_abs_diff"] for record in records])) if records else 0.0,
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
