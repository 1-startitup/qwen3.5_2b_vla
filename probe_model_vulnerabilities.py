from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from data.libero_dataset import LiberoVLADataset
from eval_libero import get_task_suites
from train import build_model, load_lora_adapters, normalize_action_head_state_dict


def load_model(config_path: str, checkpoint_dir: Path, device: torch.device):
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
        model.set_norm_stats(
            payload["action_q01"],
            payload["action_q99"],
            payload.get("local_action_q01", payload["action_q01"]),
            payload.get("local_action_q99", payload["action_q99"]),
            payload.get("state_q01", None),
            payload.get("state_q99", None),
        )
    lora_dir = checkpoint_dir / "lora_adapters"
    if lora_dir.exists() and hasattr(model.vlm, "model"):
        load_lora_adapters(model.vlm.model, lora_dir)
    model = model.to(device)
    model.eval()
    return model, cfg


def make_blank_image(image):
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
        blank = np.zeros_like(arr)
        return Image.fromarray(blank)
    arr = np.asarray(image)
    return np.zeros_like(arr)


def normalize_raw_actions(model, raw_actions: np.ndarray) -> np.ndarray:
    low = model.action_q01.detach().cpu().numpy()
    high = model.action_q99.detach().cpu().numpy()
    return 2.0 * (raw_actions - low) / (high - low + 1e-8) - 1.0


@torch.no_grad()
def predict_action(model, images, instruction: str, state: np.ndarray, num_steps: int, deterministic_seed: int):
    device = next(model.parameters()).device
    pred = model.predict_action(
        [images],
        [instruction],
        state=torch.from_numpy(state).float().unsqueeze(0).to(device),
        num_inference_steps=num_steps,
        deterministic_seed=deterministic_seed,
    )
    return np.asarray(pred[0, 0], dtype=np.float32)


def collect_samples(dataset, task_texts: list[str], samples_per_task: int) -> list[dict]:
    task_to_episode = {}
    for ep in dataset.dataset.meta.episodes:
        tasks = ep.get("tasks", [])
        if not tasks:
            continue
        task_text = tasks[0].strip().lower()
        if task_text in task_texts and task_text not in task_to_episode:
            task_to_episode[task_text] = ep
        if len(task_to_episode) == len(task_texts):
            break

    samples = []
    for task_text in task_texts:
        ep = task_to_episode.get(task_text)
        if ep is None:
            continue
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        for idx in range(start, min(end, start + samples_per_task)):
            frame = dataset.dataset[idx]
            samples.append(
                {
                    "task_text": task_text,
                    "frame_index": int(idx),
                    "images": dataset._get_images(frame),
                    "instruction": dataset._get_instruction(frame),
                    "state": np.asarray(frame["observation.state"], dtype=np.float32),
                    "expert_action": np.asarray(frame["action"], dtype=np.float32),
                }
            )
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--samples_per_task", type=int, default=4)
    parser.add_argument("--data_root", type=str, default="/home/frankkkz/datasets")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.config, checkpoint_dir, device)

    suite = get_task_suites([args.suite])[args.suite]
    task_texts = [suite.get_task(i).language.strip().lower() for i in range(suite.n_tasks)]

    dataset = LiberoVLADataset(
        dataset_name=cfg.dataset.name,
        data_root=args.data_root,
        action_horizon=cfg.action_head.action_horizon,
        image_size=tuple(cfg.dataset.image_size),
        action_type=cfg.dataset.action_type,
        vlm_model_id=cfg.model.vlm_model_id,
        norm_stats_sample_size=cfg.dataset.get("norm_stats_sample_size", 20000),
    )

    samples = collect_samples(dataset, task_texts, args.samples_per_task)
    if not samples:
        raise RuntimeError("No samples collected for vulnerability probe.")

    metrics = {
        "base_first_step_mae_sum": 0.0,
        "wrong_instruction_shift_l2_sum": 0.0,
        "blank_image_shift_l2_sum": 0.0,
        "zero_state_shift_l2_sum": 0.0,
        "wrong_instruction_mae_sum": 0.0,
        "blank_image_mae_sum": 0.0,
        "zero_state_mae_sum": 0.0,
        "overall_norm_saturation_sum": 0.0,
        "count": 0,
    }
    per_sample = []

    task_to_wrong_instruction = {}
    for i, task_text in enumerate(task_texts):
        task_to_wrong_instruction[task_text] = task_texts[(i + 1) % len(task_texts)]

    for sample in samples:
        base = predict_action(
            model,
            sample["images"],
            sample["instruction"],
            sample["state"],
            args.num_inference_steps,
            args.deterministic_seed,
        )
        wrong_instruction = predict_action(
            model,
            sample["images"],
            task_to_wrong_instruction[sample["task_text"]],
            sample["state"],
            args.num_inference_steps,
            args.deterministic_seed,
        )
        blank_images = [make_blank_image(img) for img in sample["images"]]
        blank_image = predict_action(
            model,
            blank_images,
            sample["instruction"],
            sample["state"],
            args.num_inference_steps,
            args.deterministic_seed,
        )
        zero_state = predict_action(
            model,
            sample["images"],
            sample["instruction"],
            np.zeros_like(sample["state"]),
            args.num_inference_steps,
            args.deterministic_seed,
        )

        expert = sample["expert_action"]
        base_norm = normalize_raw_actions(model, base)
        saturation = float(np.mean(np.abs(base_norm) > 0.95))

        metrics["base_first_step_mae_sum"] += float(np.mean(np.abs(base - expert)))
        metrics["wrong_instruction_shift_l2_sum"] += float(np.linalg.norm(wrong_instruction - base))
        metrics["blank_image_shift_l2_sum"] += float(np.linalg.norm(blank_image - base))
        metrics["zero_state_shift_l2_sum"] += float(np.linalg.norm(zero_state - base))
        metrics["wrong_instruction_mae_sum"] += float(np.mean(np.abs(wrong_instruction - expert)))
        metrics["blank_image_mae_sum"] += float(np.mean(np.abs(blank_image - expert)))
        metrics["zero_state_mae_sum"] += float(np.mean(np.abs(zero_state - expert)))
        metrics["overall_norm_saturation_sum"] += saturation
        metrics["count"] += 1

        per_sample.append(
            {
                "task_text": sample["task_text"],
                "frame_index": sample["frame_index"],
                "expert_action": expert.tolist(),
                "base_action": base.tolist(),
                "wrong_instruction_action": wrong_instruction.tolist(),
                "blank_image_action": blank_image.tolist(),
                "zero_state_action": zero_state.tolist(),
                "wrong_instruction_shift_l2": float(np.linalg.norm(wrong_instruction - base)),
                "blank_image_shift_l2": float(np.linalg.norm(blank_image - base)),
                "zero_state_shift_l2": float(np.linalg.norm(zero_state - base)),
                "base_first_step_mae": float(np.mean(np.abs(base - expert))),
                "norm_saturation_fraction": saturation,
            }
        )

    count = max(metrics["count"], 1)
    summary = {
        "suite": args.suite,
        "samples_per_task": args.samples_per_task,
        "num_samples": metrics["count"],
        "base_first_step_mae": metrics["base_first_step_mae_sum"] / count,
        "wrong_instruction_shift_l2": metrics["wrong_instruction_shift_l2_sum"] / count,
        "blank_image_shift_l2": metrics["blank_image_shift_l2_sum"] / count,
        "zero_state_shift_l2": metrics["zero_state_shift_l2_sum"] / count,
        "wrong_instruction_mae": metrics["wrong_instruction_mae_sum"] / count,
        "blank_image_mae": metrics["blank_image_mae_sum"] / count,
        "zero_state_mae": metrics["zero_state_mae_sum"] / count,
        "mean_norm_saturation_fraction": metrics["overall_norm_saturation_sum"] / count,
        "per_sample": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
