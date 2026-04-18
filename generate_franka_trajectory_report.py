from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class TaskSummary:
    task_dir_name: str
    task_key: str
    task_id: int
    instruction: str
    rollout_success: bool
    full_eval_success_rate: float
    full_eval_successes: int
    full_eval_episodes: int
    num_actions: int
    num_states: int
    path_length_m: float
    displacement_m: float
    efficiency: float
    max_step_m: float
    image_name: str
    json_name: str
    agentview_sheet_name: str | None = None
    wrist_sheet_name: str | None = None
    agentview_video_name: str | None = None


def _slug_to_task_key(task_dir_name: str) -> str:
    parts = task_dir_name.split("_", 2)
    if len(parts) < 3:
        return task_dir_name
    return parts[2]


def _compute_motion_metrics(points: list[list[float]]) -> tuple[float, float, float, float]:
    total = 0.0
    max_step = 0.0
    for a, b in zip(points, points[1:]):
        step = math.dist(a, b)
        total += step
        max_step = max(max_step, step)
    displacement = math.dist(points[0], points[-1]) if len(points) >= 2 else 0.0
    efficiency = displacement / total if total else 0.0
    return total, displacement, efficiency, max_step


def load_task_summaries(rollout_base: Path, eval_json: Path) -> tuple[list[TaskSummary], float]:
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))["libero_spatial_spatial"]
    per_task = eval_data["tasks"]
    average_success_rate = float(eval_data.get("average_success_rate", 0.0))

    summaries: list[TaskSummary] = []
    for task_dir in sorted(rollout_base.iterdir()):
        if not task_dir.is_dir():
            continue
        episode_dir = task_dir / "episode_00"
        trajectory_path = episode_dir / "trajectory.json"
        image_path = episode_dir / "eef_trajectory.png"
        if not trajectory_path.exists() or not image_path.exists():
            continue

        traj = json.loads(trajectory_path.read_text(encoding="utf-8"))
        task_key = _slug_to_task_key(task_dir.name)
        eval_task = per_task[task_key]
        path_length, displacement, efficiency, max_step = _compute_motion_metrics(traj["eef_pos"])

        summaries.append(
            TaskSummary(
                task_dir_name=task_dir.name,
                task_key=task_key,
                task_id=int(eval_task["task_id"]),
                instruction=str(eval_task["instruction"]),
                rollout_success=bool(traj["success"]),
                full_eval_success_rate=float(eval_task["success_rate"]),
                full_eval_successes=int(eval_task["successes"]),
                full_eval_episodes=int(eval_task["episodes"]),
                num_actions=int(traj["num_actions"]),
                num_states=int(traj["num_states"]),
                path_length_m=path_length,
                displacement_m=displacement,
                efficiency=efficiency,
                max_step_m=max_step,
                image_name=f"{task_dir.name}_eef_trajectory.png",
                json_name=f"{task_dir.name}_trajectory.json",
            )
        )

    return summaries, average_success_rate


def copy_rollout_assets(rollout_base: Path, output_assets: Path, summaries: list[TaskSummary]) -> None:
    output_assets.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        episode_dir = rollout_base / summary.task_dir_name / "episode_00"
        shutil.copy2(episode_dir / "eef_trajectory.png", output_assets / summary.image_name)
        shutil.copy2(episode_dir / "trajectory.json", output_assets / summary.json_name)

        agentview_dir = episode_dir / "agentview_frames"
        if agentview_dir.exists():
            summary.agentview_sheet_name = f"{summary.task_dir_name}_agentview_sheet.png"
            save_contact_sheet(agentview_dir, output_assets / summary.agentview_sheet_name)

        wrist_dir = episode_dir / "wrist_frames"
        if wrist_dir.exists():
            summary.wrist_sheet_name = f"{summary.task_dir_name}_wrist_sheet.png"
            save_contact_sheet(wrist_dir, output_assets / summary.wrist_sheet_name)

        agentview_video_path = output_assets / f"{summary.task_dir_name}_agentview.mp4"
        if agentview_video_path.exists():
            summary.agentview_video_name = agentview_video_path.name


def save_contact_sheet(frame_dir: Path, output_path: Path, num_frames: int = 8) -> None:
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        return

    if len(frame_paths) <= num_frames:
        selected = frame_paths
    else:
        last_index = len(frame_paths) - 1
        selected_indices = sorted(
            {
                round(i * last_index / (num_frames - 1))
                for i in range(num_frames)
            }
        )
        selected = [frame_paths[i] for i in selected_indices]

    images = [Image.open(path).convert("RGB") for path in selected]
    tile_w, tile_h = images[0].size
    columns = 4
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), color=(255, 255, 255))

    for index, image in enumerate(images):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        sheet.paste(image, (x, y))

    sheet.save(output_path)

    for image in images:
        image.close()


def maybe_copy_loss_curve(loss_curve: Path | None, output_dir: Path) -> str | None:
    if not loss_curve or not loss_curve.exists():
        return None
    dst = output_dir / loss_curve.name
    if loss_curve.resolve() != dst.resolve():
        shutil.copy2(loss_curve, dst)
    return dst.name


def write_report(
    output_dir: Path,
    summaries: list[TaskSummary],
    average_success_rate: float,
    loss_curve_name: str | None,
) -> Path:
    report_path = output_dir / "FRANKA_TRAJECTORY_AND_PRECISION_REPORT.md"
    lines: list[str] = []
    lines.append("# Qwen3.5-2B VLA v2 LIBERO Spatial Franka Trajectory Report")
    lines.append("")
    lines.append("This report summarizes one recorded Franka rollout trajectory for each LIBERO spatial task using the 30k-step v2 checkpoint.")
    lines.append("")
    lines.append("## What \"precision\" means here")
    lines.append("")
    lines.append("- Hard precision metric: full LIBERO spatial success rate over 10 episodes per task.")
    lines.append("- Motion-quality proxies: recorded rollout success, end-effector path length, net displacement, path efficiency, and max per-step end-effector motion.")
    lines.append("- Caveat: target-pose error is not saved by the current evaluator, so exact placement error in meters cannot be reconstructed from artifacts alone.")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append(f"- Full spatial average success rate: `{average_success_rate * 100:.1f}%`")
    lines.append(f"- Tasks covered with recorded Franka trajectories: `{len(summaries)}` / `10`")
    lines.append("- All recorded rollouts below are real simulator trajectories exported from the evaluator, not hand-drawn sketches.")
    lines.append("- Important eval caveat: the current eval logs showed action-head checkpoint key mismatch (`_orig_mod.*`), so these trajectories reflect the current evaluator output but likely understate final policy quality.")
    lines.append("")
    if loss_curve_name:
        lines.append("## Training Snapshot")
        lines.append("")
        lines.append("This training-loss figure is included for context only. It is not a success metric.")
        lines.append("")
        lines.append(f"![Training loss snapshot](./{loss_curve_name})")
        lines.append("")
    lines.append("## Per-Task Franka Motion")
    lines.append("")

    for summary in summaries:
        lines.append(f"### Task {summary.task_id:02d}")
        lines.append("")
        lines.append(f"- Instruction: `{summary.instruction}`")
        lines.append(f"- Full eval success rate: `{summary.full_eval_successes}/{summary.full_eval_episodes}` = `{summary.full_eval_success_rate * 100:.1f}%`")
        lines.append(f"- Recorded rollout success: `{summary.rollout_success}`")
        lines.append(f"- Trajectory samples: `{summary.num_states}` states, `{summary.num_actions}` actions")
        lines.append(f"- End-effector path length: `{summary.path_length_m:.4f} m`")
        lines.append(f"- Start-to-end displacement: `{summary.displacement_m:.4f} m`")
        lines.append(f"- Path efficiency: `{summary.efficiency:.3f}`")
        lines.append(f"- Max single-step EEF motion: `{summary.max_step_m:.4f} m`")
        lines.append(f"- Raw trajectory JSON: [assets/{summary.json_name}](./assets/{summary.json_name})")
        if summary.agentview_video_name:
            lines.append(f"- Agent-view video: [assets/{summary.agentview_video_name}](./assets/{summary.agentview_video_name})")
        lines.append("")
        if summary.agentview_sheet_name:
            lines.append("Agent-view simulation frames:")
            lines.append("")
            lines.append(f"![Task {summary.task_id:02d} agent view](./assets/{summary.agentview_sheet_name})")
            lines.append("")
        if summary.wrist_sheet_name:
            lines.append("Wrist-camera simulation frames:")
            lines.append("")
            lines.append(f"![Task {summary.task_id:02d} wrist view](./assets/{summary.wrist_sheet_name})")
            lines.append("")
        lines.append(f"![Task {summary.task_id:02d} trajectory](./assets/{summary.image_name})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-base", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-curve", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, average_success_rate = load_task_summaries(args.rollout_base, args.eval_json)
    if not summaries:
        raise SystemExit("No rollout summaries found.")

    assets_dir = args.output_dir / "assets"
    copy_rollout_assets(args.rollout_base, assets_dir, summaries)
    loss_curve_name = maybe_copy_loss_curve(args.loss_curve, args.output_dir)
    report_path = write_report(args.output_dir, summaries, average_success_rate, loss_curve_name)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
