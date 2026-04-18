import argparse
import json
import re
import shutil
from pathlib import Path


DESKTOP_ROOT = Path("/mnt/c/Users/frank/OneDrive/Desktop")
CURRENT_DIR = DESKTOP_ROOT / "Qwen35_2B_VLA_Current_30k_Spatial_Eval_Report"
V1_DIR = DESKTOP_ROOT / "Qwen35_2B_VLA_Previous_v1_Rollout_Report"
V2_DIR = DESKTOP_ROOT / "Qwen35_2B_VLA_v2_Spatial_Eval_Report"

LOSS_IMAGE_SRC = Path("/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla/training_loss_curve_30k_bs128fresh.png")
TRAIN_LOG = Path("/home/frankkkz/qwen35_2b_vla/train_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh.log")
CURRENT_RESULT_JSON = Path("/home/frankkkz/qwen35_2b_vla/eval_results_libero_spatial_checkpoint_30000.json")
V1_STANDARD_JSON = Path("/home/frankkkz/qwen35_2b_vla/eval_results_closedloop_v1.json")
V1_SMOKE_JSON = Path("/home/frankkkz/qwen35_2b_vla/eval_smoke_spatial_chunk1_steps16_v1.json")

STEP_RE = re.compile(
    r"\[(?P<time>[^\]]+)\]\s+Step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+loss=(?P<loss>[0-9]*\.?[0-9]+).*?\|\s+ETA\s+(?P<eta>[0-9]+)min"
)


def latest_training_snapshot():
    if not TRAIN_LOG.exists():
        return None
    latest = None
    for line in TRAIN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if match:
            latest = {
                "timestamp": match.group("time"),
                "step": int(match.group("step")),
                "total": int(match.group("total")),
                "loss": float(match.group("loss")),
                "eta_min": int(match.group("eta")),
            }
    return latest


def ensure_current_assets():
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    if LOSS_IMAGE_SRC.exists():
        shutil.copy2(LOSS_IMAGE_SRC, CURRENT_DIR / LOSS_IMAGE_SRC.name)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_result_summary(results: dict | None) -> str:
    if not results:
        return "Evaluation results are pending."

    blocks = []
    for suite_name, suite in results.items():
        blocks.append(f"### {suite_name}")
        avg = suite.get("average_success_rate", 0.0)
        blocks.append(f"- Average success rate: `{avg:.1%}`")
        for task_name, task in suite.get("tasks", {}).items():
            blocks.append(
                f"- `{task_name}`: `{task.get('success_rate', 0.0):.1%}` "
                f"({task.get('successes', 0)}/{task.get('episodes', 0)})"
            )
        blocks.append("")
    return "\n".join(blocks).strip()


def write_current_report():
    ensure_current_assets()
    snapshot = latest_training_snapshot()
    result_json = load_json(CURRENT_RESULT_JSON)

    status = "Completed" if result_json else "Pending"
    img_name = LOSS_IMAGE_SRC.name

    lines = [
        "# Qwen3.5-2B VLA Current 30k Spatial Eval Report",
        "",
        f"- Status: `{status}`",
        "- Suite: `libero_spatial`",
        "- Eval level: `spatial`",
        "- Checkpoint target: `checkpoint-30000`",
        "",
    ]

    if snapshot:
        lines += [
            "## Training Snapshot",
            "",
            f"- Latest logged time: `{snapshot['timestamp']}`",
            f"- Progress: `{snapshot['step']}/{snapshot['total']}`",
            f"- Latest train loss: `{snapshot['loss']:.4f}`",
            f"- Latest logged ETA: `{snapshot['eta_min']} min`",
            "",
        ]

    lines += [
        "## Pre-Eval Prediction",
        "",
        (
            "> This loss curve was generated before the closed-loop spatial eval finished. "
            "It is included here as a pre-eval prediction snapshot."
            if result_json
            else "> This loss curve is a prediction aid only. The closed-loop spatial eval has not been completed yet."
        ),
        "",
        f"![Training Loss Curve]({img_name})",
        "",
        "Interpretation:",
        "- The training loss has come down substantially from early training.",
        "- The recent window is still drifting down slowly, which suggests the run is still learning rather than fully flatlining.",
        (
            "- The eval has now completed; use the success-rate section below as the actual result."
            if result_json
            else "- Actual LIBERO spatial success is still unknown until eval finishes."
        ),
        "",
        "## Eval Results",
        "",
        format_result_summary(result_json),
        "",
    ]

    (CURRENT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_v2_eval_report():
    V2_DIR.mkdir(parents=True, exist_ok=True)
    if LOSS_IMAGE_SRC.exists():
        shutil.copy2(LOSS_IMAGE_SRC, V2_DIR / LOSS_IMAGE_SRC.name)
    if CURRENT_RESULT_JSON.exists():
        shutil.copy2(CURRENT_RESULT_JSON, V2_DIR / CURRENT_RESULT_JSON.name)

    snapshot = latest_training_snapshot()
    result_json = load_json(CURRENT_RESULT_JSON)
    img_name = LOSS_IMAGE_SRC.name
    json_name = CURRENT_RESULT_JSON.name

    lines = [
        "# Qwen3.5-2B VLA v2 Spatial Eval Report",
        "",
        "- Status: `Completed`" if result_json else "- Status: `Pending`",
        "- Model line: `v2 bs128 DeepSpeed 30k`",
        "- Suite: `libero_spatial`",
        "- Eval level: `spatial`",
        "- Eval scope: `10 tasks x 10 episodes`",
        "",
    ]

    if snapshot:
        lines += [
            "## Final Training Snapshot",
            "",
            f"- Latest logged time: `{snapshot['timestamp']}`",
            f"- Progress: `{snapshot['step']}/{snapshot['total']}`",
            f"- Latest train loss: `{snapshot['loss']:.4f}`",
            "",
        ]

    lines += [
        "## Prediction Figure",
        "",
        "> This loss image is a pre-eval prediction snapshot that was generated before the eval finished. It is included for context, not as the final metric.",
        "",
        f"![Training Loss Curve]({img_name})",
        "",
        "## Spatial Eval Result",
        "",
        format_result_summary(result_json),
        "",
    ]

    if result_json:
        avg = next(iter(result_json.values())).get("average_success_rate", 0.0)
        lines += [
            "## Conclusion",
            "",
            f"- Final spatial average success rate: `{avg:.1%}`",
            "- This 30k v2 run still failed all evaluated LIBERO spatial tasks in the saved results.",
            f"- Raw eval JSON is included as [{json_name}]({json_name}).",
            "",
        ]

    (V2_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_v1_report():
    V1_DIR.mkdir(parents=True, exist_ok=True)
    standard = load_json(V1_STANDARD_JSON)
    smoke = load_json(V1_SMOKE_JSON)

    lines = [
        "# Qwen3.5-2B VLA Previous v1 Rollout Report",
        "",
        "- Status: `Historical baseline`",
        "- Purpose: separate previous rollout record from the current 30k eval report",
        "",
        "## Standard Closed-Loop Result",
        "",
        format_result_summary(standard),
        "",
        "## Spatial Smoke Result",
        "",
        format_result_summary(smoke),
        "",
        "## Takeaway",
        "",
        "- The v1 rollout baseline remained at `0%` success on the saved LIBERO spatial results in this repository.",
        "- This report is intentionally kept separate from the current bs128 30k run on the Desktop.",
        "",
    ]

    (V1_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-current", action="store_true")
    parser.add_argument("--write-v1", action="store_true")
    parser.add_argument("--write-v2", action="store_true")
    args = parser.parse_args()

    if not args.write_current and not args.write_v1 and not args.write_v2:
        args.write_current = True
        args.write_v1 = True
        args.write_v2 = True

    if args.write_current:
        write_current_report()
    if args.write_v1:
        write_v1_report()
    if args.write_v2:
        write_v2_eval_report()


if __name__ == "__main__":
    main()
