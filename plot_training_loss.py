import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"Step\s+(\d+)/(\d+)\s+\|\s+loss=([0-9]*\.?[0-9]+)")


def parse_log(path: Path):
    steps = []
    losses = []
    total_steps = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        total_steps = int(match.group(2))
        loss = float(match.group(3))
        steps.append(step)
        losses.append(loss)
    return steps, losses, total_steps


def ema(values, alpha=0.15):
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: python plot_training_loss.py <log_path> <output_png>")
        raise SystemExit(2)

    log_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    steps, losses, total_steps = parse_log(log_path)
    if not steps:
        raise RuntimeError(f"No step/loss entries found in {log_path}")

    smooth = ema(losses, alpha=0.18)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(steps, losses, color="#9ca3af", linewidth=1.2, label="Raw loss")
    axes[0].plot(steps, smooth, color="#0f766e", linewidth=2.2, label="EMA")
    axes[0].set_title("Training Loss vs Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    recent_n = min(120, len(steps))
    axes[1].plot(steps[-recent_n:], losses[-recent_n:], color="#9ca3af", linewidth=1.2, label="Raw loss")
    axes[1].plot(steps[-recent_n:], smooth[-recent_n:], color="#b45309", linewidth=2.2, label="EMA")
    axes[1].set_title("Recent Window")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    latest = losses[-1]
    best = min(losses)
    first = losses[0]
    total_text = f"latest={latest:.4f}  best={best:.4f}  first={first:.4f}"
    if total_steps is not None:
        total_text += f"  progress={steps[-1]}/{total_steps}"
    fig.suptitle(total_text, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(output_path)


if __name__ == "__main__":
    main()
