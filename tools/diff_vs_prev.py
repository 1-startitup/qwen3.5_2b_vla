#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


BUCKETS = {
    "A_refresh_or_gripper": [1, 3, 8],
    "B_continuity": [0, 5],
    "C_grounding_or_subtask": [4, 6, 7, 9],
}


def load_tasks(path: Path) -> tuple[float, dict[int, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    suite = next(iter(data.values()))
    tasks = {}
    for _, item in suite["tasks"].items():
        tasks[int(item["task_id"])] = item
    return float(suite["average_success_rate"]), tasks


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def average_for_bucket(tasks: dict[int, dict], tids: list[int]) -> float:
    present = [tasks[tid]["success_rate"] for tid in tids if tid in tasks]
    if not present:
        return 0.0
    return sum(present) / len(present)


def decide(avg_delta: float, best_delta: float, worst_delta: float) -> str:
    if avg_delta >= 0.02 and worst_delta > -0.10:
        return "accept"
    if avg_delta <= -0.02 and best_delta < 0.10:
        return "rollback"
    return "ambiguous"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two official eval JSONs and emit a markdown summary.")
    parser.add_argument("--curr", required=True, help="Current eval json")
    parser.add_argument("--prev", required=True, help="Previous eval json")
    parser.add_argument("--label", required=True, help="Current version label")
    parser.add_argument("--prev-label", required=True, help="Previous version label")
    parser.add_argument("--out", required=True, help="Output markdown path")
    args = parser.parse_args()

    curr_avg, curr_tasks = load_tasks(Path(args.curr))
    prev_avg, prev_tasks = load_tasks(Path(args.prev))

    shared = sorted(set(curr_tasks) & set(prev_tasks))
    deltas = []
    for tid in shared:
        curr_sr = float(curr_tasks[tid]["success_rate"])
        prev_sr = float(prev_tasks[tid]["success_rate"])
        deltas.append((tid, curr_sr - prev_sr, curr_sr, prev_sr))
    deltas.sort(key=lambda row: row[1], reverse=True)

    best = deltas[0] if deltas else (None, 0.0, 0.0, 0.0)
    worst = deltas[-1] if deltas else (None, 0.0, 0.0, 0.0)
    avg_delta = curr_avg - prev_avg

    lines = []
    lines.append(f"## {args.label} vs {args.prev_label}")
    lines.append("")
    lines.append(f"- Delta avg SR: {fmt_pct(avg_delta)} ({fmt_pct(prev_avg)} -> {fmt_pct(curr_avg)})")
    if best[0] is not None:
        lines.append(
            f"- Best task delta: task{best[0]:02d} ({fmt_pct(best[3])} -> {fmt_pct(best[2])}, delta {fmt_pct(best[1])})"
        )
        lines.append(
            f"- Worst task delta: task{worst[0]:02d} ({fmt_pct(worst[3])} -> {fmt_pct(worst[2])}, delta {fmt_pct(worst[1])})"
        )

    lines.append("")
    lines.append("### Buckets")
    for bucket_name, tids in BUCKETS.items():
        prev_bucket = average_for_bucket(prev_tasks, tids)
        curr_bucket = average_for_bucket(curr_tasks, tids)
        lines.append(
            f"- {bucket_name}: {fmt_pct(prev_bucket)} -> {fmt_pct(curr_bucket)} (delta {fmt_pct(curr_bucket - prev_bucket)})"
        )

    lines.append("")
    lines.append("### Task Table")
    for tid, delta, curr_sr, prev_sr in deltas:
        lines.append(f"- task{tid:02d}: {fmt_pct(prev_sr)} -> {fmt_pct(curr_sr)} (delta {fmt_pct(delta)})")

    judgment = decide(avg_delta, best[1], worst[1])
    lines.append("")
    lines.append(f"- Judgment: {judgment}")
    if judgment == "accept":
        lines.append("- Recommendation: keep this version as the new baseline and select the next module from the dominant remaining bucket.")
    elif judgment == "rollback":
        lines.append("- Recommendation: rollback to the previous version and inspect the single changed module before trying another delta.")
    else:
        lines.append("- Recommendation: inspect bucket videos and tuned diagnostics before deciding whether to keep or rollback.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
