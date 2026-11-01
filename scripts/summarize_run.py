#!/usr/bin/env python3
"""Summarize one SimplerEnv evaluation run with Wilson confidence intervals."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TASK_ORDER = {
    "widowx_spoon_on_towel": 0,
    "widowx_put_eggplant_in_basket": 1,
    "widowx_carrot_on_plate": 2,
    "widowx_stack_cube": 3,
}


def truth(value: str) -> bool:
    return str(value).strip().lower() == "true"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, center - half), min(1.0, center + half)


def average(rows: list[dict], field: str) -> float | str:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return sum(values) / len(values) if values else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    episodes_path = run_dir / "episodes.csv"
    with episodes_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    unique: dict[tuple[str, str, int, int], dict] = {}
    for row in rows:
        if truth(row.get("video_only", "false")):
            continue
        row["success"] = truth(row.get("success", "false"))
        key = (row["model"], row["task"], int(row["action_step"]), int(row["episode_id"]))
        unique[key] = row
    rows = sorted(
        unique.values(),
        key=lambda row: (
            row["model"],
            TASK_ORDER.get(row["task"], 99),
            int(row["action_step"]),
            int(row["episode_id"]),
        ),
    )

    fields = list(rows[0].keys()) if rows else []
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["task"], int(row["action_step"]))].append(row)

    result_rows = []
    representatives = []
    for (model, task, action_step), items in sorted(
        groups.items(), key=lambda item: (item[0][0], TASK_ORDER.get(item[0][1], 99), item[0][2])
    ):
        successes = sum(1 for item in items if item["success"])
        rate, low, high = wilson(successes, len(items))
        result_rows.append(
            {
                "model": model,
                "backend": items[0].get("backend", ""),
                "task": task,
                "action_step": action_step,
                "successes": successes,
                "episodes": len(items),
                "success_rate": round(rate, 4),
                "wilson_low": round(low, 4),
                "wilson_high": round(high, 4),
                "mean_steps": round(average(items, "steps"), 2) if average(items, "steps") != "" else "",
                "mean_inference_calls": round(average(items, "inference_calls"), 2)
                if average(items, "inference_calls") != ""
                else "",
                "mean_inference_sec": round(average(items, "mean_inference_sec"), 4)
                if average(items, "mean_inference_sec") != ""
                else "",
                "mean_elapsed_sec": round(average(items, "elapsed_sec"), 2)
                if average(items, "elapsed_sec") != ""
                else "",
            }
        )
        successful = sorted(
            (item for item in items if item["success"]), key=lambda item: int(item["episode_id"])
        )
        chosen = successful[0] if successful else min(items, key=lambda item: int(item["episode_id"]))
        representatives.append(
            {
                "model": model,
                "task": task,
                "action_step": action_step,
                "episode_id": int(chosen["episode_id"]),
                "selected_because": "first_success" if successful else "fallback_episode_0",
                "video_path": chosen.get("video_path", ""),
                "final_frame_path": chosen.get("final_frame_path", ""),
            }
        )

    result_fields = [
        "model", "backend", "task", "action_step", "successes", "episodes",
        "success_rate", "wilson_low", "wilson_high", "mean_steps",
        "mean_inference_calls", "mean_inference_sec", "mean_elapsed_sec",
    ]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(result_rows)
    (run_dir / "representatives.json").write_text(
        json.dumps(representatives, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"episodes={len(rows)} groups={len(result_rows)} representatives={len(representatives)}")
    for row in result_rows:
        print(
            row["model"], row["task"], row["action_step"],
            f"{row['successes']}/{row['episodes']}", row["success_rate"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
