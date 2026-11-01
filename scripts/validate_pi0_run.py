#!/usr/bin/env python3
"""Validate the pi0-base 80-episode evaluation run."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_TASKS = {
    "widowx_spoon_on_towel",
    "widowx_put_eggplant_in_basket",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("video_only", "false").lower() != "true"]
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    representatives = json.loads((run_dir / "representatives.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    if len(rows) != 80:
        errors.append(f"expected 80 episodes, got {len(rows)}")
    if len(results) != 8:
        errors.append(f"expected 8 result groups, got {len(results)}")
    if len(representatives) != 8:
        errors.append(f"expected 8 representatives, got {len(representatives)}")
    if {row["task"] for row in rows} != EXPECTED_TASKS:
        errors.append(f"unexpected tasks: {sorted({row['task'] for row in rows})}")
    if {row["model"] for row in rows} != {"pi0_base"}:
        errors.append(f"unexpected models: {sorted({row['model'] for row in rows})}")

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["task"], int(row["action_step"]))].append(row)
        if row.get("error"):
            errors.append(
                f"episode error {row['task']} a{row['action_step']} ep{row['episode_id']}: {row['error']}"
            )
        if row.get("ensemble_enabled", "false").lower() != "false":
            errors.append(f"ensemble enabled: {row['task']} a{row['action_step']} ep{row['episode_id']}")
        steps = int(row["steps"])
        calls = int(row["inference_calls"])
        if int(row["action_step"]) == 1 and calls != steps:
            errors.append(
                f"inference mismatch a1 {row['task']} ep{row['episode_id']}: calls={calls}, steps={steps}"
            )
        if int(row["action_step"]) == 4 and calls != math.ceil(steps / 4):
            errors.append(
                f"inference mismatch a4 {row['task']} ep{row['episode_id']}: calls={calls}, steps={steps}"
            )
        for field in ("video_path", "final_frame_path"):
            path = row.get(field, "")
            if not path or not Path(path).exists():
                errors.append(f"missing {field}: {row['task']} a{row['action_step']} ep{row['episode_id']}")

    for key, items in grouped.items():
        ids = sorted(int(item["episode_id"]) for item in items)
        if ids != list(range(10)):
            errors.append(f"episode ids mismatch for {key}: {ids}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("-", error)
        return 1

    print("VALIDATION PASSED")
    print(f"episodes={len(rows)} groups={len(results)} representatives={len(representatives)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
