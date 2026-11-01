#!/usr/bin/env python3
import csv, json, math, os
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("RUN_DIR", "/root/autodl-tmp/pi0_bridge_a/runs/extended_models_20261023"))
SOURCES = [ROOT/"episodes.csv", ROOT/"spatialvla_full/episodes.csv", ROOT/"octo_full/episodes.csv"]
MODEL_ORDER = {"pi0_bridge": 0, "spatialvla_bridge": 1, "octo_base": 2}
TASK_ORDER = {
    "widowx_spoon_on_towel": 0,
    "widowx_put_eggplant_in_basket": 1,
    "widowx_carrot_on_plate": 2,
    "widowx_stack_cube": 3,
}

def truth(v):
    return str(v).strip().lower() == "true"

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return p, max(0.0, center-half), min(1.0, center+half)

def main():
    rows = []
    for src in SOURCES:
        if not src.exists():
            print("missing", src)
            continue
        with src.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("video_only", "")).lower() == "true":
                    continue
                row["success"] = truth(row.get("success"))
                rows.append(row)
    unique = {}
    for row in rows:
        key = (row["model"], row["task"], int(row["action_step"]), int(row["episode_id"]))
        unique[key] = row
    rows = list(unique.values())
    rows.sort(key=lambda r: (MODEL_ORDER.get(r["model"], 99), TASK_ORDER.get(r["task"], 99),
                             int(r["action_step"]), int(r["episode_id"])))

    fields = [
        "model", "backend", "checkpoint", "task", "action_step", "episode_id",
        "env_seed", "model_seed", "ensemble_enabled", "predicted_chunk_size",
        "inference_calls", "success", "steps", "max_steps", "mean_inference_sec",
        "max_inference_sec", "elapsed_sec", "final_frame_path", "video_path",
        "video_only", "error",
    ]
    with (ROOT/"episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["task"], int(r["action_step"]))].append(r)

    out_rows = []
    reps = []
    for key, items in sorted(groups.items(), key=lambda kv: (
            MODEL_ORDER.get(kv[0][0], 99), TASK_ORDER.get(kv[0][1], 99), kv[0][2])):
        model, task, step = key
        n = len(items)
        k = sum(1 for r in items if r["success"])
        p, lo, hi = wilson(k, n)
        def avg(field):
            vals = [float(r[field]) for r in items if r.get(field) not in ("", None)]
            return sum(vals)/len(vals) if vals else ""
        out_rows.append({
            "model": model, "task": task, "action_step": step,
            "successes": k, "episodes": n, "success_rate": round(p, 4),
            "wilson_low": round(lo, 4), "wilson_high": round(hi, 4),
            "mean_steps": round(avg("steps"), 2) if avg("steps") != "" else "",
            "mean_inference_calls": round(avg("inference_calls"), 2) if avg("inference_calls") != "" else "",
            "mean_inference_sec": round(avg("mean_inference_sec"), 4) if avg("mean_inference_sec") != "" else "",
            "mean_elapsed_sec": round(avg("elapsed_sec"), 2) if avg("elapsed_sec") != "" else "",
            "backend": items[0].get("backend", ""),
        })
        successes = sorted((r for r in items if r["success"]), key=lambda r: int(r["episode_id"]))
        chosen = int(successes[0]["episode_id"]) if successes else 0
        reps.append({"model": model, "task": task, "action_step": step, "episode_id": chosen,
                     "selected_because": "first_success" if successes else "fallback_episode_0"})

    fields2 = ["model", "task", "action_step", "successes", "episodes", "success_rate",
               "wilson_low", "wilson_high", "mean_steps", "mean_inference_calls",
               "mean_inference_sec", "mean_elapsed_sec", "backend"]
    with (ROOT/"results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader(); w.writerows(out_rows)
    (ROOT/"representatives.json").write_text(json.dumps(reps, indent=2), encoding="utf-8")
    print("rows", len(rows), "groups", len(out_rows), "reps", len(reps))
    for r in out_rows:
        print(r["model"], r["task"], r["action_step"], f"{r['successes']}/{r['episodes']}", r["success_rate"])

if __name__ == "__main__":
    main()
