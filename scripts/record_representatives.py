#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("PI0_ROOT", "/root/autodl-tmp/pi0_bridge_a"))
OUT = Path(os.environ.get("RUN_DIR", ROOT / "runs" / "extended_models_20261023"))
SCRIPT = Path(__file__).resolve().parent / "extended_eval.py"
TRAIN_PYTHON = Path(os.environ.get("TRAIN_PYTHON", ROOT / "envs/train/bin/python"))
OCTO_PYTHON = Path(os.environ.get("OCTO_PYTHON", ROOT / "envs/octo/bin/python"))
PYTHON = {
    "pi0_bridge": TRAIN_PYTHON,
    "spatialvla_bridge": TRAIN_PYTHON,
    "octo_base": OCTO_PYTHON,
}

def main():
    reps = json.loads((OUT/"representatives.json").read_text())
    video_csv = OUT/"video_reruns.csv"
    if video_csv.exists():
        video_csv.unlink()
    for i, rep in enumerate(reps, 1):
        model = rep["model"]; task = rep["task"]; step = rep["action_step"]; ep = rep["episode_id"]
        log = OUT/"logs"/f"video_{model}_{task}_a{step}_ep{ep}.log"
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["MPLBACKEND"] = "Agg"
        cmd = [str(PYTHON[model]), str(SCRIPT),
               "--model", model, "--tasks", task, "--action-steps", str(step),
               "--episode-ids", str(ep), "--seed", "42", "--max-steps", "60",
               "--output-dir", str(OUT), "--record", "--video-only"]
        print(f"[{i}/{len(reps)}] {model} {task} a{step} ep{ep}", flush=True)
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print("exit", result.returncode, "log", log, flush=True)
    print("video records", len(reps), flush=True)

if __name__ == "__main__":
    main()
