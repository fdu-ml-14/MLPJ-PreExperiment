#!/usr/bin/env python3
"""Write a SHA256 manifest for the pi0-base run and checkpoint."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 32), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "manifest.json"
    files = [
        Path(args.model_file),
        run_dir / "README.md",
        run_dir / "Result.md",
        run_dir / "episodes.csv",
        run_dir / "results.csv",
        run_dir / "representatives.json",
    ]
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "checkpoint_file": str(Path(args.model_file)),
        "videos": sorted(str(path) for path in (run_dir / "videos").glob("*.mp4")),
        "final_frames": sorted(str(path) for path in (run_dir / "final_frames").glob("*.png")),
        "hashes": {},
    }
    for path in files:
        if path.exists():
            manifest["hashes"][str(path)] = sha256(path)
    manifest["counts"] = {
        "videos": len(manifest["videos"]),
        "final_frames": len(manifest["final_frames"]),
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
