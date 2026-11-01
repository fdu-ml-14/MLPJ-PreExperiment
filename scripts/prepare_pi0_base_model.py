#!/usr/bin/env python3
"""Create a Bridge-compatible inference config for the original pi0 checkpoint."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((raw / "config.json").read_text(encoding="utf-8"))
    for key in ("push_to_hub", "repo_id", "private", "tags", "license"):
        config.pop(key, None)
    config.update(
        {
            "input_features": {
                "observation.images.top": {"type": "VISUAL", "shape": [3, 224, 224]},
                "observation.state": {"type": "STATE", "shape": [7]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            "normalization_mapping": {
                "VISUAL": "IDENTITY",
                "STATE": "IDENTITY",
                "ACTION": "IDENTITY",
            },
            "chunk_size": 50,
            "n_action_steps": 4,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "resize_imgs_with_padding": [224, 224],
            "empty_cameras": 0,
            "tokenizer_max_length": 48,
            "num_steps": 10,
            "use_cache": True,
            "paligemma_pretrained_path": None,
        }
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in ("README.md", "policy_preprocessor.json", "policy_postprocessor.json"):
        source = raw / name
        if source.exists():
            shutil.copy2(source, output / name)

    weights = output / "model.safetensors"
    try:
        if weights.exists() or weights.is_symlink():
            weights.unlink()
        weights.symlink_to((raw / "model.safetensors").resolve())
    except OSError:
        shutil.copy2(raw / "model.safetensors", weights)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
