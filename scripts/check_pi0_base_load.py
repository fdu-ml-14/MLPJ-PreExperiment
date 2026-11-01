#!/usr/bin/env python3
"""Load the original pi0 checkpoint on GPU and print a compact sanity report."""

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("PI0_ROOT", "/root/autodl-tmp/pi0_bridge_a"))
sys.path.insert(0, str(ROOT / "repos" / "INT-ACT"))

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402


def main() -> int:
    path = ROOT / "models" / "pi0_base_bridge"
    policy = PI0Policy.from_pretrained(str(path), local_files_only=True)
    parameters = sum(parameter.numel() for parameter in policy.parameters())
    policy = policy.to(dtype=torch.bfloat16).to("cuda").eval()
    torch.cuda.synchronize()
    report = {
        "checkpoint": str(path),
        "parameters": parameters,
        "device": str(next(policy.parameters()).device),
        "dtype": str(next(policy.parameters()).dtype),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "max_state_dim": policy.config.max_state_dim,
        "max_action_dim": policy.config.max_action_dim,
        "tokenizer_max_length": policy.config.tokenizer_max_length,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
