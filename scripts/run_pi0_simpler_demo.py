#!/usr/bin/env python3
"""Single-episode pi0-Bridge + SimplerEnv inference and MP4 recorder."""
import argparse
import csv
import json
import os
import site
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio
import numpy as np
import torch

ROOT = Path(os.environ.get("PI0_ROOT", "/root/autodl-tmp/pi0_bridge_a"))
REPO = Path(os.environ.get("INTACT_REPO", ROOT / "repos" / "INT-ACT"))

# Import torch from the training environment before exposing SimplerEnv packages.
site.addsitedir(str(ROOT / "envs" / "simpler" / "lib" / "python3.10" / "site-packages"))
sys.path.insert(0, str(REPO))

import simpler_env  # noqa: E402
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402
from src.experiments.env_adapters.simpler import BridgeSimplerAdapter  # noqa: E402
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def overlay(frame: np.ndarray, lines: list[str], final_success: bool | None = None) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if final_success is None:
        y = 24
        for line in lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 25
    else:
        height = 92
        color = (0, 145, 0) if final_success else (0, 0, 165)
        cv2.rectangle(image, (0, 0), (image.shape[1], height), color, -1)
        status = "FINAL SUCCESS: TRUE" if final_success else "FINAL SUCCESS: FALSE"
        cv2.putText(image, status, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        for idx, line in enumerate(lines):
            cv2.putText(
                image, line, (12, 66 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="widowx_spoon_on_towel")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-step", type=int, choices=(1, 4), default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--model-path", default=str(ROOT / "models" / "pi0_bridge"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "smoke"))
    parser.add_argument("--video-dir", default="", help="Defaults to <output-dir>/videos")
    parser.add_argument("--csv-name", default="episodes.csv")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    video_dir = Path(args.video_dir) if args.video_dir else output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_ep{args.episode_id}_a{args.action_step}_{datetime.now():%Y%m%d_%H%M%S}"
    video_path = video_dir / f"{tag}.mp4"
    metadata_path = output_dir / f"{tag}.json"
    csv_path = output_dir / args.csv_name

    cfg = SimpleNamespace(
        env=SimpleNamespace(
            image_size=(224, 224),
            action_normalization_type="bound",
            state_normalization_type="bound",
            dataset_statistics_path=str(REPO / "config" / "dataset" / "bridge_statistics.json"),
        ),
        use_bf16=True,
        seed=args.seed,
    )
    adapter = BridgeSimplerAdapter(cfg)

    env = None
    writer = None
    steps = 0
    success = False
    error = ""
    started = time.time()
    model_load_sec = None
    inference_times = []
    final_frame_path = None

    try:
        print(f"[info] task={args.task} episode={args.episode_id} seed={args.seed} action_step={args.action_step}")
        env = simpler_env.make(args.task)
        obs, reset_info = env.reset(seed=args.seed, options={"obj_init_options": {"episode_id": args.episode_id}})
        instruction = str(env.get_language_instruction())
        max_steps = min(args.max_steps or env.spec.max_episode_steps, env.spec.max_episode_steps)
        print(f"[info] instruction={instruction!r} max_steps={max_steps}")

        t0 = time.time()
        model = PI0Policy.from_pretrained(args.model_path, local_files_only=True)
        model = model.to(dtype=torch.bfloat16).to("cuda").eval()
        model.config.n_action_steps = args.action_step
        model.reset()
        model_load_sec = time.time() - t0
        print(f"[info] model_load_sec={model_load_sec:.1f} gpu_mem_gb={torch.cuda.memory_allocated()/2**30:.2f}")

        writer = imageio.get_writer(video_path, fps=5)
        action_plan = deque()

        while steps < max_steps:
            frame = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            writer.append_data(
                overlay(
                    frame,
                    [
                        f"task: {args.task}",
                        f"episode: {args.episode_id}  action_step: {args.action_step}",
                        f"step: {steps}/{max_steps}",
                        "status: RUNNING",
                    ],
                )
            )

            if not action_plan:
                batch = adapter.preprocess(
                    {"observation.images.top": frame, "observation.state": obs, "task": instruction}
                )
                batch = {
                    key: (value.to("cuda", non_blocking=True) if isinstance(value, torch.Tensor) else value)
                    for key, value in batch.items()
                }
                t_infer = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        raw_actions = torch.stack(
                            [model.select_action(batch) for _ in range(args.action_step)], dim=1
                        )
                torch.cuda.synchronize()
                inference_times.append(time.time() - t_infer)
                env_actions = adapter.postprocess(
                    raw_actions[0].float().cpu().numpy()
                )
                if not np.isfinite(env_actions).all():
                    raise RuntimeError("non-finite action produced by the policy")
                action_plan.extend(env_actions[: args.action_step])

            action = action_plan.popleft()
            obs, reward, success, truncated, info = env.step(action.copy())
            success = bool(success)
            steps += 1

            if success or truncated:
                final_frame = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
                final_overlay = overlay(
                    final_frame,
                    [
                        f"task: {args.task}  episode: {args.episode_id}",
                        f"action_step: {args.action_step}  steps: {steps}/{max_steps}",
                    ],
                    final_success=success,
                )
                final_frame_path = output_dir / f"{tag}_final.png"
                cv2.imwrite(
                    str(final_frame_path),
                    cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR),
                )
                # Hold the authoritative terminal frame for one second at 5 FPS.
                for _ in range(5):
                    writer.append_data(final_overlay)
                break

        if final_frame_path is None:
            final_frame = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            final_overlay = overlay(
                final_frame,
                [
                    f"task: {args.task}  episode: {args.episode_id}",
                    f"action_step: {args.action_step}  steps: {steps}/{max_steps}",
                ],
                final_success=success,
            )
            final_frame_path = output_dir / f"{tag}_final.png"
            cv2.imwrite(
                str(final_frame_path),
                cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR),
            )
            for _ in range(5):
                writer.append_data(final_overlay)

        print(f"[result] success={success} steps={steps} elapsed_sec={time.time()-started:.1f}")
        print(f"[result] video={video_path}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[error] {error}", file=sys.stderr)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()

    record = {
        "setting": f"base_action_step_{args.action_step}",
        "model": "pi0_bridge",
        "model_path": args.model_path,
        "task": args.task,
        "episode_id": args.episode_id,
        "env_seed": args.seed,
        "model_seed": args.seed,
        "action_step": args.action_step,
        "success": success,
        "steps": steps,
        "max_steps": locals().get("max_steps", args.max_steps),
        "video_path": str(video_path),
        "final_frame_path": str(final_frame_path) if final_frame_path else "",
        "inference_calls": len(inference_times),
        "model_load_sec": model_load_sec,
        "mean_inference_sec": float(np.mean(inference_times)) if inference_times else None,
        "elapsed_sec": time.time() - started,
        "error": error,
    }
    metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, record)
    print(f"[result] metadata={metadata_path}")
    return 0 if not error else 1


def _write_csv(path: Path, record: dict) -> None:
    fields = [
        "setting", "model", "model_path", "task", "episode_id", "env_seed", "model_seed",
        "action_step", "success", "steps", "max_steps", "video_path", "final_frame_path",
        "inference_calls", "model_load_sec", "mean_inference_sec", "elapsed_sec", "error",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fields})


if __name__ == "__main__":
    raise SystemExit(main())
