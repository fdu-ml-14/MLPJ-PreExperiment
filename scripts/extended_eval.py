#!/usr/bin/env python3
"""Unified SimplerEnv evaluator for pi0-Bridge, Octo-Base, and SpatialVLA."""
from __future__ import annotations

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
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ.get("PI0_ROOT", "/root/autodl-tmp/pi0_bridge_a"))
REPO = Path(os.environ.get("INTACT_REPO", ROOT / "repos" / "INT-ACT"))
STATS = REPO / "config" / "dataset" / "bridge_statistics.json"
DEFAULT_OUT = Path(os.environ.get("RUN_DIR", ROOT / "runs" / "extended_models_20261023"))

site.addsitedir(str(ROOT / "envs" / "simpler" / "lib" / "python3.10" / "site-packages"))
sys.path.insert(0, str(REPO))

import simpler_env  # noqa: E402
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402


MODEL_TASKS = {
    "pi0_bridge": [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
    ],
    "pi0_base": [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
    ],
    "octo_base": ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"],
    "spatialvla_bridge": ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"],
}

MODEL_PATHS = {
    "pi0_bridge": ROOT / "models" / "pi0_bridge",
    "pi0_base": ROOT / "models" / "pi0_base_bridge",
    "octo_base": ROOT / "models" / "octo_base",
    "spatialvla_bridge": ROOT / "models" / "spatialvla_bridge",
}


def parse_ints(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def bridge_to_env_actions(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None]
    out = np.zeros((len(raw), 7), dtype=np.float32)
    for i, action in enumerate(raw):
        out[i, :3] = action[:3]
        out[i, 3:6] = Rotation.from_euler("xyz", action[3:6]).as_rotvec()
        out[i, 6] = 2.0 * (float(action[6]) > 0.5) - 1.0
    return out


def overlay(frame: np.ndarray, lines: list[str], final_success: bool | None = None) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if final_success is None:
        y = 24
        for line in lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 25
    else:
        color = (0, 145, 0) if final_success else (0, 0, 165)
        cv2.rectangle(image, (0, 0), (image.shape[1], 92), color, -1)
        status = "FINAL SUCCESS: TRUE" if final_success else "FINAL SUCCESS: FALSE"
        cv2.putText(image, status, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        for idx, line in enumerate(lines):
            cv2.putText(image, line, (12, 66 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Backend:
    name = "unknown"
    backend = "unknown"

    def __init__(self, model_path: Path, seed: int):
        self.model_path = str(model_path)
        self.seed = seed

    def reset_episode(self) -> None:
        pass

    def predict_chunk(self, frame: np.ndarray, obs: dict, instruction: str) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Pi0Backend(Backend):
    name = "pi0_bridge"

    def __init__(self, model_path: Path, seed: int):
        super().__init__(model_path, seed)
        import torch
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        from src.experiments.env_adapters.simpler import BridgeSimplerAdapter
        self.torch = torch
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                image_size=(224, 224),
                action_normalization_type="bound",
                state_normalization_type="bound",
                dataset_statistics_path=str(STATS),
            ),
            use_bf16=True,
            seed=seed,
        )
        self.adapter = BridgeSimplerAdapter(cfg)
        model = PI0Policy.from_pretrained(str(model_path), local_files_only=True)
        self.model = model.to(dtype=torch.bfloat16).to("cuda").eval()
        self.model.config.n_action_steps = 4
        self.model.reset()
        self.backend = "cuda-torch"

    def reset_episode(self) -> None:
        self.model.reset()

    def predict_chunk(self, frame: np.ndarray, obs: dict, instruction: str) -> np.ndarray:
        torch = self.torch
        element = self.adapter.preprocess(
            {"observation.images.top": frame, "observation.state": obs, "task": instruction}
        )
        batch = {
            key: (value.to("cuda", non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in element.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            with torch.inference_mode():
                raw = torch.stack([self.model.select_action(batch) for _ in range(4)], dim=1)
        actions = self.adapter.postprocess(raw[0].float().cpu().numpy())
        return np.asarray(actions, dtype=np.float32)[:4]


class Pi0BaseBackend(Pi0Backend):
    """Original generalist pi0 with a Bridge-compatible inference config."""

    name = "pi0_base"


class SpatialVLABackend(Backend):
    name = "spatialvla_bridge"

    def __init__(self, model_path: Path, seed: int):
        super().__init__(model_path, seed)
        import torch
        from transformers import AutoModel, AutoProcessor
        from src.experiments.env_adapters.simpler import BridgeSimplerSpatialVLAAdapter
        self.torch = torch
        cfg = SimpleNamespace(
            env=SimpleNamespace(
                image_size=(224, 224),
                action_normalization_type="bound",
                state_normalization_type="bound",
                dataset_statistics_path=str(STATS),
            ),
            model_cfg=SimpleNamespace(chunk_size=4),
            use_bf16=True,
            seed=seed,
        )
        self.adapter = BridgeSimplerSpatialVLAAdapter(cfg)
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16
        ).eval().cuda()
        self.unnorm_key = "bridge_orig/1.0.0"
        self.backend = "cuda-torch"

    def predict_chunk(self, frame: np.ndarray, obs: dict, instruction: str) -> np.ndarray:
        torch = self.torch
        element = self.adapter.preprocess(
            {"observation.images.top": frame, "observation.state": obs, "task": instruction}
        )
        inputs = self.processor(
            images=element["observation.images.top"],
            text=element["task"],
            return_tensors="pt",
            do_normalize=False,
        ).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            with torch.inference_mode():
                generation_outputs = self.model.predict_action(inputs)
                decoded = self.processor.decode_actions(
                    generation_outputs, unnorm_key=self.unnorm_key
                )
        raw = decoded["actions"]
        if isinstance(raw, torch.Tensor):
            raw = raw.float().cpu().numpy()
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim == 1:
            raw = raw[None]
        return np.stack([
            self.adapter.postprocess_action(action[None], gripper_binarize=False)[0]
            for action in raw[:4]
        ], axis=0).astype(np.float32)


class OctoBackend(Backend):
    name = "octo_base"

    def __init__(self, model_path: Path, seed: int):
        super().__init__(model_path, seed)
        import jax
        from octo.model.octo_model import OctoModel
        self.jax = jax
        self.model = OctoModel.load_pretrained(str(model_path), step=300000)
        self.history = deque(maxlen=2)
        self.task = None
        self.key = jax.random.PRNGKey(seed)
        self.stats = json.loads(STATS.read_text())
        self.backend = f"{jax.default_backend()}-jax"

    def reset_episode(self) -> None:
        self.history.clear()
        self.task = None
        self.key = self.jax.random.PRNGKey(self.seed)

    def predict_chunk(self, frame: np.ndarray, obs: dict, instruction: str) -> np.ndarray:
        image = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        self.history.append(image)
        images = np.stack(list(self.history), axis=0)[None]
        pad_mask = np.ones((1, images.shape[1]), dtype=np.float32)
        observation = {"image_primary": images, "timestep_pad_mask": pad_mask}
        if self.task is None:
            self.task = self.model.create_tasks(texts=[instruction])
        raw = self.model.sample_actions(
            observation, self.task, rng=self.key, timestep_pad_mask=pad_mask
        )
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim == 1:
            raw = raw[None]
        raw = raw.astype(np.float32).copy()
        mean = np.asarray(self.stats["action"]["mean"], dtype=np.float32)
        std = np.asarray(self.stats["action"]["std"], dtype=np.float32)
        raw[:, :6] = raw[:, :6] * (std[:6] + 1e-8) + mean[:6]
        return bridge_to_env_actions(raw[:4])


def write_csv(path: Path, record: dict, fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fields})


def run_episode(backend: Backend, env, task: str, action_step: int, episode_id: int,
                seed: int, max_steps: int, output_dir: Path, video_dir: Path,
                record: bool, video_only: bool) -> dict:
    final_dir = output_dir / "final_frames"
    final_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    obs, reset_info = env.reset(seed=seed, options={"obj_init_options": {"episode_id": episode_id}})
    instruction = str(env.get_language_instruction())
    limit = min(max_steps or env.spec.max_episode_steps, env.spec.max_episode_steps)
    backend.reset_episode()
    set_seed(seed)
    plan = deque()
    latencies: list[float] = []
    steps = 0
    success = False
    error = ""
    started = time.time()
    tag = f"{backend.name}_{task}_a{action_step}_ep{episode_id}_{datetime.now():%Y%m%d_%H%M%S}"
    video_path = video_dir / f"{tag}.mp4"
    final_path = final_dir / f"{tag}_final.png"
    writer = imageio.get_writer(video_path, fps=5) if record else None

    try:
        while steps < limit:
            frame = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            if writer is not None:
                writer.append_data(overlay(frame, [
                    f"model: {backend.name}  task: {task}",
                    f"episode: {episode_id}  action_step: {action_step}",
                    f"step: {steps}/{limit}",
                    "status: RUNNING",
                ]))
            if not plan:
                t0 = time.time()
                chunk = backend.predict_chunk(frame, obs, instruction)
                latencies.append(time.time() - t0)
                if not np.isfinite(chunk).all():
                    raise RuntimeError("non-finite action chunk")
                plan.extend(chunk[:action_step])
            action = plan.popleft()
            obs, reward, success, terminated, info = env.step(action.copy())
            success = bool(success)
            steps += 1
            if success or terminated:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            try:
                if 'obs' in locals():
                    final_frame = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
                    final_overlay = overlay(final_frame, [
                        f"model: {backend.name}  task: {task}",
                        f"episode: {episode_id}  action_step: {action_step}",
                        f"steps: {steps}/{limit}",
                    ], final_success=success)
                    cv2.imwrite(str(final_path), cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))
                    for _ in range(5):
                        writer.append_data(final_overlay)
            finally:
                writer.close()
        elif 'obs' in locals():
            final_frame = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            final_overlay = overlay(final_frame, [
                f"model: {backend.name}  task: {task}",
                f"episode: {episode_id}  action_step: {action_step}",
                f"steps: {steps}/{limit}",
            ], final_success=success)
            cv2.imwrite(str(final_path), cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))

    return {
        "model": backend.name,
        "backend": backend.backend,
        "checkpoint": backend.model_path,
        "task": task,
        "action_step": action_step,
        "episode_id": episode_id,
        "env_seed": seed,
        "model_seed": seed,
        "ensemble_enabled": False,
        "predicted_chunk_size": 4,
        "inference_calls": len(latencies),
        "success": success,
        "steps": steps,
        "max_steps": limit,
        "mean_inference_sec": float(np.mean(latencies)) if latencies else "",
        "max_inference_sec": float(np.max(latencies)) if latencies else "",
        "elapsed_sec": time.time() - started,
        "final_frame_path": str(final_path),
        "video_path": str(video_path) if record else "",
        "video_only": video_only,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODEL_TASKS))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--action-steps", default="1,4")
    parser.add_argument("--episode-ids", default="0-9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    output_dir = Path(args.output_dir)
    video_dir = Path(args.video_dir) if args.video_dir else output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [x for x in args.tasks.split(",") if x] or MODEL_TASKS[args.model]
    action_steps = parse_ints(args.action_steps)
    episode_ids = parse_ints(args.episode_ids)
    model_path = Path(args.model_path) if args.model_path else MODEL_PATHS[args.model]
    backend_cls = {
        "pi0_bridge": Pi0Backend,
        "pi0_base": Pi0BaseBackend,
        "octo_base": OctoBackend,
        "spatialvla_bridge": SpatialVLABackend,
    }[args.model]

    print(json.dumps({
        "event": "start", "model": args.model, "tasks": tasks,
        "action_steps": action_steps, "episode_ids": episode_ids,
        "backend": backend_cls.__name__, "record": args.record,
    }, ensure_ascii=False), flush=True)
    t0 = time.time()
    backend = backend_cls(model_path, args.seed)
    print(json.dumps({"event": "model_loaded", "model": args.model,
                      "backend": backend.backend, "seconds": round(time.time() - t0, 2)},
                     ensure_ascii=False), flush=True)

    fields = [
        "model", "backend", "checkpoint", "task", "action_step", "episode_id",
        "env_seed", "model_seed", "ensemble_enabled", "predicted_chunk_size",
        "inference_calls", "success", "steps", "max_steps", "mean_inference_sec",
        "max_inference_sec", "elapsed_sec", "final_frame_path", "video_path",
        "video_only", "error",
    ]
    csv_path = output_dir / ("video_reruns.csv" if args.video_only else "episodes.csv")
    try:
        for task in tasks:
            env = simpler_env.make(task)
            try:
                for action_step in action_steps:
                    for episode_id in episode_ids:
                        print(json.dumps({"event": "episode_start", "model": args.model,
                                          "task": task, "action_step": action_step,
                                          "episode_id": episode_id}, ensure_ascii=False), flush=True)
                        record = run_episode(
                            backend, env, task, action_step, episode_id, args.seed,
                            args.max_steps, output_dir, video_dir, args.record, args.video_only,
                        )
                        write_csv(csv_path, record, fields)
                        print(json.dumps({"event": "episode_done", "model": args.model,
                                          "task": task, "action_step": action_step,
                                          "episode_id": episode_id, "success": record["success"],
                                          "steps": record["steps"], "calls": record["inference_calls"],
                                          "seconds": round(record["elapsed_sec"], 1),
                                          "error": record["error"]}, ensure_ascii=False), flush=True)
            finally:
                env.close()
    finally:
        backend.close()
    print(json.dumps({"event": "finished", "model": args.model,
                      "seconds": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
