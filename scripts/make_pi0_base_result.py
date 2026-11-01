#!/usr/bin/env python3
"""Generate the pi0-base Result.md from evaluation CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(row: dict) -> str:
    return f"{float(row['success_rate']) * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bridge-results", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = read_csv(run_dir / "results.csv")
    episodes = read_csv(run_dir / "episodes.csv")
    bridge = read_csv(Path(args.bridge_results))
    output = Path(args.output) if args.output else run_dir / "Result.md"

    lines = [
        "# 通用 π0 Base 在 SimplerEnv/WidowX 上的评测结果",
        "",
        "## 实验定位",
        "",
        "本实验使用原始通用 `lerobot/pi0` checkpoint，经单前视相机、7 维 state/action、"
        "Bridge p01/p99 bound normalization 的推理适配后，在 SimplerEnv T1–T4 上评测。",
        "这不是课程要求的 Final 微调结果，也不是 π0-Bridge 官方 checkpoint 的复现。",
        "",
        "## 固定协议",
        "",
        "- 模型：`lerobot/pi0`（revision `e4ed526af508e58f6008b29e9e48f1098278fdb5`）。",
        "- 任务：T1–T4 四个 WidowX 任务。",
        "- `episode_id=0..9`，`seed=42`，最大 60 个环境步，原生 `success` 为唯一成功标准。",
        "- `action_step=1`：每步重新预测；`action_step=4`：一次预测后执行 4 个动作。",
        "- 关闭 temporal ensemble；每组 10 个 episode。",
        "",
        "## 主结果",
        "",
        "| 任务 | action_step | 成功/总数 | 成功率 | Wilson 95% CI | 平均步数 | 平均推理次数 | 平均推理耗时(s) |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in results:
        if row.get("model") != "pi0_base":
            continue
        lines.append(
            f"| {row['task']} | {row['action_step']} | {row['successes']}/{row['episodes']} | "
            f"{pct(row)} | [{float(row['wilson_low'])*100:.1f}%, {float(row['wilson_high'])*100:.1f}%] | "
            f"{row['mean_steps']} | {row['mean_inference_calls']} | {row['mean_inference_sec']} |"
        )

    by_step = defaultdict(lambda: {"successes": 0, "episodes": 0})
    for row in results:
        if row.get("model") != "pi0_base":
            continue
        bucket = by_step[row["action_step"]]
        bucket["successes"] += int(row["successes"])
        bucket["episodes"] += int(row["episodes"])
    lines += [
        "",
        "## 按 action_step 汇总",
        "",
        "| action_step | 成功/总数 | 成功率 |",
        "|---:|---:|---:|",
    ]
    for action_step in sorted(by_step, key=int):
        bucket = by_step[action_step]
        rate = bucket["successes"] / bucket["episodes"] if bucket["episodes"] else 0.0
        lines.append(f"| {action_step} | {bucket['successes']}/{bucket['episodes']} | {rate*100:.1f}% |")

    lines += [
        "",
        "## 与 π0-Bridge 的同任务参考对比",
        "",
        "| 任务 | action_step | 通用 π0 | π0-Bridge | 差值（通用- Bridge） |",
        "|---|---:|---:|---:|---:|",
    ]
    bridge_index = {
        (row["model"], row["task"], row["action_step"]): row
        for row in bridge
        if row.get("model") == "pi0_bridge"
    }
    for row in results:
        if row.get("model") != "pi0_base":
            continue
        ref = bridge_index.get(("pi0_bridge", row["task"], row["action_step"]))
        if not ref:
            continue
        delta = (float(row["success_rate"]) - float(ref["success_rate"])) * 100
        lines.append(
            f"| {row['task']} | {row['action_step']} | {pct(row)} | {pct(ref)} | {delta:+.1f} pp |"
        )

    errors = [row for row in episodes if row.get("error")]
    lines += [
        "",
        "## 异常与限制",
        "",
        f"- 主实验 episode 数：{len(episodes)}；异常 episode 数：{len(errors)}。",
        "- 单 seed、每组 10 集，置信区间较宽，只适合作为预实验和方向判断。",
        "- 通用 π0 的原始输入是 6 维/多相机/MEAN_STD；本实验做了 Bridge 推理适配，"
        "因此结果衡量的是“适配后通用 Base”的可用性，不等于官方通用 π0 的原生成绩。",
        "- 通用 π0 未在 BridgeData V2 上用本项目数据微调；不能把它表述为课程 Final。",
        "",
        "## 产物",
        "",
        "- `episodes.csv`：逐 episode 记录。",
        "- `results.csv`：分组成功率、Wilson CI、步数、推理次数和延迟。",
        "- `videos/`：服务器保留全部 80 个 episode 视频。",
        "- `final_frames/`：80 张终止帧。",
        "- `representatives.json`：每个任务/action_step 的代表视频选择。",
    ]
    if errors:
        lines.append("")
        lines.append("异常记录：")
        for row in errors[:20]:
            lines.append(f"- {row['task']} ep{row['episode_id']} a{row['action_step']}: {row['error']}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
