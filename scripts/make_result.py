#!/usr/bin/env python3
import csv, json, os
from collections import defaultdict
from pathlib import Path

OUT = Path(os.environ.get("RUN_DIR", "/root/autodl-tmp/pi0_bridge_a/runs/extended_models_20261023"))
MODEL_LABEL = {"pi0_bridge": "π0-Bridge", "spatialvla_bridge": "SpatialVLA Bridge", "octo_base": "Octo-Base"}
TASK_LABEL = {
    "widowx_spoon_on_towel": "T1 spoon_on_towel",
    "widowx_put_eggplant_in_basket": "T2 put_eggplant_in_basket",
    "widowx_carrot_on_plate": "T3 carrot_on_plate",
    "widowx_stack_cube": "T4 stack_cube",
}

def read_csv(path):
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def main():
    episodes = read_csv(OUT/"episodes.csv")
    results = read_csv(OUT/"results.csv")
    videos = read_csv(OUT/"video_reruns.csv")
    counts = defaultdict(lambda: [0,0])
    for r in episodes:
        k=(r["model"],r["action_step"]); counts[k][1]+=1; counts[k][0]+=int(str(r["success"]).lower()=="true")
    lines = [
        "# 三模型 action_step 扩展实验结果",
        "",
        "生成日期：2026-10-23  ",
        "实验环境：AutoDL，NVIDIA RTX PRO 6000 Blackwell 96GB；SimplerEnv WidowX/Bridge 评测栈；Python 3.10.12。",
        "",
        "## 实验协议",
        "",
        "- 固定 `episode_id=0..9`、环境 seed=42、模型 seed=42、最大 60 个环境步、5Hz。",
        "- 关闭 Octo/SpatialVLA temporal ensemble，统一比较“一次预测动作块后执行前 1 或 4 步”。",
        "- π0-Bridge：T1–T4；Octo-Base 与 SpatialVLA：T1–T2。",
        "- 主实验共 160 个 episode；代表性视频重跑不计入主统计。",
        "- 权威成功标准为 SimplerEnv 原生 `success`，不是视频叠加层。",
        "",
        "## 模型与 checkpoint",
        "",
        "- π0-Bridge：`models/pi0_bridge`，CUDA Torch 后端。",
        "- SpatialVLA：`models/spatialvla_bridge`，Bridge SFT checkpoint，CUDA Torch 后端。",
        "- Octo-Base：`rail-berkeley/octo-base`，CPU JAX 后端（Blackwell JAX 兼容性限制）。",
        "",
        "## 总体结果",
        "",
        "| 模型 | action_step | 成功/总数 | 成功率 |",
        "|---|---:|---:|---:|",
    ]
    for model in ("pi0_bridge","spatialvla_bridge","octo_base"):
        for step in (1,4):
            k,n=counts[(model,str(step))]
            if n: lines.append(f"| {MODEL_LABEL[model]} | {step} | {k}/{n} | {k/n:.1%} |")
    lines += ["", "## 分任务结果", "",
              "| 模型 | 任务 | step | 成功/总数 | 成功率 | 平均步数 | 平均推理次数 | 平均单次推理(s) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {MODEL_LABEL.get(r['model'],r['model'])} | {TASK_LABEL.get(r['task'],r['task'])} | {r['action_step']} | {r['successes']}/{r['episodes']} | {float(r['success_rate']):.1%} | {r['mean_steps']} | {r['mean_inference_calls']} | {r['mean_inference_sec']} |")
    lines += ["", "Wilson 95% 置信区间保存在 `results.csv`。", ""]
    if videos:
        lines += ["## 代表性视频", "", "| 模型 | 任务 | step | episode | 视频 |", "|---|---|---:|---:|---|"]
        for r in videos:
            lines.append(f"| {MODEL_LABEL.get(r['model'],r['model'])} | {TASK_LABEL.get(r['task'],r['task'])} | {r['action_step']} | {r['episode_id']} | `{Path(r['video_path']).name}` |")
        lines.append("")
    lines += [
        "## 结论与限制",
        "",
        "- 这轮实验是单 seed、每设置 10 个 episode 的扩展预实验，不能替代课程正式评测。",
        "- Octo 在 CPU JAX 上运行，速度明显慢于另两个 CUDA 模型；其成功率差异不能直接解释为模型能力差异。",
        "- action_step=1 的推理次数约等于步数，action_step=4 的推理次数约为步数的四分之一；推理延迟与成功率需分开解读。",
        "- SpatialVLA 和 Octo 的 adapter 已关闭 temporal ensemble；如果采用原生 ensemble，数字会不同。",
        "",
        "## 文件",
        "",
        "- `episodes.csv`：160 条主实验记录。",
        "- `results.csv`：分组汇总和 Wilson 区间。",
        "- `videos/`：16 个代表性 MP4。",
        "- `final_frames/`：每个 episode 的终止帧。",
        "- `logs/`：模型、下载和运行日志。",
    ]
    (OUT/"Result.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("wrote", OUT/"Result.md")

if __name__ == "__main__": main()
