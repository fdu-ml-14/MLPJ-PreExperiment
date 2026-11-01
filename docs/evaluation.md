# 评测协议与脚本

## 1. 任务

| 编号 | Task ID | 说明 |
|---|---|---|
| T1 | `widowx_spoon_on_towel` | 勺子放到毛巾上 |
| T2 | `widowx_put_eggplant_in_basket` | 茄子放入篮子 |
| T3 | `widowx_carrot_on_plate` | 胡萝卜放到盘子上 |
| T4 | `widowx_stack_cube` | 绿色方块叠到黄色方块上 |

## 2. action_step 的含义

模型一次预测一个动作块（预实验中统一取 4 个动作）：

- `action_step=1`：只执行动作块第 1 步，然后重新观察、重新预测；推理次数约等于环境步数。
- `action_step=4`：连续执行动作块前 4 步，然后重新观察、重新预测；推理次数约为步数的四分之一。

π0-Bridge 的 checkpoint 原生 `n_action_steps=4`。`extended_eval.py` 的 `Pi0Backend` 固定 `n_action_steps=4`，每次 `predict_chunk` 调用 4 次 `select_action` 取满一个 4 步动作块；`action_step=1` 时丢弃后 3 步，下一步重新取块，因此是真正的逐步重规划。单集脚本 `run_pi0_simpler_demo.py` 则直接把 `n_action_steps` 设为 `action_step`。

## 3. 固定条件

- `episode_id=0..9`（预实验），环境 seed 与模型 seed 均为 42。
- 最多 60 个环境步，5Hz 控制。
- 成功标准：SimplerEnv 原生 `success`，不是 `truncated`，也不是视频叠加层。
- 关闭 Octo / SpatialVLA 的 temporal ensemble。
- Base / 不同模型在独立进程中加载，避免缓存或权重泄漏。

> `extended_eval.py` 会把 `$PI0_ROOT/envs/simpler/lib/python3.10/site-packages` 加入 `sys.path`，并从 `$INTACT_REPO/config/dataset/bridge_statistics.json` 读取统计量。如果仿真环境路径不同，需要改这一行或设置 `PYTHONPATH`。

## 4. 逐 episode CSV 字段

`episodes.csv` 每行一个 episode：

| 字段 | 含义 |
|---|---|
| `model` | `pi0_bridge` / `pi0_base` / `octo_base` / `spatialvla_bridge` |
| `backend` | `cuda-torch` 或 `cpu-jax` |
| `checkpoint` | 模型目录 |
| `task` | SimplerEnv task ID |
| `action_step` | 1 或 4 |
| `episode_id` | 初始状态编号 |
| `env_seed` / `model_seed` | 环境与模型 seed |
| `ensemble_enabled` | 预实验中均为 `False` |
| `predicted_chunk_size` | 预实验中为 4 |
| `inference_calls` | 模型调用次数 |
| `success` | 原生成功判定 |
| `steps` / `max_steps` | 实际步数 / 上限 |
| `mean_inference_sec` / `max_inference_sec` | 单次推理耗时 |
| `elapsed_sec` | 整个 episode 墙钟时间 |
| `final_frame_path` / `video_path` | 终止帧与视频路径 |
| `video_only` | 是否为代表性视频重跑 |
| `error` | 异常信息；正常为空 |

## 5. 常用命令

先准备 `env.sh`：

```bash
cp env.example.sh env.sh
# 编辑 PI0_ROOT
source env.sh
```

通用评测：

```bash
# 模型默认任务，action_step=1/4，episode 0-9
bash launchers/eval_simpler.sh pi0_bridge

# 指定任务、步长、episode
bash launchers/eval_simpler.sh pi0_bridge \
  widowx_spoon_on_towel,widowx_put_eggplant_in_basket 1,4 0-9 \
  "$PI0_ROOT/runs/my_eval"

# Octo 需要单独的 JAX 环境
PYTHON="$PI0_ROOT/envs/octo/bin/python" \
  bash launchers/eval_simpler.sh octo_base "" 1,4 0-9
```

pi0-base 冒烟与 80 集矩阵：

```bash
bash launchers/run_pi0_base_smoke.sh
bash launchers/run_pi0_base_matrix.sh
```

单 episode 视频：

```bash
"$PI0_ROOT/envs/train/bin/python" scripts/run_pi0_simpler_demo.py \
  --task widowx_spoon_on_towel --episode-id 0 --seed 42 \
  --action-step 4 --max-steps 60 \
  --output-dir "$PI0_ROOT/runs/smoke"
```

汇总与校验：

```bash
python scripts/summarize_run.py --run-dir "$PI0_ROOT/runs/pi0_base_20261023"
python scripts/validate_pi0_run.py --run-dir "$PI0_ROOT/runs/pi0_base_20261023"
python scripts/make_pi0_manifest.py \
  --run-dir "$PI0_ROOT/runs/pi0_base_20261023" \
  --model-file "$PI0_ROOT/models/pi0_original_raw/model.safetensors"
```

## 6. 输出目录

```text
<run-dir>/
├── episodes.csv          # 逐 episode 记录
├── results.csv           # 分组成功率 + Wilson 95% CI
├── representatives.json  # 代表视频选择
├── videos/               # MP4
├── final_frames/         # 终止帧 PNG
└── manifest.json         # 可选：SHA256 清单
```

## 7. 重要限制

- 预实验是单 seed、每设置 10 集，不能替代正式评测。
- 同一批 episode 既用于模型筛选又用于报告，存在选择偏差。
- 正式实验应提前冻结 episode 集合、seed、最大步数和模型权重，不能边跑边改。
