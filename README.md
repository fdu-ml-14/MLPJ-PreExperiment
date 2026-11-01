# MLPJ-PreExperiment

**π0-Bridge / WidowX 预实验：公开 VLA 模型下载、推理适配与 SimplerEnv 闭环评测**

本仓库整理机器学习课程 A 类项目的**预实验**代码与结果：加载公开 VLA checkpoint，在 SimplerEnv 的 WidowX 四个任务上做闭环推理评测，比较 `action_step=1` 与 `action_step=4`。

> **范围声明**
>
> - 本仓库**只包含推理与评测**：模型下载、推理适配、动作执行、逐 episode 记录、汇总统计和视频。
> - **不包含训练或微调**：没有训练配置、训练启动脚本、LoRA、Final checkpoint 或训练日志。
> - **不包含模型权重和数据集**：`models/`、`data/`、`envs/`、`runs/` 均不在仓库中。
> - 所有结果都是单 seed、每设置 10 集的预实验，不是课程正式 Base/Final 评测。

---

## 1. 预实验做了什么

固定 SimplerEnv 任务、初始状态、seed 和最大步数，只改变“一次预测动作块后执行几步”：

| 模型 | 来源 | 任务 | 后端 |
|---|---|---|---|
| π0-Bridge | `juexzz/INTACT-pi0-finetune-bridge` | T1–T4 | CUDA Torch |
| SpatialVLA Bridge | `IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge` | T1–T2 | CUDA Torch |
| Octo-Base | `rail-berkeley/octo-base` | T1–T2 | CPU JAX |
| 通用 π0 | `lerobot/pi0` | T1–T4 | CUDA Torch |

- 任务：T1 `widowx_spoon_on_towel`、T2 `widowx_put_eggplant_in_basket`、T3 `widowx_carrot_on_plate`、T4 `widowx_stack_cube`。
- `episode_id=0..9`，环境 seed 与模型 seed 均为 42，最多 60 个环境步，5Hz。
- 成功标准：SimplerEnv 原生 `success`，不是 `truncated`，也不是视频叠加层。
- 关闭 Octo / SpatialVLA 的 temporal ensemble。
- 三模型 160 集 + 通用 π0 80 集，共 **240 个 episode**。

### `action_step` 的含义

- `action_step=1`：每执行 1 个动作就重新观察、重新预测；推理次数约等于环境步数。
- `action_step=4`：一次预测后连续执行 4 个动作，再重新观察；推理次数约为步数的四分之一。

---

## 2. 主要结果

### 三模型 160 集

| 模型 | action_step=1 | action_step=4 | 变化 |
|---|---:|---:|---:|
| π0-Bridge（T1–T4） | 13/40，32.5% | 25/40，62.5% | +30.0 pp |
| SpatialVLA Bridge（T1–T2） | 14/20，70.0% | 12/20，60.0% | −10.0 pp |
| Octo-Base（T1–T2，CPU） | 0/20，0.0% | 9/20，45.0% | +45.0 pp |

### 通用 π0 80 集

T1–T4 × `action_step=1/4` × 10 集，**全部 0 成功**（`0/40` 与 `0/40`）。

解释：公开 `lerobot/pi0` 的 config 是 6 维动作，WidowX 需要 7 维夹爪动作；只改推理配置不能让第 7 维获得语义。这是一个明确的负结果，不代表 π0 不能学 WidowX。

完整表格、分任务结果和限制见 [`results/Result.md`](results/Result.md)。

---

## 3. 目录结构

```text
MLPJ-PreExperiment/
├── README.md
├── .gitignore
├── env.example.sh
├── configs/
│   └── inference/pi0_base_bridge/
│       ├── config.json
│       ├── policy_preprocessor.json
│       ├── policy_postprocessor.json
│       └── README.md
├── launchers/
│   ├── eval_simpler.sh
│   ├── run_pi0_base_smoke.sh
│   └── run_pi0_base_matrix.sh
├── scripts/
│   ├── extended_eval.py
│   ├── run_pi0_simpler_demo.py
│   ├── check_pi0_base_load.py
│   ├── prepare_pi0_base_model.py
│   ├── summarize_run.py
│   ├── summarize_extended.py
│   ├── make_result.py
│   ├── make_pi0_base_result.py
│   ├── make_pi0_manifest.py
│   ├── validate_pi0_run.py
│   ├── record_representatives.py
│   ├── download_pi0_original.sh
│   ├── download_pi0_direct.sh
│   └── start_model_aria2.sh
├── patches/
│   └── lerobot-tokenizer-path.patch
├── docs/
│   ├── environment.md
│   ├── models.md
│   ├── evaluation.md
│   ├── experiment-log.md
│   └── third_party.md
└── results/
    ├── README.md
    ├── Result.md
    ├── episodes/
    │   ├── extended_models_episodes.csv
    │   ├── extended_models_results.csv
    │   ├── pi0_base_episodes.csv
    │   ├── pi0_base_results.csv
    │   ├── action_step_T1_T4_episodes.csv
    │   ├── action_step_T1_T4_results.csv
    │   ├── representatives_extended.json
    │   └── representatives_pi0_base.json
    └── videos/
        └── 20 个代表性 MP4
```

---

## 4. 环境与依赖

详细步骤见 [`docs/environment.md`](docs/environment.md) 和 [`docs/third_party.md`](docs/third_party.md)。

核心版本：

```text
Ubuntu 22.04 / Python 3.10.12
torch 2.8.0+cu128（Blackwell sm_120 必须 >= 2.8）
transformers 4.48.3 / datasets 2.21.0 / fsspec 2024.6.1
sapien 2.2.2 / setuptools 80.10.2
aria2（下载脚本使用）
```

外部依赖（不随仓库分发，按固定 commit 获取）：

| 项目 | Commit |
|---|---|
| INT-ACT | `66ae3ed4719c68cb2f3f22868015a7a171b04e1f` |
| LeRobot | `35f6e02315dcb3fa25f3a740478265c6c793a95e` |
| SimplerEnv | `fea52fbb9e0da8a2e4e7e5a155b8e5b7f9dd5b87` |
| ManiSkill2_real2sim | `eeb04c788feafdf08f4565bcda34370e5a555325` |

---

## 5. 快速开始

### 5.1 克隆与配置

```bash
git clone https://github.com/fdu-ml-14/MLPJ-PreExperiment.git
cd MLPJ-PreExperiment
cp env.example.sh env.sh
# 编辑 env.sh，把 PI0_ROOT 指向包含 repos/ models/ envs/ runs/ 的运行目录
source env.sh
```

### 5.2 获取外部依赖

按 [`docs/third_party.md`](docs/third_party.md) 克隆 INT-ACT 并初始化子模块，然后应用 tokenizer patch：

```bash
cd "$PI0_ROOT/repos/INT-ACT/third_party/lerobot"
git apply /path/to/MLPJ-PreExperiment/patches/lerobot-tokenizer-path.patch
export PALIGEMMA_TOKENIZER_PATH="$PI0_ROOT/models/paligemma_tokenizer"
```

### 5.3 下载模型

```bash
# 通用 π0：config + 权重，带大小和 SHA256 校验
bash scripts/download_pi0_original.sh

# π0-Bridge：从 HF 镜像解析直链后 aria2 后台下载
bash scripts/start_model_aria2.sh
```

SpatialVLA 与 Octo 需按各自模型卡下载到：

```text
$PI0_ROOT/models/spatialvla_bridge
$PI0_ROOT/models/octo_base
```

### 5.4 跑一条评测

```bash
# π0-Bridge，T1，action_step=4，episode 0
bash launchers/eval_simpler.sh pi0_bridge widowx_spoon_on_towel 4 0

# pi0-base 单集冒烟
bash launchers/run_pi0_base_smoke.sh

# pi0-base 80 集矩阵
bash launchers/run_pi0_base_matrix.sh

# Octo 需要单独的 JAX 环境
PYTHON="$PI0_ROOT/envs/octo/bin/python" \
  bash launchers/eval_simpler.sh octo_base "" 1,4 0-9
```

### 5.5 汇总

```bash
python scripts/summarize_run.py --run-dir "$PI0_ROOT/runs/pi0_base_20261023"
python scripts/validate_pi0_run.py --run-dir "$PI0_ROOT/runs/pi0_base_20261023"
```

---

## 6. 推理适配要点

- WidowX 动作是 7 维：`[x, y, z, roll, pitch, yaw, gripper]`。
- 评测只使用前视主相机：`image_0` → `observation.images.top`，224×224。
- 前 6 维按 Bridge p01/p99 映射到 `[-1, 1]`，夹爪保持原值；策略侧 `normalization_mapping` 为 `IDENTITY`。
- 通用 π0 公开 config 是 6 维；本仓库用 `prepare_pi0_base_model.py` 生成 7 维推理配置，但**不训练第 7 维**，因此 0/80 是预期内的负结果。
- π0 checkpoint 原生 `n_action_steps=4`；`action_step=1` 时要丢弃动作块剩余 3 步并在下一步重新预测，不能只改外层参数。

---

## 7. 文件说明

以下逐个说明仓库中的每个文件。模型权重、数据集、虚拟环境、运行产物不在仓库中。

### 7.1 根目录

| 文件 | 作用 | 关键内容 / 输出 |
|---|---|---|
| `README.md` | 项目入口文档 | 预实验设计、结果、快速开始、逐文件说明、限制。 |
| `.gitignore` | Git 排除规则 | 排除 `models/`、`data/`、`envs/`、`runs/`、`*.safetensors`、`*.pt`、全量视频/PNG、缓存和本地 `env.sh`。 |
| `env.example.sh` | 环境变量模板 | 复制为 `env.sh` 后使用；定义 `PI0_ROOT`、`INTACT_REPO`、`TRAIN_PYTHON`、`SIMPLER_ENV`、`PALIGEMMA_TOKENIZER_PATH`、`VK_ICD_FILENAMES` 等。 |

### 7.2 `configs/inference/pi0_base_bridge/`

| 文件 | 作用 | 关键内容 / 输出 |
|---|---|---|
| `config.json` | 通用 π0 的 7 维 Bridge 推理配置 | 单前视 224×224、7 维 state/action、`chunk_size=50`、`n_action_steps=4`、`max_*_dim=32`、`IDENTITY` 归一化。 |
| `policy_preprocessor.json` | 从 `lerobot/pi0` 保留的预处理器描述 | 仍写 6 维 / `MEAN_STD`；本仓库评测路径不使用它，实际归一化由 `BridgeSimplerAdapter` 完成。 |
| `policy_postprocessor.json` | 从 `lerobot/pi0` 保留的后处理器描述 | 同上；评测路径直接调用 `adapter.postprocess`。 |
| `README.md` | 本目录说明 | 解释每个配置的作用、为什么不包含权重、7 维/夹爪/归一化约定。 |

### 7.3 `launchers/`

| 文件 | 作用 | 关键输入 / 输出 |
|---|---|---|
| `eval_simpler.sh` | 通用单模型评测入口 | 参数：`<model> [tasks] [action_steps] [episode_ids] [output_dir]`；可用 `PYTHON=...` 指定解释器（Octo 用 `envs/octo`）；调用 `scripts/extended_eval.py`，输出到 `$PI0_ROOT/runs/<model>_<date>/`。 |
| `run_pi0_base_smoke.sh` | pi0-base 单集冒烟 | 固定 T1、`action_step=4`、episode 0；输出 `$PI0_ROOT/runs/pi0_base_20261023/smoke/`。 |
| `run_pi0_base_matrix.sh` | pi0-base 80 集矩阵 | T1–T4 × `action_step=1/4` × episode 0–9；输出 `$PI0_ROOT/runs/pi0_base_20261023/`。 |

### 7.4 `scripts/`

| 文件 | 作用 | 关键输入 / 输出 |
|---|---|---|
| `extended_eval.py` | 统一 SimplerEnv 评测器 | 参数：`--model`（必填）、`--tasks`、`--action-steps`、`--episode-ids`、`--seed`、`--max-steps`、`--output-dir`、`--video-dir`、`--record`、`--video-only`、`--model-path`。输出 `episodes.csv`（或 `video_reruns.csv`）、`videos/`、`final_frames/`。支持 `pi0_bridge` / `pi0_base` / `octo_base` / `spatialvla_bridge`。 |
| `run_pi0_simpler_demo.py` | 单 episode π0-Bridge 推理 + 视频 | 参数：`--task`、`--episode-id`、`--seed`、`--action-step`（1/4）、`--max-steps`、`--model-path`、`--output-dir`、`--video-dir`、`--csv-name`。输出 1 个 MP4、1 个 JSON、1 行 CSV。 |
| `check_pi0_base_load.py` | 加载检查 | 无参数；读取 `$PI0_ROOT/models/pi0_base_bridge`，打印参数量、dtype、GPU 显存、`chunk_size`、`n_action_steps` 等 JSON。 |
| `prepare_pi0_base_model.py` | 生成通用 π0 的 Bridge 推理配置 | 参数：`--raw-dir`（原始 `lerobot/pi0` 目录）、`--output-dir`。写出 7 维 `config.json`，复制 pre/post processor，并为 `model.safetensors` 建符号链接。 |
| `summarize_run.py` | 单次评测汇总 | 参数：`--run-dir`。读取 `episodes.csv`，去重、写回，生成 `results.csv`（成功率 + Wilson 95% CI）和 `representatives.json`。 |
| `summarize_extended.py` | 三模型扩展实验汇总 | 无参数；读取 `$RUN_DIR` 下的 `episodes.csv`、`spatialvla_full/episodes.csv`、`octo_full/episodes.csv`，合并去重后写 `episodes.csv`、`results.csv`、`representatives.json`。 |
| `make_result.py` | 生成三模型 `Result.md` | 无参数；读取 `$RUN_DIR` 的 `episodes.csv`、`results.csv`、`video_reruns.csv`，写出三模型结果报告。 |
| `make_pi0_base_result.py` | 生成通用 π0 的 `Result.md` | 参数：`--run-dir`、`--bridge-results`（π0-Bridge 的 `results.csv`）、`--output`。写出 pi0-base 结果和与 π0-Bridge 的同任务对比。 |
| `make_pi0_manifest.py` | 生成 SHA256 清单 | 参数：`--run-dir`、`--model-file`、`--output`。写出 `manifest.json`，含模型文件、CSV、代表视频/终止帧列表和哈希。 |
| `validate_pi0_run.py` | 80 集结果校验 | 参数：`--run-dir`。检查 80 条 episode、8 个分组、8 个代表、任务集合、推理次数（a1 等于步数，a4 等于 `ceil(steps/4)`）和视频/终止帧存在性；失败返回 1。 |
| `record_representatives.py` | 代表性视频重跑 | 无参数；读取 `representatives.json`，逐个用 `--video-only` 重跑选中的 episode，写 `video_reruns.csv` 和 `logs/`。 |
| `download_pi0_original.sh` | 下载通用 π0 | 下载 `config.json`、README、pre/post processor 和 `model.safetensors` 到 `$PI0_ROOT/models/pi0_original_raw`；校验大小 `14,005,623,256` 和 SHA256 `e1bbcefd...febc1`。 |
| `download_pi0_direct.sh` | 通用 π0 权重直连下载 | 用 aria2 从 Hugging Face 直链下载 `model.safetensors`；同样校验大小和 SHA256。 |
| `start_model_aria2.sh` | 下载 π0-Bridge | 从 HF 镜像解析 `model.safetensors` 直链，后台启动 aria2 到 `$PI0_ROOT/models/pi0_bridge/`，日志写 `$PI0_ROOT/models/model_download.log`。 |

### 7.5 `patches/`

| 文件 | 作用 | 关键内容 |
|---|---|---|
| `lerobot-tokenizer-path.patch` | 修复 gated PaliGemma tokenizer | 给 LeRobot `modeling_pi0.py` 增加 `import os`，把 `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")` 改为读取 `PALIGEMMA_TOKENIZER_PATH`，未设置时回退原模型名。用法见 `docs/third_party.md`。 |

### 7.6 `docs/`

| 文件 | 作用 | 关键内容 |
|---|---|---|
| `environment.md` | 环境搭建 | 固定版本、系统依赖、Vulkan、π0 推理/仿真双环境、网络与验收命令。 |
| `models.md` | 模型与适配 | 四个模型的来源、动作空间、后端、通用 π0 的 7 维推理适配和 tokenizer。 |
| `evaluation.md` | 评测协议 | 任务、`action_step` 语义、CSV 字段、常用命令、输出目录和限制。 |
| `experiment-log.md` | 预实验日志 | 2026-10-18 至 10-25 的环境、迁移、单集冒烟、160 集与 80 集过程和问题。 |
| `third_party.md` | 第三方依赖 | INT-ACT / LeRobot / SimplerEnv / ManiSkill2_real2sim 的固定 commit、clone 命令、patch 用法和许可证说明。 |

### 7.7 `results/`

| 文件 | 作用 | 关键内容 |
|---|---|---|
| `README.md` | 结果目录索引 | 每个 CSV/JSON 的用途、字段来源，以及 20 个代表视频的模型/任务/步长/episode 对照表。 |
| `Result.md` | 预实验总报告 | 统一协议、三模型 160 集、通用 π0 80 集、分任务表、解释与限制。 |
| `episodes/extended_models_episodes.csv` | 三模型逐 episode 记录 | 160 行；字段含 model、task、action_step、episode_id、success、steps、inference_calls、延迟、视频路径、error。 |
| `episodes/extended_models_results.csv` | 三模型分组汇总 | 16 个“模型 × 任务 × action_step”分组；含成功数、成功率、Wilson 95% CI、平均步数/推理次数/延迟。 |
| `episodes/pi0_base_episodes.csv` | 通用 π0 逐 episode 记录 | 80 行；字段同上。 |
| `episodes/pi0_base_results.csv` | 通用 π0 分组汇总 | 8 个“任务 × action_step”分组；全部 0 成功。 |
| `episodes/action_step_T1_T4_episodes.csv` | 早期单集冒烟记录 | 8 行；T1–T4 × action_step 1/4，每组合 1 个 episode。 |
| `episodes/action_step_T1_T4_results.csv` | 早期单集汇总 | 8 个组合的成功、步数、推理次数。 |
| `episodes/representatives_extended.json` | 三模型代表视频选择 | 16 条；每条含 model、task、action_step、episode_id、选择原因（首个成功或回退 ep0）。 |
| `episodes/representatives_pi0_base.json` | 通用 π0 代表视频选择 | 8 条；每条含 model、task、action_step、episode_id、video_path、final_frame_path。 |

### 7.8 `results/videos/`

20 个代表性 MP4。文件名格式为 `<model>_<task>_a<action_step>_ep<episode>_<timestamp>.mp4`。

| 文件 | 模型 | 任务 | action_step | episode |
|---|---|---|---:|---:|
| `pi0_bridge_widowx_spoon_on_towel_a1_ep2_20261023_130722.mp4` | π0-Bridge | T1 | 1 | 2 |
| `pi0_bridge_widowx_spoon_on_towel_a4_ep0_20261023_130840.mp4` | π0-Bridge | T1 | 4 | 0 |
| `pi0_bridge_widowx_put_eggplant_in_basket_a1_ep5_20261023_130947.mp4` | π0-Bridge | T2 | 1 | 5 |
| `pi0_bridge_widowx_put_eggplant_in_basket_a4_ep0_20261023_131104.mp4` | π0-Bridge | T2 | 4 | 0 |
| `pi0_bridge_widowx_carrot_on_plate_a1_ep0_20261023_131211.mp4` | π0-Bridge | T3 | 1 | 0 |
| `pi0_bridge_widowx_carrot_on_plate_a4_ep2_20261023_131319.mp4` | π0-Bridge | T3 | 4 | 2 |
| `pi0_bridge_widowx_stack_cube_a1_ep3_20261023_131430.mp4` | π0-Bridge | T4 | 1 | 3 |
| `pi0_bridge_widowx_stack_cube_a4_ep1_20261023_131547.mp4` | π0-Bridge | T4 | 4 | 1 |
| `spatialvla_bridge_widowx_spoon_on_towel_a1_ep0_20261023_131605.mp4` | SpatialVLA | T1 | 1 | 0 |
| `spatialvla_bridge_widowx_spoon_on_towel_a4_ep0_20261023_131640.mp4` | SpatialVLA | T1 | 4 | 0 |
| `spatialvla_bridge_widowx_put_eggplant_in_basket_a1_ep0_20261023_131702.mp4` | SpatialVLA | T2 | 1 | 0 |
| `spatialvla_bridge_widowx_put_eggplant_in_basket_a4_ep0_20261023_131728.mp4` | SpatialVLA | T2 | 4 | 0 |
| `octo_base_widowx_spoon_on_towel_a1_ep0_20261023_131750.mp4` | Octo-Base | T1 | 1 | 0 |
| `octo_base_widowx_spoon_on_towel_a4_ep5_20261023_131937.mp4` | Octo-Base | T1 | 4 | 5 |
| `octo_base_widowx_put_eggplant_in_basket_a1_ep0_20261023_132032.mp4` | Octo-Base | T2 | 1 | 0 |
| `octo_base_widowx_put_eggplant_in_basket_a4_ep0_20261023_132211.mp4` | Octo-Base | T2 | 4 | 0 |
| `pi0_base_widowx_spoon_on_towel_a1_ep0_20261023_150444.mp4` | 通用 π0 | T1 | 1 | 0 |
| `pi0_base_widowx_spoon_on_towel_a4_ep0_20261023_150716.mp4` | 通用 π0 | T1 | 4 | 0 |
| `pi0_base_widowx_put_eggplant_in_basket_a1_ep0_20261023_151503.mp4` | 通用 π0 | T2 | 1 | 0 |
| `pi0_base_widowx_put_eggplant_in_basket_a4_ep0_20261023_151725.mp4` | 通用 π0 | T2 | 4 | 0 |

---

## 8. 限制

- 单 seed（42）、每设置 10 个 episode，Wilson 95% 置信区间很宽，不能作为正式模型排名。
- Octo 跑在 CPU JAX 上，单次推理约 0.9 秒；其成功率差异不能直接解释为模型能力差异。
- SpatialVLA 与 Octo 只覆盖 T1/T2，任务覆盖不完整。
- 这 240 集同时用于模型筛选和趋势判断，存在选择偏差；正式评测应使用预先冻结的新 episode 集合。
- 通用 π0 的 0/80 是“6 维权重 + 7 维推理适配”的负结果，不能据此否定 π0 或 π0-Bridge。
- **本仓库没有训练或微调**，因此没有 Base/Final 对照，也没有 Final checkpoint。

---

## 9. 第三方与许可证

- 评测依赖 INT-ACT、LeRobot、SimplerEnv、ManiSkill2_real2sim，固定 commit 和 clone 命令见 [`docs/third_party.md`](docs/third_party.md)。
- 本仓库**不复制**第三方源码；INT-ACT 当前没有附带 LICENSE 文件，因此只通过文档引用。
- LeRobot 采用 Apache License 2.0；SimplerEnv、ManiSkill2_real2sim 的许可证以其上游仓库为准。
- π0-Bridge、SpatialVLA、Octo、通用 π0 的权重和模型卡版权归各自作者所有，本仓库不再分发。

---

## 10. 不包含的内容

- 模型权重：`*.safetensors`、`*.pt`、`models/`。
- 数据集：`data/`、BridgeV2 子集、SimplerEnv 轨迹。
- 运行环境：`envs/`、`cache/`。
- 运行产物：`runs/`、全量 105 个视频、351 张终止帧 PNG、日志。
- 训练/微调：训练配置、启动脚本、LoRA、Final checkpoint、训练日志。
- 课程报告、PPT、PDF 论文和内部工作文档。

---

## 11. 参考

- π0: *A Vision-Language-Action Flow Model for General Robot Control*，<https://arxiv.org/abs/2410.24164>
- SimplerEnv: <https://github.com/simpler-env/SimplerEnv>
- LeRobot: <https://github.com/huggingface/lerobot>
- INT-ACT: <https://github.com/ai4ce/INT-ACT>
- π0-Bridge checkpoint: <https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge>
- BridgeData V2: <https://rail.eecs.berkeley.edu/datasets/bridge_release/data/>
