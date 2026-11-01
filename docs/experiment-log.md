# 预实验日志

记录 2026-10-18 至 2026-10-25 的实际过程。**没有训练或微调**；所有模型都是公开 checkpoint，所有结果都来自推理评测。

## 2026-10-18：环境与资产准备

- 创建 Linux GPU 实例，安装 Python 3.10、Vulkan、uv、ffmpeg 等系统依赖。
- 克隆 INT-ACT 并固定 commit `66ae3ed4719c68cb2f3f22868015a7a171b04e1f`，初始化 LeRobot / SimplerEnv / ManiSkill2_real2sim 子模块。
- 建立两个独立环境：`envs/train`（torch + LeRobot）和 `envs/simpler`（sapien + SimplerEnv）。
- 下载 π0-Bridge checkpoint；BridgeV2 子集仅作环境准备，未参与预实验评测。
- 处理 gated tokenizer：使用公开 Gemma tokenizer，并在 LeRobot `modeling_pi0.py` 中改为本地路径。

遇到的问题：

- `datasets 5.x` 与旧版 LeRobot 不兼容，回退 `2.21.0` / `fsspec 2024.6.1`。
- CPU 环境内存限制导致完整模型无法加载，改用 GPU 实例。

## 2026-10-22：Blackwell 迁移与单集冒烟

- 迁移到 NVIDIA RTX PRO 6000 Blackwell 96GB。
- 修复 Python 3.10 头文件、Vulkan、OpenGL 依赖。
- 原 torch 2.6 不支持 sm_120，升级到 `2.8.0+cu128`。
- π0-Bridge 加载成功：3,501,372,176 参数，bf16 占用约 6.53GB。
- 完成 T1–T4 × `action_step=1/4` 的单 episode 冒烟，共 8 个视频。

关键修正：

- `action_step` 不能只改外层参数；必须同步 `n_action_steps` 并 reset 动作队列，否则“1 步”仍会消费缓存动作。
- 普通帧只显示 `RUNNING`，episode 结束后单独生成带 `FINAL SUCCESS: TRUE/FALSE` 的终止帧；权威结果以 CSV/JSON 的原生 `success` 为准。

## 2026-10-23：三模型 160 集

- 模型：π0-Bridge（T1–T4）、SpatialVLA Bridge（T1–T2）、Octo-Base（T1–T2）。
- 统一关闭 temporal ensemble，固定 `episode_id=0..9`、seed=42、最多 60 步、5Hz。
- 主实验 160 个 episode；另重跑 16 个代表性 episode 录制视频，不计入主统计。
- 结果：
  - π0-Bridge：@1 13/40（32.5%），@4 25/40（62.5%）。
  - SpatialVLA：@1 14/20（70.0%），@4 12/20（60.0%）。
  - Octo-Base：@1 0/20（0.0%），@4 9/20（45.0%）。
- Octo 因 Blackwell JAX 兼容问题跑在 CPU 上，单次推理约 0.9 秒。

## 2026-10-23：通用 π0 80 集

- 下载原始 `lerobot/pi0`：14,005,623,256 bytes，SHA256 `e1bbcefd...febc1`。
- 用 `prepare_pi0_base_model.py` 生成 7 维、单前视、`IDENTITY` 归一化的推理配置；权重用符号链接，不复制。
- T1–T4 × `action_step=1/4` × 10 集，共 80 个 episode。
- 结果：全部 0 成功；无运行异常，推理次数和视频校验通过。
- 解释：公开 config 是 6 维动作，WidowX 需要 7 维夹爪；只改推理配置不能让第 7 维获得语义。

## 2026-10-24 至 10-25：整理与汇报

- 工作区按 `code/docs/results/artifacts/archive` 重组。
- 中期汇报整理为：预实验 240 集、四个模型、动作执行长度对照。
- 本仓库把预实验脚本、推理配置、结果和代表性视频单独整理出来，不含训练/微调内容。

## 结论

- 预实验验证了评测链路、7 维动作接口、视频与原生成功判定。
- `action_step=4` 在 π0-Bridge 上明显优于 1，但方向因模型和任务而异。
- 通用 π0 的 0/80 是负结果，说明“下载权重 + 改推理配置”不等于完成 WidowX 适配。
- 这些结果只用于预实验和方向判断，不是正式 Base/Final 评测；本仓库不包含训练过程。
