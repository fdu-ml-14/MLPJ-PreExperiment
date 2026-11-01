# 模型与推理适配

本仓库只做公开 checkpoint 的下载、推理适配和评测，不训练、不微调。模型权重不随仓库分发。

## 1. 四个模型

| 模型 | 来源 | 动作空间 | 后端 | 在预实验中的任务 |
|---|---|---|---|---|
| π0-Bridge | `juexzz/INTACT-pi0-finetune-bridge` | 7 维 `[x,y,z,roll,pitch,yaw,gripper]`，chunk 50，`n_action_steps=4` | CUDA Torch | T1–T4 |
| SpatialVLA Bridge | `IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge` | 7 维，Bridge SFT | CUDA Torch | T1–T2 |
| Octo-Base | `rail-berkeley/octo-base` | 7 维 Bridge 转换 | CPU JAX | T1–T2 |
| 通用 π0 | `lerobot/pi0`（实际 repo `pi0_old`） | 公开 config 为 6 维，本仓库做 7 维推理适配 | CUDA Torch | T1–T4 |

## 2. π0-Bridge

- 约 3.5B 参数，bf16 权重约 6.53GB。
- 模型卡说明该 checkpoint 使用 BridgeData V2 微调，并给出 SimplerEnv 四个 Bridge 任务成绩。
- 本地目录：`$PI0_ROOT/models/pi0_bridge`。
- 下载脚本：`scripts/start_model_aria2.sh`（从 HF 镜像解析直链后用 aria2 下载）。

## 3. SpatialVLA

- 4B 级别，Bridge SFT checkpoint，`AutoModel` + `trust_remote_code`。
- 评测时关闭 temporal ensemble，否则动作会被压成 1 个，`action_step` 失去意义。
- 本地目录：`$PI0_ROOT/models/spatialvla_bridge`。

## 4. Octo-Base

- 使用 `OctoModel.load_pretrained(..., step=300000)`。
- 原 JAX CUDA 0.4.20 与 Blackwell 不兼容，预实验中跑在 CPU JAX 上；单次推理约 0.9 秒，延迟不可与 CUDA 模型直接比较。
- 本地目录：`$PI0_ROOT/models/octo_base`。

## 5. 通用 π0 的推理适配

通用 `lerobot/pi0` 的公开 config 是 6 维动作；WidowX 需要 7 维（第 7 维夹爪）。本仓库的处理是：

1. 下载原始权重：大小 `14,005,623,256` bytes，SHA256 `e1bbcefd55bab3ddff07ead42f4773505f07c68c9694ae44d636d86733dfebc1`。
2. 用 `scripts/prepare_pi0_base_model.py` 生成 `models/pi0_base_bridge`：
   - 输入：`observation.images.top`（3×224×224）+ `observation.state`（7 维）；
   - 输出：7 维 action；
   - `chunk_size=50`、`n_action_steps=4`、`max_state_dim=32`、`max_action_dim=32`；
   - `normalization_mapping` 全部 `IDENTITY`；
   - 权重用符号链接，避免复制 14GB。
3. 评测时由 INT-ACT 的 `BridgeSimplerAdapter` 做 p01/p99 bound normalization（前 6 维）和夹爪原值处理。

**这只是推理适配，不是训练。** 第 7 维夹爪没有经过训练，因此 80 集全部 0 成功是合理的负结果，不能据此否定 π0 或 π0-Bridge。

## 6. tokenizer

`google/paligemma-3b-pt-224` 是 gated repo。预实验使用公开的 Gemma tokenizer 文件，本地路径由 `PALIGEMMA_TOKENIZER_PATH` 指定：

```bash
export PALIGEMMA_TOKENIZER_PATH="$PI0_ROOT/models/paligemma_tokenizer"
```

需要对 LeRobot 的 `modeling_pi0.py` 应用 `patches/lerobot-tokenizer-path.patch`，把硬编码路径改为环境变量；具体命令见 `docs/third_party.md`。
