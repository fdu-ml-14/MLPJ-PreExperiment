# pi0_base_bridge 推理配置

本目录是**通用 `lerobot/pi0` checkpoint 的 Bridge/WidowX 推理适配配置**，由 `scripts/prepare_pi0_base_model.py` 生成。它只改变输入输出形状和推理设置，**不改变原始权重，也不是训练得到的 Final**。

## 文件

| 文件 | 作用 |
|---|---|
| `config.json` | 7 维 state/action、单前视 `observation.images.top`、224×224、`chunk_size=50`、`n_action_steps=4`、策略侧 `IDENTITY` 归一化、`max_state_dim=32`、`max_action_dim=32`。 |
| `policy_preprocessor.json` | 从 `lerobot/pi0` 原样保留的预处理器描述。注意它仍写着 6 维 state/action 和 `MEAN_STD`；本仓库的评测路径不使用它，实际的 Bridge 归一化由 INT-ACT 的 `BridgeSimplerAdapter` 按 p01/p99 完成。 |
| `policy_postprocessor.json` | 同上，从原模型保留；评测路径直接调用 `adapter.postprocess`，不经过这个文件。 |
| `README.md` | 本说明。 |

## 不包含

- `model.safetensors`：权重不随仓库分发。运行 `prepare_pi0_base_model.py` 会在目标目录创建指向原始 checkpoint 的符号链接（或复制）。
- 训练配置、优化器状态、Final checkpoint。

## 关键约定

- 动作 7 维：`[x, y, z, roll, pitch, yaw, gripper]`。
- 只使用前视主相机 `observation.images.top`。
- 前 6 维按 Bridge p01/p99 映射到 `[-1, 1]`，夹爪保持原值；策略配置里的 `normalization_mapping` 为 `IDENTITY`。
- `action_step` 只影响评测时执行动作块的前几步，不改变模型权重。
