# 预实验结果汇总

本文件汇总 2026-10-22 至 2026-10-23 的 SimplerEnv/WidowX 预实验。**全部结果均为推理评测，没有训练或微调**；模型权重均为公开 checkpoint。

## 统一协议

- 任务：T1 `widowx_spoon_on_towel`、T2 `widowx_put_eggplant_in_basket`、T3 `widowx_carrot_on_plate`、T4 `widowx_stack_cube`。
- `episode_id=0..9`，环境 seed 与模型 seed 均为 42，最多 60 个环境步，控制频率 5Hz。
- 成功标准：SimplerEnv 原生 `success`，不是视频叠加层，也不是 `truncated`。
- 关闭 Octo / SpatialVLA 的 temporal ensemble，统一为“一次预测动作块后执行前 1 或 4 步”。
- 单 seed、每设置 10 个 episode，仅用于预实验和方向判断。

## 三模型 160 集

| 模型 | action_step=1 | action_step=4 | 变化 |
|---|---:|---:|---:|
| π0-Bridge（T1–T4） | 13/40，32.5% | 25/40，62.5% | +30.0 pp |
| SpatialVLA Bridge（T1–T2） | 14/20，70.0% | 12/20，60.0% | −10.0 pp |
| Octo-Base（T1–T2，CPU JAX） | 0/20，0.0% | 9/20，45.0% | +45.0 pp |

### 分任务：π0-Bridge

| 任务 | @1 | @4 |
|---|---:|---:|
| T1 spoon_on_towel | 4/10，40% | 5/10，50% |
| T2 put_eggplant_in_basket | 2/10，20% | 8/10，80% |
| T3 carrot_on_plate | 3/10，30% | 3/10，30% |
| T4 stack_cube | 4/10，40% | 9/10，90% |

### 分任务：SpatialVLA Bridge

| 任务 | @1 | @4 |
|---|---:|---:|
| T1 spoon_on_towel | 4/10，40% | 2/10，20% |
| T2 put_eggplant_in_basket | 10/10，100% | 10/10，100% |

### 分任务：Octo-Base

| 任务 | @1 | @4 |
|---|---:|---:|
| T1 spoon_on_towel | 0/10，0% | 1/10，10% |
| T2 put_eggplant_in_basket | 0/10，0% | 8/10，80% |

## 通用 π0 80 集

原始 `lerobot/pi0` 经单前视、7 维 state/action、Bridge p01/p99 bound normalization 的**推理适配**后评测；权重没有更新。

| 任务 | @1 | @4 |
|---|---:|---:|
| T1 spoon_on_towel | 0/10 | 0/10 |
| T2 put_eggplant_in_basket | 0/10 | 0/10 |
| T3 carrot_on_plate | 0/10 | 0/10 |
| T4 stack_cube | 0/10 | 0/10 |
| **合计** | **0/40** | **0/40** |

## 解释

- π0-Bridge 在 4 步设置下四任务合计 62.5%，明显高于 1 步的 32.5%；T2、T4 的提升最大。
- SpatialVLA 在 T2 两个设置都是 10/10，T1 则是 1 步更好；说明执行长度的效应与模型、任务有关。
- Octo-Base 在 CPU JAX 上运行，单次推理约 0.9 秒，其成功率差异不能直接解释为模型能力差异。
- 通用 π0 的 0/80 是一个明确的负结果：公开 checkpoint 的动作空间是 6 维，而 WidowX 需要 7 维夹爪动作；只改推理配置不足以完成适配。
- `action_step=1` 的推理次数等于环境步数；`action_step=4` 的推理次数约为步数的四分之一。成功率与延迟需要分开解读。

## 限制

- 单 seed、每设置 10 集，Wilson 95% 置信区间很宽，不能作为正式模型排名。
- Octo 与 SpatialVLA 只跑了 T1/T2，任务覆盖不完整。
- Octo 使用 CPU 后端，延迟不可与 CUDA 模型直接比较。
- 这 160 + 80 集同时用于模型筛选和趋势判断，存在选择偏差；不能当作冻结后的正式评测。
- 没有训练或微调，因此不存在 Base/Final 对照，也没有 Final checkpoint。
