# 第三方依赖

本仓库**不复制**第三方源码，只记录评测所需的固定版本与获取方式。模型权重、数据集和虚拟环境也不随仓库分发。

## 固定版本

| 项目 | Commit | 来源 | 用途 |
|---|---|---|---|
| INT-ACT | `66ae3ed4719c68cb2f3f22868015a7a171b04e1f` | <https://github.com/ai4ce/INT-ACT.git> | SimplerEnv 环境适配器、`bridge_statistics.json` |
| LeRobot | `35f6e02315dcb3fa25f3a740478265c6c793a95e` | <https://github.com/IrvingF7/lerobot.git>（branch `pi0`） | π0 policy、数据与归一化实现 |
| SimplerEnv | `fea52fbb9e0da8a2e4e7e5a155b8e5b7f9dd5b87` | <https://github.com/juexZZ/SimplerEnv.git> | WidowX 仿真环境 |
| ManiSkill2_real2sim | `eeb04c788feafdf08f4565bcda34370e5a555325` | <https://github.com/juexZZ/ManiSkill2_real2sim.git> | SimplerEnv 的仿真后端 |

## 获取方式

```bash
export PI0_ROOT=/path/to/pi0_bridge_a

git clone https://github.com/ai4ce/INT-ACT.git "$PI0_ROOT/repos/INT-ACT"
cd "$PI0_ROOT/repos/INT-ACT"
git checkout 66ae3ed4719c68cb2f3f22868015a7a171b04e1f
git submodule update --init \
  third_party/lerobot \
  third_party/SimplerEnv \
  third_party/ManiSkill2_real2sim
```

说明：

- `third_party/lerobot` 指向 `IrvingF7/lerobot` 的 `pi0` 分支；`third_party/SimplerEnv` 与 `third_party/ManiSkill2_real2sim` 指向作者 fork。
- INT-ACT 还有 `LIBERO`、`ManiSkill`、`SimplerEnv_MS3` 等子模块，本预实验不使用，可以不初始化。
- 不建议直接使用当前 LeRobot `main`：本 checkpoint 依赖旧版 `lerobot.common.policies.pi0` 接口。

## tokenizer patch

`google/paligemma-3b-pt-224` 是 gated repo。预实验使用公开 Gemma tokenizer，并对 LeRobot 打一个最小 patch：

```bash
cd "$PI0_ROOT/repos/INT-ACT/third_party/lerobot"
git apply /path/to/MLPJ-PreExperiment/patches/lerobot-tokenizer-path.patch
export PALIGEMMA_TOKENIZER_PATH="$PI0_ROOT/models/paligemma_tokenizer"
```

patch 的作用：

- 增加 `import os`；
- 把 `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")` 改为读取 `PALIGEMMA_TOKENIZER_PATH`，未设置时回退到原模型名。

## 许可证

- LeRobot 采用 Apache License 2.0（以仓库内 LICENSE 为准）。
- INT-ACT 仓库当前没有附带 LICENSE 文件；本仓库只通过文档给出获取方式，不复制其源码。
- SimplerEnv、ManiSkill2_real2sim 的许可证以其上游仓库为准。
- 使用或再分发第三方代码前，请自行核对其许可证。
