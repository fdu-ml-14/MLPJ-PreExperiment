# 环境说明

本仓库的预实验在 Linux + NVIDIA GPU 上运行，推理和仿真分成两个独立环境。模型权重、数据集和虚拟环境都不随仓库分发。

## 1. 固定版本

```text
OS:            Ubuntu 22.04
Python:        3.10.12
GPU:           NVIDIA RTX PRO 6000 Blackwell 96GB（sm_120）
torch:         2.8.0+cu128
torchvision:   0.23.0+cu128
torchaudio:    2.8.0+cu128
transformers:  4.48.3
datasets:      2.21.0
fsspec:        2024.6.1
sapien:        2.2.2
setuptools:    80.10.2
uv:            0.13.4
```

关键点：

- **torch 2.6 不支持 Blackwell（sm_120）**，会报 `no kernel image is available`；必须用 2.8.0+cu128。
- 旧版 LeRobot 与 `datasets 5.x` 不兼容，锁 `datasets==2.21.0`、`fsspec==2024.6.1`。
- π0 推理环境（`envs/train`）与 SimplerEnv 环境依赖冲突较大，建议保持分开，不要合并。

## 2. 目录约定

```text
$PI0_ROOT/
├── repos/INT-ACT/          # 外部评测依赖，不随本仓库分发
├── models/                 # 权重与推理配置，不随本仓库分发
├── envs/train/             # Python 3.10 + torch + LeRobot
├── envs/simpler/           # Python 3.10 + sapien + SimplerEnv
├── runs/                   # 评测输出
├── logs/                   # 下载与运行日志
└── cache/hf/               # Hugging Face 缓存
```

复制环境模板并修改 `PI0_ROOT`：

```bash
cp env.example.sh env.sh
# 编辑 env.sh 中的 PI0_ROOT
source env.sh
```

## 3. 系统依赖

```bash
apt update
apt install -y git curl wget unzip tmux ffmpeg aria2 \
  build-essential libgl1 libglib2.0-0 \
  vulkan-tools libvulkan1 libsm6 libegl1
```

## 4. Vulkan

SimplerEnv 通过 Vulkan 使用 GPU。仅有 `nvidia-smi` 正常还不够，必须让 `vulkaninfo` 识别到 NVIDIA 物理 GPU。

创建 `/etc/vulkan/icd.d/my_nvidia_icd.json`：

```json
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
    "api_version": "1.3.277"
  }
}
```

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
vulkaninfo --summary
```

验收标准：输出中出现 NVIDIA GPU，且 `deviceType` 为 `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`，不是 CPU。

## 5. π0 推理环境（`envs/train`）

该环境用于 π0 加载与推理；本仓库不包含训练步骤。

```bash
cd "$PI0_ROOT/repos/INT-ACT"
export UV_PROJECT_ENVIRONMENT="$PI0_ROOT/envs/train"
uv sync
```

> 上游 `pyproject.toml` 钉的是 torch 2.6.0；Blackwell（sm_120）需要手动升级到 `2.8.0+cu128`，并保持 torchvision / torchaudio 版本匹配。

检查：

```bash
"$PI0_ROOT/envs/train/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
"$PI0_ROOT/envs/train/bin/python" -c "from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy; print('PI0 import OK')"
```

## 6. SimplerEnv 环境

```bash
cd "$PI0_ROOT/repos/INT-ACT/src/experiments/envs/simpler"
uv venv "$PI0_ROOT/envs/simpler" --python=3.10
source "$PI0_ROOT/envs/simpler/bin/activate"
uv pip install -r pyproject.toml
```

检查：

```bash
"$PI0_ROOT/envs/simpler/bin/python" -c "import simpler_env, sapien; print('SimplerEnv OK')"
```

## 7. 网络

下载模型或依赖时：

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

`apt` / `pip` / `uv` 前建议先取消代理：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

## 8. 总验收

```bash
nvidia-smi
vulkaninfo --summary
"$PI0_ROOT/envs/train/bin/python" -c "import torch; print(torch.cuda.get_device_name(0))"
"$PI0_ROOT/envs/train/bin/python" -c "import simpler_env; print('SimplerEnv OK')"
```
