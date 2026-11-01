# Copy this file to env.sh and edit PI0_ROOT before running anything.
# All paths below default to a single runtime root that contains repos/, models/,
# envs/, runs/ and logs/. Model weights and datasets are NOT part of this repo.

export PI0_ROOT="${PI0_ROOT:-/root/autodl-tmp/pi0_bridge_a}"
export ROOT="$PI0_ROOT"

# External evaluation dependency (INT-ACT checkout, not vendored here).
export INTACT_REPO="${INTACT_REPO:-$ROOT/repos/INT-ACT}"
export REPO="$INTACT_REPO"

# Runtime caches and logs.
export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$ROOT/envs/train}"
export VLA_DATA_DIR="${VLA_DATA_DIR:-$ROOT/data}"
export VLA_LOG_DIR="${VLA_LOG_DIR:-$ROOT/runs}"

# Python interpreters.
export TRAIN_PYTHON="${TRAIN_PYTHON:-$ROOT/envs/train/bin/python}"
export SIMPLER_ENV="${SIMPLER_ENV:-$ROOT/envs/simpler}"
export OCTO_PYTHON="${OCTO_PYTHON:-$ROOT/envs/octo/bin/python}"

# Local replacement for the gated google/paligemma-3b-pt-224 tokenizer.
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-$ROOT/models/paligemma_tokenizer}"

# SimplerEnv needs Vulkan to see the NVIDIA GPU.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/my_nvidia_icd.json}"

# Optional mirrors; unset if your network can reach Hugging Face directly.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"

export TOKENIZERS_PARALLELISM=false
