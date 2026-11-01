#!/bin/bash
set -euo pipefail
ROOT=${PI0_ROOT:-/root/autodl-tmp/pi0_bridge_a}
mkdir -p "$ROOT/models/pi0_bridge"
URL=$(curl -sI "https://hf-mirror.com/juexzz/INTACT-pi0-finetune-bridge/resolve/main/model.safetensors" | grep -i '^location:' | sed 's/^[Ll]ocation: //' | tr -d '\r' | tail -n 1)
echo "URL length: ${#URL}"
nohup aria2c -c -x 8 -s 8 -k 1M --file-allocation=none --console-log-level=warn -d "$ROOT/models/pi0_bridge" -o model.safetensors "$URL" > "$ROOT/models/model_download.log" 2>&1 &
echo "PID=$!"