#!/usr/bin/env bash
set -euo pipefail
ROOT=${PI0_ROOT:-/root/autodl-tmp/pi0_bridge_a}
OUT="$ROOT/models/pi0_original_raw"
LOG="$ROOT/logs/download_pi0_direct.log"
mkdir -p "$OUT" "$ROOT/logs"
source /etc/network_turbo >/dev/null 2>&1 || true
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] direct download resume"
aria2c -x 16 -s 16 -k 1M -c --max-tries=20 --retry-wait=5 --file-allocation=none --summary-interval=10 \
  -d "$OUT" -o model.safetensors \
  "https://huggingface.co/lerobot/pi0/resolve/main/model.safetensors"
size=$(stat -c %s "$OUT/model.safetensors")
echo "[verify] size=$size"
[ "$size" -eq 14005623256 ]
sha=$(sha256sum "$OUT/model.safetensors" | awk "{print \$1}")
echo "[verify] sha256=$sha"
[ "$sha" = "e1bbcefd55bab3ddff07ead42f4773505f07c68c9694ae44d636d86733dfebc1" ]
echo "[$(date -Is)] download done"
