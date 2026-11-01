#!/usr/bin/env bash
set -euo pipefail
ROOT=${PI0_ROOT:-/root/autodl-tmp/pi0_bridge_a}
OUT="$ROOT/models/pi0_original_raw"
LOG="$ROOT/logs/download_pi0_original.log"
mkdir -p "$OUT" "$ROOT/logs"
source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT=https://hf-mirror.com
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] download resume"
for f in config.json README.md policy_preprocessor.json policy_postprocessor.json; do
  echo "[download] $f"
  curl -L --fail --retry 8 --retry-delay 2 --continue-at - \
    -o "$OUT/$f" "https://huggingface.co/lerobot/pi0/resolve/main/$f"
done
if [ ! -f "$OUT/model.safetensors" ] || [ "$(stat -c %s "$OUT/model.safetensors")" -ne 14005623256 ]; then
  echo "[download] model.safetensors via aria2c mirror"
  aria2c -x 8 -s 8 -k 1M -c --file-allocation=none --summary-interval=10 \
    -d "$OUT" -o model.safetensors \
    "https://hf-mirror.com/lerobot/pi0/resolve/main/model.safetensors"
fi
size=$(stat -c %s "$OUT/model.safetensors")
echo "[verify] size=$size"
[ "$size" -eq 14005623256 ]
sha=$(sha256sum "$OUT/model.safetensors" | awk "{print \$1}")
echo "[verify] sha256=$sha"
[ "$sha" = "e1bbcefd55bab3ddff07ead42f4773505f07c68c9694ae44d636d86733dfebc1" ]
echo "[$(date -Is)] download done"
