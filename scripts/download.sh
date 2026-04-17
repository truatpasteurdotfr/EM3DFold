#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EM3DFOLD_DIR="${ROOT_DIR}/em3dfold"
MODEL_WEIGHTS_DIR="${EM3DFOLD_DIR}/weights"
LM_WEIGHTS_DIR="${EM3DFOLD_DIR}/lm_weights"

# Override EM3DFOLD_WEIGHTS_URL if your hosted EM3DFold weights tarball uses a different path.
EM3DFOLD_WEIGHTS_URL="${EM3DFOLD_WEIGHTS_URL:-http://huanglab.phys.hust.edu.cn/EM3DFold/weights/weights.tgz}"
RINALMO_WEIGHTS_URL="${RINALMO_WEIGHTS_URL:-https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt}"
ESM2_WEIGHTS_URL="${ESM2_WEIGHTS_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt}"

function check_last() {
  local status=$1
  local msg=$2

  if [ ! "$status" -eq 0 ]; then
    if [ -n "$msg" ]; then
      echo "$msg"
    fi
    exit 1
  fi
}

function require_cmd() {
  local cmd=$1
  "$cmd" --help >/dev/null 2>&1 || {
    echo "Failed to detect '$cmd', maybe '$cmd' is not installed?"
    exit 1
  }
}

echo "Checking required commands"
require_cmd wget
require_cmd tar

mkdir -p "${MODEL_WEIGHTS_DIR}"
mkdir -p "${LM_WEIGHTS_DIR}"

echo "1. Download EM3DFold pretrained weights"
cd "${MODEL_WEIGHTS_DIR}"
wget "${EM3DFOLD_WEIGHTS_URL}" -O weights.tgz
check_last $? "Failed at step 1, unable to download EM3DFold pretrained weights"
tar -zxf weights.tgz
check_last $? "Failed at step 1, unable to extract EM3DFold pretrained weights"
cd "${ROOT_DIR}"

echo "2. Download RiNALMo pretrained weights"
wget "${RINALMO_WEIGHTS_URL}" -O "${LM_WEIGHTS_DIR}/rinalmo_giga_pretrained.pt"
check_last $? "Failed at step 2, unable to download RiNALMo weights"

echo "3. Download ESM2 pretrained weights"
wget "${ESM2_WEIGHTS_URL}" -O "${LM_WEIGHTS_DIR}/esm2_t33_650M_UR50D.pt"
check_last $? "Failed at step 3, unable to download ESM2 weights"

echo "Done download"
echo "EM3DFold model weights: ${MODEL_WEIGHTS_DIR}"
echo "LM weights: ${LM_WEIGHTS_DIR}"
echo "Use"
echo "> em3dfold build --weights-dir ${MODEL_WEIGHTS_DIR} --lm-weights-dir ${LM_WEIGHTS_DIR} ..."
