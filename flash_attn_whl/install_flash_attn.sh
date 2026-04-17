#!/bin/bash

set -euo pipefail

fail() {
  echo "$*" >&2
  exit 1
}

run_step() {
  local msg="$1"
  shift
  "$@" || fail "$msg"
}

version="$(python -c "import torch; print(torch.compiled_with_cxx11_abi())")"
version=${version^^}
wheel_name="flash_attn-2.3.2+cu118torch2.1cxx11abi${version}-cp310-cp310-linux_x86_64.whl"
wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/${wheel_name}"

echo "Will download the flash attention package"
sleep 2
run_step "Failed to download flash attention wheel" \
  wget "${wheel_url}" -O "${wheel_name}"

echo "Will run command: pip install ${wheel_name}"
sleep 2
run_step "Failed to install flash attention wheel" \
  pip install "${wheel_name}"
