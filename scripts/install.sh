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

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found in PATH: $cmd"
}

if [[ $# -gt 0 ]]; then
  fail "This script no longer accepts a conda path. Usage: bash scripts/install.sh"
fi

require_cmd conda
eval "$(conda shell.bash hook)"

echo "1 Create conda env"
run_step "Failed at step 1, unable to create conda env" \
  conda env create -f env.yml

echo "2 Activate env"
run_step "Failed at step 2, unable to activate em3dfold env" \
  conda activate em3dfold

echo "3 Install flash attention"
run_step "Failed at step 3, unable to install flash attention" \
  bash -c "cd flash_attn_whl && bash install_flash_attn.sh"
echo "Done install base env"

echo "4 Install EM3DFold"
run_step "Failed at step 4, unable to install EM3DFold" \
  pip install .
echo "Done install EM3DFold"

# Check installation
run_step "Failed at step 5, em3dfold --help did not run successfully" \
  em3dfold --help
echo "Done check installation"
