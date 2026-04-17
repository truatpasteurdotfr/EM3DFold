#!/bin/bash

# path to conda
conda_dir=$1

if [[ $# -lt 1 ]]; then
    echo "usage: bash install.sh /path/to/your/conda"
    exit 1
fi

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

# 0 check existence
if [[ ! -e ${conda_dir}/bin/conda ]]; then
    echo "Conda is not in this dir -> ${conda_dir}"
    exit 1
fi

source "${conda_dir}/bin/activate"

echo "1 Create conda env"
conda env create -f env.yml
check_last $? "Failed at step 1, unable to create conda env"

echo "2 Activate env"
conda activate em3dfold
check_last $? "Failed at step 2, unable to activate em3dfold env"

echo "3 Install flash attention"
cd flash_attn_whl && bash install_flash_attn.sh && cd ..
check_last $? "Failed at step 3, unable to install flash attention"
echo "Done install base env"

echo "4 Install EM3DFold"
pip install .
echo "Done install EM3DFold"

# Check installation
em3dfold --help
echo "Done check installation"
