#!/bin/bash

version=`python -c "import torch; print(torch.compiled_with_cxx11_abi())"`
version=${version^^}

echo "Will download the flash attention package"
sleep 2
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu118torch2.1cxx11abi${version}-cp310-cp310-linux_x86_64.whl -O flash_attn-2.3.2+cu118torch2.1cxx11abi${version}-cp310-cp310-linux_x86_64.whl

echo "Will run command: pip install flash_attn-2.3.2+cu118torch2.1cxx11abi${version}-cp310-cp310-linux_x86_64.whl"
sleep 2
pip install flash_attn-2.3.2+cu118torch2.1cxx11abi${version}-cp310-cp310-linux_x86_64.whl
