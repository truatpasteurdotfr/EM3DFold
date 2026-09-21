git clone https://github.com/truatpasteurdotfr/EM3DFold
#git clone https://github.com/huang-laboratory/EM3DFold
cd EM3DFold
pixi install
pixi run python3 -c "import torch; print(torch.compiled_with_cxx11_abi())"
pixi run bash flash_attn_whl/install_flash_attn.sh
pixi run pip install .
# mkdir /home/tru/pixi.d/em3dfold/download
# bash scripts/download.sh /home/tru/pixi.d/em3dfold/download
