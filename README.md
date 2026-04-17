# EM3DFold

## Overview
EM3DFold is a software package for automatic protein, RNA, and DNA (and small molecules in the upcoming update) structure modeling from cryo-EM density maps.

## Requirements
**Platform**: Linux.

**GPU**: A GPU with at least 12 GB VRAM is required.

**CUDA**: CUDA >= 11.8 is required.

**Disk Storage**: EM3DFold pretrained weights and language model weights require at least 4 GB free disk space.

## Installation
#### 0. Install conda

We recommend using conda to manage the environment. If conda is not installed, please install Miniforge, Miniconda or Anaconda first.

[Click here](https://docs.conda.io/en/latest/miniconda.html) to see guidance to install Miniconda.

#### 1. Download EM3DFold

Download EM3DFold via `git` (recommended):
```bash
git clone https://github.com/huang-laboratory/EM3DFold.git
cd EM3DFold
```

If you do not use git, download the source archive via `wget` and extract it manually, e.g. with command:
```bash
wget https://github.com/huang-laboratory/EM3DFold/archive/refs/heads/main.zip
unzip main.zip
cd EM3DFold-main
```

#### 2. Create conda env and install dependencies

We write the tedious installation steps into one script, so installation is one command like:
```bash
# Make sure you are in the EM3DFold directory
bash scripts/install.sh /path/to/your/conda
```

If you are familiar with conda/pip/shell or you encounter problems when running the above command, please execute commands in `install.sh` step-by-step.

#### 3. Download pretrained weights

The provided `download.sh` scripts automatically downloads the pretrained weights of EM3DFold and needed language models into specified directory:

```bash
# Download all weights
bash scripts/download.sh /path/to/save/weights/
```

## Usage
Running EM3DFold is straight forward with one command like
```bash
em3dfold build --map/-m MAP.mrc \
    --protein/-p PROTEIN.fa \
    --rna/-r RNA.fa \
    --dna/-d DNA.fa \
    --protein-template/-pt PROTEIN_TEMPLATE_0.cif [...] \
    --output OUT \
    --device 0
```

- The cryo-EM density map and output directory are **required**.
- Input FASTA files can each include multiple sequences.
- You can provide only `--protein`, only `--rna`, only `--dna`, or any valid combination of them.
- If you launch >1 modeling job, the output directory **must** be different for each run.
- Input protein template(s) can either be a single chain PDB/mmCIF file or a multi-chain PDB/mmCIF file.
- By default, intermediate results (predicted maps recycled structures) will be removed. Use `--keep-temp-files` if you want to keep them. 
- Currently, only supports protein templates, nucleic-acids support depends on the community needs.

**Check the command usage any time you forget how to run EM3DFold**
```bash
em3dfold build --help
```

## Examples
#### 1. Protein denovo modeling
```bash
em3dfold build --map MAP.mrc \ 
  --protein protein.fa \ 
  --output out_protein \ 
  --device 0
```

#### 2. RNA denovo modeling
```bash
em3dfold build --map MAP.mrc \ 
  --rna rna.fa \ 
  --output out_rna \ 
  --device 0
```

#### 3. Protein-RNA complex denovo modeling
```bash
em3dfold build --map MAP.mrc \ 
  --protein protein.fa \ 
  --rna rna.fa \ 
  --output out_complex \ 
  --device 0
```

#### 4. Protein modeling with templates
```bash
em3dfold build --map MAP.mrc \ 
  --protein protein.fa \ 
  --protein-template PROTEIN_TEMPLATE_0.cif [...] \
  --output out_protein \ 
  --device 0
```

## Notes
- Sequence branches are only enabled when the corresponding CLI argument is explicitly provided and the formatted sequence file is non-empty.
- Protein and nucleic-acid chains are modeled together in a unified `inferlm_v2` run.
- For nucleic acids, EM3DFold can compare network-predicted residue logits and sampled `na_aa_logits`, and use the higher-scoring alignment.
- The main output model is written to `OUT/output.cif`.

## Trouble shooting
- **No module named "xxx"**: package `xxx` is missing in your current environment. Install it with `pip install xxx` or `conda install xxx`.

- **em3dfold: command not found**: you are probably not in the correct environment, activate EM3DFold environment and run again.

- **CUDA / PyTorch related errors**: check whether your installed `torch` matches the local CUDA runtime and GPU driver.

## Citation
If you find EM3DFold useful in your work, please cite the corresponding papers.

```bibtex
@article{EM3DFold,
  title={Highly accurate protein-nucleic acid modeling from cryo-EM maps with EM3DFold},
  author={Tao Li, Sheng-You Huang},
  journal={bioRxiv},
  year={2026}
}

@article{EMProt,
  title={EMProt improves protein structure determination from cryo-EM maps},
  author={Tao Li, Ji Chen, Hao Li, Hong Cao and Sheng-You Huang},
  journal={Nature Structural & Molecylar Biology},
  year={2025}
}
```
