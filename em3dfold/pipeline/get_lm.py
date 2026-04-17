import glob
import os
from pathlib import Path

import esm
import numpy as np
import tqdm
import torch

from em3dfold.utils.apply_sequence_transformer import run_transformer_on_fasta
from em3dfold.utils.fasta_utils import FASTASequence
from em3dfold.utils.torch_utils import clear_cuda_cache

from em3dfold.polymer_utils.residue_constants import prot_restype1


def read_fasta(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs

class BadFastaFile(Exception):
    pass


def get_lm_embeddings(
    lang_model, batch_converter, sequences, max_chain_length=1000
):
    try:
        sequences = [FASTASequence(seq, "", "A") for seq in sequences]
        seq_names = [str(x) for x in range(len(sequences))]
        result = run_transformer_on_fasta(
            lang_model,
            batch_converter,
            sequences,
            seq_names,
            repr_layers=[33],
            max_chain_length=max_chain_length,
        )
        lm_embeddings = np.concatenate(
            [result[s]["representations"][33].cpu().numpy() for s in seq_names], axis=0,
        )
    except KeyError:
        raise BadFastaFile(
            f"Fasta file is badly formatted."
            f"The issue is most likely that the Fasta file has a sequence made entirely of X, or similar issues."
        )
    return lm_embeddings


def add_args(parser):
    parser.add_argument(
        "--input-path", "--i", required=True, help="Input PDB/mmCIF files"
    )
    parser.add_argument("--device", default="cuda:0", help="Which device to run on")
    parser.add_argument(
        "--max-chain-length",
        type=int,
        default=1000,
        help="Maximum chain length for the transformer",
    )
    parser.add_argument(
        "--lm-weights-dir",
        help="Optional shared directory for ESM weights",
    )
    return parser


def _load_esm_model(device, lm_weights_dir=None):
    if lm_weights_dir:
        shared_dir = Path(lm_weights_dir).expanduser().resolve()
        local_ckpt = shared_dir / "esm2_t33_650M_UR50D.pt"
        if local_ckpt.exists() and hasattr(esm.pretrained, "load_model_and_alphabet_local"):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(local_ckpt))
        else:
            os.environ["TORCH_HOME"] = str(shared_dir)
            torch.hub.set_dir(str(shared_dir))
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    model = model.eval().to(device)
    return model, alphabet


def _resolve_items(input_path):
    if input_path.endswith(".fa") or input_path.endswith(".fasta"):
        return [Path(input_path).stem]
    with open(input_path, "r") as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def main(args):
    model, alphabet = _load_esm_model(args.device, lm_weights_dir=args.lm_weights_dir)
    batch_converter = alphabet.get_batch_converter()
    print("# Done load model")

    items = _resolve_items(args.input_path)

    for item in items:
        try:
            sequences = read_fasta(f"seqres/{item}_prot.fa")

            new_sequences = []
            for sequence in sequences:
                new_sequence = []
                for ch in sequence:
                    if ch in prot_restype1 and ch != "1":
                        new_sequence.append(ch)
                new_sequence = "".join(new_sequence)

                if len(new_sequence) > 2:
                    new_sequences.append(new_sequence)

            sequences = new_sequences

            lm_embeddings = get_lm_embeddings(
                model, batch_converter, sequences, max_chain_length=args.max_chain_length
            )

            print(lm_embeddings.shape)

            np.save(
                f"data/esms/{item}.npy",
                lm_embeddings,
            )
            print(f"# Save to data/esms/{item}.npy")

        except Exception as e:
            lm_embeddings = np.zeros((3, 1280), dtype=np.float32)

            np.save(
                f"data/esms/{item}.npy",
                lm_embeddings,
            )
            print(e, f"# Maybe no sequence but save a dummy tensor len = 3 to data/esms/{item}.npy")

    del model
    del alphabet
    del batch_converter
    clear_cuda_cache(args.device, note="get_lm")

if __name__ == "__main__":
    import argparse
    args = add_args(argparse.ArgumentParser()).parse_args()
    main(args)
