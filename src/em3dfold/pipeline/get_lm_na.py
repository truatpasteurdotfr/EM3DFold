import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from em3dfold.utils.fasta_utils import FASTASequence, filter_small_sequences
from em3dfold.utils.torch_utils import clear_cuda_cache, get_module_device


def read_fasta(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith(">"):
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


def crop_long_chain(chain: FASTASequence, max_chain_length: int = 1024):
    """
    Split an overlong sequence into overlapping windows with 50% overlap.
    """
    length = len(chain.seq)
    i = 0
    chain_crops = []
    chain_starts = []

    while i < length:
        if i + max_chain_length < length:
            chain_crops.append(chain.seq[i : i + max_chain_length])
            chain_starts.append(i)
            i += max_chain_length // 2
        else:
            chain_crops.append(chain.seq[i:])
            chain_starts.append(i)
            break

    return chain_crops, chain_starts


def crop_long_chains(
    sequences: List[FASTASequence],
    seq_names: List[str],
    max_chain_length: int = 1024,
):
    """
    Crop multiple sequences and record how each original sequence maps to crops.
    """
    new_sequences = []
    chain_ids = []
    old_to_new_sequence = {}
    j = 0

    for sequence, seq_name in zip(sequences, seq_names):
        old_to_new_sequence[seq_name] = {}
        old_to_new_sequence[seq_name]["full_seq_len"] = len(sequence.seq)

        if len(sequence.seq) > max_chain_length:
            cropped_chain_list, chain_starts = crop_long_chain(
                sequence, max_chain_length=max_chain_length
            )
            chain_ids += [f"{seq_name}_{i}" for i in range(len(cropped_chain_list))]
            new_sequences += cropped_chain_list
            old_to_new_sequence[seq_name]["mapping"] = [
                j + k for k in range(len(cropped_chain_list))
            ]
            old_to_new_sequence[seq_name]["multi_part"] = True
            old_to_new_sequence[seq_name]["chain_starts"] = chain_starts
            j += len(cropped_chain_list)
        else:
            chain_ids += [f"{seq_name}_0"]
            new_sequences += [sequence.seq]
            old_to_new_sequence[seq_name]["mapping"] = [j]
            old_to_new_sequence[seq_name]["multi_part"] = False
            old_to_new_sequence[seq_name]["chain_starts"] = [0]
            j += 1

    return new_sequences, chain_ids, old_to_new_sequence


def empty_transformer_results(
    seq_name: str,
    seq_len: int,
    emb_dim: int,
    device: str = "cpu",
):
    """
    Create an empty result container for stitching long chains back together.
    """
    results = {}
    results["label"] = seq_name
    results["representations"] = torch.zeros(seq_len, emb_dim, device=device)
    results["mean_representations"] = torch.zeros(emb_dim, device=device)
    results["str_length"] = seq_len
    return results


def process_transformer_result(
    result: Dict,
    str_length: int,
    batch_idx: int = 0,
    seq_name: str = None,
):
    """
    Process language-model output for a single sequence.

    Assumes result["representation"] has shape (B, L, C) and that L matches
    the original sequence length without BOS/EOS tokens.
    """
    processed_result = {}
    if seq_name is not None:
        processed_result["label"] = seq_name

    rep = result["representation"][batch_idx, :str_length].cpu().clone()

    processed_result["representations"] = rep
    processed_result["mean_representations"] = rep.mean(0).clone()
    processed_result["str_length"] = str_length
    return processed_result


def collate_sequence_results(
    seq_name: str,
    batch_results: List[Dict],
    sequence_mapping: Dict,
):
    """
    Stitch cropped window results back into a full-length sequence result.
    Overlapping regions are averaged.
    """
    emb_dim = batch_results[sequence_mapping["mapping"][0]]["representations"].shape[-1]

    full_result = empty_transformer_results(
        seq_name=seq_name,
        seq_len=sequence_mapping["full_seq_len"],
        emb_dim=emb_dim,
        device="cpu",
    )

    sequence_results = [batch_results[i] for i in sequence_mapping["mapping"]]
    representation_counts = torch.zeros_like(full_result["representations"])

    for result, chain_start in zip(sequence_results, sequence_mapping["chain_starts"]):
        str_length = result["str_length"]

        full_result["representations"][
            chain_start : chain_start + str_length
        ] += result["representations"]

        representation_counts[
            chain_start : chain_start + str_length
        ] += 1

    full_result["representations"] /= representation_counts + 1e-6
    full_result["mean_representations"] = (
        full_result["representations"].mean(0).clone()
    )

    return full_result


@torch.no_grad()
def run_transformer_on_fasta(
    model,
    alphabet,
    raw_chains,
    sequence_names,
    device=None,
    max_chain_length=1000,
    use_amp=True,
):
    """
    Run the nucleotide language model on FASTA sequences with crop-and-stitch
    handling for long chains.

    Returns:
        {
            seq_name: {
                "label": str,
                "representations": (L, C),
                "mean_representations": (C,),
                "str_length": int,
            }
        }
    """
    if device is None:
        device = get_module_device(model)

    chains, sequence_names = filter_small_sequences(raw_chains, sequence_names)
    chains, updated_sequence_names, old_to_new_sequence = crop_long_chains(
        chains,
        sequence_names,
        max_chain_length=max_chain_length,
    )

    batch_results = []

    for seq_name, seq in zip(updated_sequence_names, chains):
        tokens = torch.tensor(
            alphabet.batch_tokenize([seq]),
            dtype=torch.int64,
            device=device,
        )

        if use_amp and "cuda" in str(device):
            with torch.cuda.amp.autocast():
                output = model(tokens)
        else:
            output = model(tokens)

        batch_results.append(
            process_transformer_result(
                output,
                str_length=len(seq),
                batch_idx=0,
                seq_name=seq_name,
            )
        )

    full_result = {}
    for old_seq_name in old_to_new_sequence:
        if not old_to_new_sequence[old_seq_name]["multi_part"]:
            result = batch_results[old_to_new_sequence[old_seq_name]["mapping"][0]]
            result["label"] = old_seq_name
            full_result[old_seq_name] = result
        else:
            full_result[old_seq_name] = collate_sequence_results(
                old_seq_name,
                batch_results,
                old_to_new_sequence[old_seq_name],
            )

    return full_result


def get_lm_embeddings(
    lang_model,
    alphabet,
    sequences,
    max_chain_length=1000,
):
    """
    Return concatenated residue embeddings for nucleotide sequences.

    Returns:
        np.ndarray with shape (sum(L_i), C)
    """
    try:
        sequences = [FASTASequence(seq, "", "A") for seq in sequences]
        seq_names = [str(x) for x in range(len(sequences))]

        result = run_transformer_on_fasta(
            model=lang_model,
            alphabet=alphabet,
            raw_chains=sequences,
            sequence_names=seq_names,
            max_chain_length=max_chain_length,
        )

        lm_embeddings = np.concatenate(
            [result[s]["representations"].cpu().numpy() for s in seq_names],
            axis=0,
        )

    except KeyError:
        raise BadFastaFile(
            "Fasta file is badly formatted. "
            "The issue is most likely that the fasta file has a sequence made "
            "entirely of X, or similar issues."
        )

    return lm_embeddings


def filter_seqs(sequences):
    # filter to have real protein residues
    new_sequences = []
    for sequence in sequences:
        new_sequence = []
        for ch in sequence:
            if ch not in ["A", "C", "G", "U", "T"]:
                continue

            new_sequence.append(ch)
        new_sequence = "".join(new_sequence)

        if len(new_sequence) > 2:
            new_sequences.append(new_sequence)
    return new_sequences


def add_args(parser):
    parser.add_argument(
        "--input-path",
        "--i",
        required=True,
        help="Input FASTA file or item list file",
    )
    parser.add_argument(
        "--output-dir",
        default="na_lms",
        help="Directory to save NA LM embeddings",
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
        help="Optional shared directory for RiNALMo weights",
    )
    return parser


def _load_rinalmo_model(device, lm_weights_dir=None):
    local_rinalmo_dir = Path(__file__).resolve().parents[1] / "rinalmo" / "RiNALMo-1.0"
    if str(local_rinalmo_dir) not in sys.path:
        sys.path.insert(0, str(local_rinalmo_dir))
    import rinalmo.pretrained as pretrained

    pretrained_weights_path = None
    if lm_weights_dir:
        shared_dir = Path(lm_weights_dir).expanduser().resolve()
        candidate_paths = [
            shared_dir / "rinalmo_giga_pretrained.pt",
            shared_dir / "giga-v1.pt",
            shared_dir / "rinalmo" / "rinalmo_giga_pretrained.pt",
            shared_dir / "rinalmo" / "giga-v1.pt",
        ]
        pretrained_weights_path = next(
            (path for path in candidate_paths if path.exists()),
            None,
        )
        if pretrained_weights_path is None:
            pretrained.DEFAULT_CACHE_DIR = shared_dir / "rinalmo"
        else:
            config = pretrained.model_config("giga")
            model = pretrained.RiNALMo(config)
            alphabet = pretrained.Alphabet(**config["alphabet"])
            model.load_state_dict(torch.load(pretrained_weights_path, map_location="cpu"))
            model = model.to(device)
            model.eval()
            return model, alphabet

    model, alphabet = pretrained.get_pretrained_model(model_name="giga-v1")
    model = model.to(device)
    model.eval()
    return model, alphabet


def _resolve_items(input_path):
    if input_path.endswith(".fa") or input_path.endswith(".fasta"):
        return [Path(input_path).stem]

    with open(input_path, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def _get_embeddings_or_empty(model, alphabet, fasta_path, *, max_chain_length, convert_t_to_u=False):
    if not os.path.exists(fasta_path):
        return np.zeros((0, 1280), dtype=np.float32)

    sequences = filter_seqs(read_fasta(fasta_path))
    if convert_t_to_u:
        sequences = [seq.replace("T", "U") for seq in sequences]
    if not sequences:
        return np.zeros((0, 1280), dtype=np.float32)

    return get_lm_embeddings(
        lang_model=model,
        alphabet=alphabet,
        sequences=sequences,
        max_chain_length=max_chain_length,
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model, alphabet = _load_rinalmo_model(
        args.device,
        lm_weights_dir=args.lm_weights_dir,
    )
    print("# Done load model")

    items = _resolve_items(args.input_path)
    for item in items:
        output_path = os.path.join(args.output_dir, f"{item}.npy")
        try:
            rna_lm_embeddings = _get_embeddings_or_empty(
                model,
                alphabet,
                f"seqres/{item}_rna.fa",
                max_chain_length=args.max_chain_length,
            )
            print(rna_lm_embeddings.shape)

            dna_lm_embeddings = _get_embeddings_or_empty(
                model,
                alphabet,
                f"seqres/{item}_dna.fa",
                max_chain_length=args.max_chain_length,
                convert_t_to_u=True,
            )
            print(dna_lm_embeddings.shape)

            lm_embeddings = np.concatenate([rna_lm_embeddings, dna_lm_embeddings], axis=0)
            print(lm_embeddings.shape)

            if len(lm_embeddings) == 0:
                raise Exception("No RNA or DNA sequences")

            np.save(output_path, lm_embeddings)
            print(f"# Saved to {output_path}")
        except Exception as e:
            lm_embeddings = np.zeros((3, 1280), dtype=np.float32)
            np.save(output_path, lm_embeddings)
            print(e, f"# Maybe no sequence but save a dummy tensor len = 3 to {output_path}", flush=True)

    del model
    del alphabet
    clear_cuda_cache(args.device, note="get_lm_na")


if __name__ == "__main__":
    import argparse

    args = add_args(argparse.ArgumentParser()).parse_args()
    main(args)
