import math
from collections import namedtuple
from itertools import count
import json
import os
from typing import List, Tuple

import numpy as np
import pyhmmer
import torch

from em3dfold.infer.hmm.match_to_sequence import MatchToSequence
from em3dfold.infer.hmm.aa_probs_to_hmm import aa_logits_to_hmm, alphabet_to_index

from em3dfold.utils.fasta_utils import (
    find_match_range,
    in_seq_dict,
    nuc_sequence_to_purpyr,
    remove_dots,
    remove_non_residue,
    sequence_match,
)
from em3dfold.polymer_utils.residue_constants import num_prot, restype_3_to_index

HMMAlignment = namedtuple(
    "HMMAlignment",
    [
        "sequence",
        "seq_idx",
        "res_idx",
        "key_start_match",
        "key_end_match",
        "match_score",
        "hmm_output_match_sequence",
        "exists_in_sequence_mask",
    ],
)


_HMM_ALIGNMENT_DEBUG_COUNTER = count()


def _keep_hmm_artifacts_enabled() -> bool:
    flag = os.environ.get("EM3DFOLD_KEEP_HMM_FILES", "")
    return flag.lower() not in {"", "0", "false", "no"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _write_hmm_alignment_debug(base_dir: str, payload: dict) -> None:
    if not _keep_hmm_artifacts_enabled():
        return
    os.makedirs(base_dir, exist_ok=True)
    debug_idx = next(_HMM_ALIGNMENT_DEBUG_COUNTER)
    path = os.path.join(base_dir, f"alignment_{debug_idx:05d}.json")
    with open(path, "w") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


def _disable_na_logits_aware_msa_score() -> bool:
    flag = os.environ.get("EM3DFOLD_NA_DISABLE_LOGITS_AWARE_MSA_SCORE", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _disable_na_hmm_logit_scale() -> bool:
    flag = os.environ.get("EM3DFOLD_NA_DISABLE_HMM_LOGIT_SCALE", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _disable_na_hmm_logit_scale_exact_only() -> bool:
    flag = os.environ.get("EM3DFOLD_NA_DISABLE_HMM_LOGIT_SCALE_EXACT_ONLY", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def expand_shared_na_logits_to_full_vocab(aa_logits: np.ndarray) -> np.ndarray:
    if aa_logits.shape[-1] == num_prot + 8:
        return aa_logits
    if aa_logits.shape[-1] != num_prot + 4:
        raise ValueError(
            f"Expected 24 or 28 aa logits channels, got {aa_logits.shape[-1]}"
        )

    expanded = np.full(
        aa_logits.shape[:-1] + (num_prot + 8,),
        fill_value=-100.0,
        dtype=aa_logits.dtype,
    )
    expanded[..., :num_prot] = aa_logits[..., :num_prot]
    expanded[..., num_prot : num_prot + 4] = aa_logits[..., num_prot:]
    expanded[..., num_prot + 4 :] = aa_logits[..., num_prot:]
    return expanded


def get_na_processed_msa_scores(
    aa_logits: np.ndarray,
    na_processed_msas: List[str],
    raw_na_sequences: List[str],
    alphabet_type: str,
) -> np.ndarray:
    assert alphabet_type in {"RNA", "DNA"}
    restype_temp_dict = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    aa_probs = torch.from_numpy(aa_logits).softmax(dim=-1).numpy()
    if alphabet_type == "RNA":
        aa_probs = aa_probs[:, num_prot + 4 : num_prot + 8]
    else:
        aa_probs = aa_probs[:, num_prot : num_prot + 4]

    scores = []
    for msa_idx, msa in enumerate(na_processed_msas):
        score = 0.0
        start_match, end_match, num_gaps_to_start = find_match_range(msa)
        seq_pos = start_match - num_gaps_to_start + 1
        pred_pos = 0
        original_seq = raw_na_sequences[msa_idx]
        for token in msa[start_match : end_match + 1]:
            if token in sequence_match:
                if token.isalpha():
                    score += float(
                        aa_probs[pred_pos, restype_temp_dict[original_seq[seq_pos - 1]]]
                    )
                pred_pos += 1
            if token in in_seq_dict:
                seq_pos += 1
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def get_aa_from_aalogits(aa_logits: np.ndarray, match_type: str = "") -> np.ndarray:
    aa_probs = np.exp(aa_logits - np.max(aa_logits, axis=-1, keepdims=True))
    aa_probs /= aa_probs.sum(axis=-1, keepdims=True)
    if match_type == "":
        if np.sum(aa_probs[:, num_prot : num_prot + 4]) > np.sum(
            aa_probs[:, num_prot + 4 : num_prot + 8]
        ):
            match_type = "DNA"
        else:
            match_type = "RNA"
    na_logits = np.maximum(
        aa_logits[:, num_prot : num_prot + 4],
        aa_logits[:, num_prot + 4 : num_prot + 8],
    )
    if match_type == "DNA":
        return np.argmax(na_logits, axis=-1) + num_prot
    if match_type == "RNA":
        return np.argmax(na_logits, axis=-1) + (num_prot + 4)
    raise ValueError(f"Unsupported match_type: {match_type}")


def get_hmm_alignment(
    aa_logits: np.ndarray,
    digital_prot_sequences: List[pyhmmer.easel.DigitalSequence] = [],
    digital_rna_sequences: List[pyhmmer.easel.DigitalSequence] = [],
    digital_dna_sequences: List[pyhmmer.easel.DigitalSequence] = [],
    do_pp: bool = False,
    raw_rna_sequences: List[str] = None,
    raw_dna_sequences: List[str] = None,
    confidence: np.ndarray = None,
    base_dir: str = "/tmp",
    is_nucleotide: bool = False,
) -> HMMAlignment:
    debug_payload = {
        "input_length": int(len(aa_logits)),
        "is_nucleotide": bool(is_nucleotide),
        "base_dir": base_dir,
    }
    aa_logits = expand_shared_na_logits_to_full_vocab(aa_logits)
    if not is_nucleotide:
        hmm = aa_logits_to_hmm(
            aa_logits, confidence=confidence, base_dir=base_dir, alphabet_type="amino"
        )
        msas = pyhmmer.hmmer.hmmalign(
            hmm, digital_prot_sequences, all_consensus_cols=True
        )
        processed_msas = msas.alignment
        prot_match_lengths = np.array([len(remove_non_residue(x)) for x in processed_msas])
        seq_idx = np.argmax(prot_match_lengths)
        msa_index_corr = get_msa_index_correspondence(processed_msas[seq_idx])
        index_dict = alphabet_to_index["amino"]
        match_sequence = msa_index_corr.sequence
        original_pred_seq = np.argmax(aa_logits[..., :num_prot], axis=-1)
        debug_payload.update({
            "match_type": "amino",
            "candidate_match_lengths": prot_match_lengths.tolist(),
            "selected_candidate_index": int(seq_idx),
        })
    else:
        if do_pp:
            assert raw_rna_sequences is not None or raw_dna_sequences is not None
        has_rna_seq = len(digital_rna_sequences) > 0
        has_dna_seq = len(digital_dna_sequences) > 0
        match_type = ""
        rna_match_lengths = []
        dna_match_lengths = []
        rna_processed_msa_scores = []
        dna_processed_msa_scores = []
        disable_logits_aware_msa_score = _disable_na_logits_aware_msa_score()
        disable_na_hmm_logit_scale = _disable_na_hmm_logit_scale()
        disable_na_hmm_logit_scale_exact_only = _disable_na_hmm_logit_scale_exact_only()
        disable_scale_this_run = disable_na_hmm_logit_scale or (disable_na_hmm_logit_scale_exact_only and (not do_pp))
        na_hmm_logits = aa_logits if disable_scale_this_run else aa_logits * math.log(8, 4)
        if has_rna_seq:
            hmm_rna = aa_logits_to_hmm(
                na_hmm_logits,
                confidence=confidence,
                base_dir=base_dir,
                alphabet_type="RNA" if not do_pp else "PP",
            )
            rna_processed_msas = pyhmmer.hmmer.hmmalign(
                hmm_rna, digital_rna_sequences, all_consensus_cols=True
            ).alignment
            rna_match_lengths = np.array([len(remove_non_residue(x)) for x in rna_processed_msas])
            rna_processed_msa_scores = get_na_processed_msa_scores(
                aa_logits=aa_logits,
                na_processed_msas=rna_processed_msas,
                raw_na_sequences=raw_rna_sequences,
                alphabet_type="RNA",
            )
            if disable_logits_aware_msa_score:
                rna_seq_idx = int(np.argmax(rna_match_lengths))
                rna_seq_val = float(np.max(rna_match_lengths))
            else:
                rna_seq_idx = int(np.argmax(rna_processed_msa_scores))
                rna_seq_val = float(np.max(rna_processed_msa_scores))
        if has_dna_seq:
            hmm_dna = aa_logits_to_hmm(
                na_hmm_logits,
                confidence=confidence,
                base_dir=base_dir,
                alphabet_type="DNA" if not do_pp else "PP",
            )
            dna_processed_msas = pyhmmer.hmmer.hmmalign(
                hmm_dna, digital_dna_sequences, all_consensus_cols=True
            ).alignment
            dna_match_lengths = np.array([len(remove_non_residue(x)) for x in dna_processed_msas])
            dna_processed_msa_scores = get_na_processed_msa_scores(
                aa_logits=aa_logits,
                na_processed_msas=dna_processed_msas,
                raw_na_sequences=raw_dna_sequences,
                alphabet_type="DNA",
            )
            if disable_logits_aware_msa_score:
                dna_seq_idx = int(np.argmax(dna_match_lengths))
                dna_seq_val = float(np.max(dna_match_lengths))
            else:
                dna_seq_idx = int(np.argmax(dna_processed_msa_scores))
                dna_seq_val = float(np.max(dna_processed_msa_scores))
        if has_rna_seq and has_dna_seq:
            match_type = "DNA" if rna_seq_val <= dna_seq_val else "RNA"
        elif has_rna_seq:
            match_type = "RNA"
        elif has_dna_seq:
            match_type = "DNA"

        debug_payload.update({
            "match_type": match_type,
            "na_candidate_selection": "match_length" if disable_logits_aware_msa_score else "processed_msa_score",
            "na_hmm_logit_scale": (
                "disabled"
                if disable_na_hmm_logit_scale
                else ("disabled_exact_only" if disable_na_hmm_logit_scale_exact_only else "log8_over_4")
            ),
            "rna_match_lengths": rna_match_lengths.tolist() if len(rna_match_lengths) else [],
            "dna_match_lengths": dna_match_lengths.tolist() if len(dna_match_lengths) else [],
            "rna_processed_msa_scores": rna_processed_msa_scores.tolist() if len(rna_processed_msa_scores) else [],
            "dna_processed_msa_scores": dna_processed_msa_scores.tolist() if len(dna_processed_msa_scores) else [],
            "selected_rna_index": int(rna_seq_idx) if has_rna_seq else None,
            "selected_dna_index": int(dna_seq_idx) if has_dna_seq else None,
            "selected_rna_value": float(rna_seq_val) if has_rna_seq else None,
            "selected_dna_value": float(dna_seq_val) if has_dna_seq else None,
        })

        if match_type == "":
            made_up_match = "-" * len(aa_logits)
            msa_index_corr = get_msa_index_correspondence(made_up_match)
            index_dict = alphabet_to_index["RNA"]
            seq_idx = len(digital_prot_sequences)
            original_pred_seq = get_aa_from_aalogits(aa_logits, match_type)
        elif match_type == "RNA":
            original_seq = None if not do_pp else raw_rna_sequences[rna_seq_idx]
            msa_index_corr = get_msa_index_correspondence(
                rna_processed_msas[rna_seq_idx], original_seq=original_seq
            )
            index_dict = alphabet_to_index["RNA"]
            seq_idx = rna_seq_idx + len(digital_prot_sequences)
            original_pred_seq = get_aa_from_aalogits(aa_logits, match_type)
        elif match_type == "DNA":
            original_seq = None if not do_pp else raw_dna_sequences[dna_seq_idx]
            msa_index_corr = get_msa_index_correspondence(
                dna_processed_msas[dna_seq_idx], original_seq=original_seq,
            )
            index_dict = alphabet_to_index["DNA"]
            seq_idx = (
                dna_seq_idx + len(digital_prot_sequences) + len(digital_rna_sequences)
            )
            original_pred_seq = get_aa_from_aalogits(aa_logits, match_type)
        match_sequence = msa_index_corr.sequence

    msa_sequence = np.array(
        [index_dict[x] if x in index_dict else -1 for x in match_sequence]
    )
    new_sequence = np.where(msa_sequence != -1, msa_sequence, original_pred_seq)

    match_score = len(remove_non_residue(match_sequence)) / len(match_sequence)
    debug_payload.update({
        "final_seq_idx": int(seq_idx),
        "match_score": float(match_score),
        "match_sequence": match_sequence,
        "res_idx": msa_index_corr.res_idx,
        "key_start_match": int(msa_index_corr.key_start_match),
        "key_end_match": int(msa_index_corr.key_end_match),
        "exists_in_sequence_mask": msa_index_corr.exists_in_sequence_mask,
        "original_pred_seq": original_pred_seq,
        "new_sequence": new_sequence,
    })
    _write_hmm_alignment_debug(base_dir, debug_payload)
    return HMMAlignment(
        sequence=new_sequence,
        seq_idx=seq_idx,
        res_idx=msa_index_corr.res_idx,
        key_start_match=msa_index_corr.key_start_match,
        key_end_match=msa_index_corr.key_end_match,
        match_score=match_score,
        hmm_output_match_sequence=match_sequence,
        exists_in_sequence_mask=msa_index_corr.exists_in_sequence_mask,
    )


def best_match_to_sequences(
    prot_sequences: List[str],
    rna_sequences: List[str],
    dna_sequences: List[str],
    chain_prot_mask: List[np.ndarray],
    chain_aa_logits: List[np.ndarray],
    chain_confidences: List[np.ndarray] = None,
    base_dir: str = "/tmp",
    do_pp: bool = False,
) -> MatchToSequence:
    alphabets = [
        pyhmmer.easel.Alphabet.amino(),
        pyhmmer.easel.Alphabet.rna(),
        pyhmmer.easel.Alphabet.dna() if not do_pp else pyhmmer.easel.Alphabet.rna(),
    ]
    digital_sequences = [
        [
            pyhmmer.easel.TextSequence(
                name=bytes(f"seq_{j}", encoding="utf-8"),
                sequence=seq,
            ).digitize(alphabets[i])
            for j, seq in enumerate(sequences)
        ]
        for i, sequences in enumerate([
            prot_sequences,
            [nuc_sequence_to_purpyr(seq) for seq in rna_sequences] if do_pp else rna_sequences,
            [nuc_sequence_to_purpyr(seq) for seq in dna_sequences] if do_pp else dna_sequences,
        ])
    ]

    if chain_confidences is None:
        chain_confidences = [None] * len(chain_aa_logits)

    (
        new_sequences,
        residue_idxs,
        sequence_idxs,
        key_start_matches,
        key_end_matches,
        match_scores,
        hmm_output_match_sequences,
        exists_in_sequence_mask,
        is_nucleotide_list,
    ) = ([], [], [], [], [], [], [], [], [])
    null_sequence_id = len(digital_sequences)
    for aa_logits, confidence, prot_mask in zip(
        chain_aa_logits, chain_confidences, chain_prot_mask
    ):
        chain_len = len(aa_logits)
        is_nucleotide = np.all(~prot_mask)
        if chain_len < 3:
            new_sequences.append(np.argmax(aa_logits, axis=-1))
            residue_idxs.append(np.arange(1, chain_len + 1))
            sequence_idxs.append(null_sequence_id)
            key_start_matches.append(1)
            key_end_matches.append(chain_len + 1)
            match_scores.append(0)
            null_sequence_id += 1
            hmm_output_match_sequences.append("-" * chain_len)
            exists_in_sequence_mask.append(np.ones(chain_len, dtype=int))
            is_nucleotide_list.append(is_nucleotide)
        else:
            hmm_alignment = get_hmm_alignment(
                aa_logits,
                digital_prot_sequences=digital_sequences[0],
                digital_rna_sequences=digital_sequences[1],
                digital_dna_sequences=digital_sequences[2],
                confidence=confidence,
                base_dir=base_dir,
                is_nucleotide=is_nucleotide,
                do_pp=do_pp,
                raw_rna_sequences=rna_sequences,
                raw_dna_sequences=dna_sequences,
            )

            new_sequences.append(hmm_alignment.sequence)
            residue_idxs.append(hmm_alignment.res_idx)
            sequence_idxs.append(hmm_alignment.seq_idx)
            key_start_matches.append(hmm_alignment.key_start_match)
            key_end_matches.append(hmm_alignment.key_end_match)
            match_scores.append(hmm_alignment.match_score)
            hmm_output_match_sequences.append(hmm_alignment.hmm_output_match_sequence)
            exists_in_sequence_mask.append(hmm_alignment.exists_in_sequence_mask)
            is_nucleotide_list.append(is_nucleotide)

    return MatchToSequence(
        new_sequences=new_sequences,
        residue_idxs=residue_idxs,
        sequence_idxs=sequence_idxs,
        key_start_matches=np.array(key_start_matches),
        key_end_matches=np.array(key_end_matches),
        match_scores=np.array(match_scores),
        hmm_output_match_sequences=hmm_output_match_sequences,
        exists_in_sequence_mask=exists_in_sequence_mask,
        is_nucleotide=is_nucleotide_list,
    )


MSAIndexCorrespondence = namedtuple(
    "MSAIndexCorrespondence",
    [
        "sequence",
        "res_idx",
        "key_start_match",
        "key_end_match",
        "exists_in_sequence_mask",
    ],
)


def get_msa_index_correspondence(
    msa: str, original_seq: str = None,
) -> MSAIndexCorrespondence:
    start_match, end_match, num_gaps_to_start = find_match_range(msa)

    idxs = []
    exists_in_sequence_mask = []
    matched_sequence = ""
    j = start_match - num_gaps_to_start + 1

    for s in msa[start_match : end_match + 1]:
        if s in sequence_match:
            if original_seq is None:
                matched_sequence += s
            else:
                matched_sequence += original_seq[j - 1] if s.isalpha() else "-"
            if s.isalpha():
                exists_in_sequence_mask.append(1)
                idxs.append(j)
            else:
                exists_in_sequence_mask.append(0)
                idxs.append(-1)  # Place-holder
        if s in in_seq_dict:
            j += 1

    idxs = np.array(idxs)

    return MSAIndexCorrespondence(
        sequence=matched_sequence,
        res_idx=idxs,
        key_start_match=idxs[idxs != -1][0],
        key_end_match=idxs[idxs != -1][-1],
        exists_in_sequence_mask=np.array(exists_in_sequence_mask, dtype=int),
    )


def fix_flanking_regions(matched_sequence, full_msa, res_idx):
    start_match, end_match = res_idx.min(), res_idx.max()
    matched_sequence_end_idx = len(matched_sequence) - 1

    i = 0
    while True:
        if i >= matched_sequence_end_idx:
            break
        if matched_sequence[i].isalpha():
            break
        i += 1
    start_flank_len = i

    i = matched_sequence_end_idx
    while True:
        if i <= 0:
            break
        if matched_sequence[i].isalpha():
            break
        i -= 1
    end_flank_len = matched_sequence_end_idx - i

    msa_remove_dots = remove_dots(full_msa)
    msa_end_idx = len(msa_remove_dots) - 1

    new_start_flank = []
    i = start_match - 1
    while start_flank_len > 0:
        if i <= 0:
            break
        if msa_remove_dots[i].isalpha():
            new_start_flank.append(msa_remove_dots[i].upper())
            start_flank_len -= 1
        i -= 1
    new_start_flank = "".join(reversed(new_start_flank))

    new_end_flank = []
    i = end_match + 1
    while end_flank_len > 0:
        if i >= msa_end_idx:
            break
        if msa_remove_dots[i].isalpha():
            new_end_flank.append(msa_remove_dots[i].upper())
            end_flank_len -= 1
        i += 1
    new_end_flank = "".join(new_end_flank)

    start_flank_len, end_flank_len = len(new_start_flank), len(new_end_flank)
    sequence_middle = matched_sequence[
        start_flank_len
        if start_flank_len > 0
        else None : -end_flank_len
        if end_flank_len > 0
        else None
    ]
    return new_start_flank + sequence_middle + new_end_flank


def sort_chains(
    match_to_sequence: MatchToSequence,
    chains,
    ca_positions,
    min_chain_len=5,
    min_match_score=0.4,
    max_dist=30,
    max_seq_gap=40,
):
    unique_seqs = np.unique(match_to_sequence.sequence_idxs)

    og_chain_lens = np.array([len(c) for c in chains])
    og_chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
    og_chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    chain_start_pos = ca_positions[chain_starts]
    chain_end_pos = ca_positions[chain_ends]

    new_chain_ids = [[i] for i in range(len(chains))]

    spent_starts, spent_ends = set(), set()

    for seq in unique_seqs:
        sequence_match_idx = np.nonzero(match_to_sequence.sequence_idxs == seq)[0]

        if len(sequence_match_idx) > 1:
            dist_mat = np.linalg.norm(
                chain_start_pos[sequence_match_idx, None]
                - chain_end_pos[sequence_match_idx][None],
                axis=-1,
            )
            np.fill_diagonal(dist_mat, np.inf)
            dist_mat = np.where(dist_mat < max_dist, dist_mat, np.inf)
            dist_mat[
                match_to_sequence.match_scores[sequence_match_idx] < min_match_score
            ] = np.inf
            dist_mat[
                :, match_to_sequence.match_scores[sequence_match_idx] < min_match_score
            ] = np.inf

            seq_match_chain_lens = og_chain_lens[sequence_match_idx]
            dist_mat[seq_match_chain_lens < min_chain_len] = np.inf
            dist_mat[:, seq_match_chain_lens < min_chain_len] = np.inf

            gaps = (
                match_to_sequence.key_start_matches[sequence_match_idx, None]
                - match_to_sequence.key_end_matches[sequence_match_idx][None]
            )

            dist_mat = np.where(gaps < max_seq_gap, dist_mat, np.inf)
            dist_mat = np.where(gaps > 0, dist_mat, np.inf)

        else:
            continue

        while np.any(dist_mat != np.inf):
            chain_start_idx, chain_end_idx = np.unravel_index(
                np.argmin(dist_mat), dist_mat.shape
            )
            chain_start_match, chain_end_match = (
                sequence_match_idx[chain_start_idx],
                sequence_match_idx[chain_end_idx],
            )

            chain_start_match_reidx = np.nonzero(
                chain_starts == og_chain_starts[chain_start_match]
            )[0][0]
            chain_end_match_reidx = np.nonzero(
                chain_ends == og_chain_ends[chain_end_match]
            )[0][0]
            if chain_start_match_reidx == chain_end_match_reidx:
                dist_mat[chain_start_idx, chain_end_idx] = np.inf
                continue

            new_chain = np.concatenate(
                (chains[chain_end_match_reidx], chains[chain_start_match_reidx]), axis=0
            )
            chain_arange = np.arange(len(chains))
            tmp_chains = np.array(chains, dtype=object)[
                chain_arange[
                    (chain_arange != chain_start_match_reidx)
                    & (chain_arange != chain_end_match_reidx)
                ]
            ].tolist()
            tmp_chains.append(new_chain)
            chains = tmp_chains
            new_chain_id = (
                new_chain_ids[chain_end_match_reidx]
                + new_chain_ids[chain_start_match_reidx]
            )
            tmp_chain_ids = np.array(new_chain_ids, dtype=object)[
                chain_arange[
                    (chain_arange != chain_start_match_reidx)
                    & (chain_arange != chain_end_match_reidx)
                ]
            ].tolist()
            tmp_chain_ids.append(new_chain_id)
            new_chain_ids = tmp_chain_ids

            chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
            chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

            spent_starts.add(chain_start_match)
            spent_ends.add(chain_end_match)

            dist_mat[chain_start_idx] = np.inf
            dist_mat[:, chain_end_idx] = np.inf

    match_to_sequence.concatenate_chains(new_chain_ids)
    return chains, match_to_sequence


def sort_chains_by_match(
    chains: List[np.ndarray], best_match_output: MatchToSequence
) -> Tuple[List[np.ndarray], MatchToSequence]:
    match_scores = np.array(best_match_output.match_scores)
    new_idxs = np.argsort(-match_scores)
    new_chains = [chains[i] for i in new_idxs]
    return (
        new_chains,
        MatchToSequence(
            new_sequences=[best_match_output.new_sequences[i] for i in new_idxs],
            residue_idxs=[best_match_output.residue_idxs[i] for i in new_idxs],
            sequence_idxs=[best_match_output.sequence_idxs[i] for i in new_idxs],
            key_start_matches=best_match_output.key_start_matches[new_idxs],
            key_end_matches=best_match_output.key_end_matches[new_idxs],
            match_scores=best_match_output.match_scores[new_idxs],
            hmm_output_match_sequences=[
                best_match_output.hmm_output_match_sequences[i] for i in new_idxs
            ],
            exists_in_sequence_mask=[
                best_match_output.exists_in_sequence_mask[i] for i in new_idxs
            ],
            is_nucleotide=[best_match_output.is_nucleotide[i] for i in new_idxs],
        ),
    )


FixChainsOutput = namedtuple(
    "FixChainsOutput", ["chains", "best_match_output", "unmodelled_sequences",],
)


def fix_chains_pipeline(
    prot_sequences: List[str],
    rna_sequences: List[str],
    dna_sequences: List[str],
    chains: List[int],
    chain_aa_logits: List[np.ndarray],
    ca_pos: np.ndarray,
    chain_prot_mask: List[np.ndarray],
    chain_confidences: List[np.ndarray] = None,
    base_dir: str = "/tmp",
    postprocess=True,
    do_pp=False,
) -> FixChainsOutput:
    """
    What you actually want is to get the smallest sum for the distance as well as the gap
    when you tie sequences together. The obvious thing that comes into mind is dynamic programming,
    which in O(n^2), but with the caveat that it can be made to be O(km^2),
    where k is the number of sequences and m is the max length of the sequence match.

    Oh its A*!!!! Shortest path between two nodes, where you can't go from all to all, just ones that have
    the average sequence being close or something. You find shortest path, with some penalty for how long the
    sequence is or whatever.

    """
    best_match_output = best_match_to_sequences(
        prot_sequences=prot_sequences,
        rna_sequences=rna_sequences,
        dna_sequences=dna_sequences,
        chain_aa_logits=chain_aa_logits,
        chain_prot_mask=chain_prot_mask,
        chain_confidences=chain_confidences,
        base_dir=base_dir,
        do_pp=do_pp,
    )
    if postprocess:
        chains = best_match_output.remove_duplicates(chains, ca_pos)
        chains, best_match_output = sort_chains(best_match_output, chains, ca_pos,)
        chains, best_match_output = sort_chains_by_match(chains, best_match_output)
    return FixChainsOutput(
        chains=chains, best_match_output=best_match_output, unmodelled_sequences=None,
    )


def prune_and_connect_chains(
    chains: List[int],
    best_match_output: MatchToSequence,
    ca_pos: np.ndarray,
    aggressive_pruning=False,
    chain_prune_length=4,
):
    # prune 1 delete most of chains here
    chains = best_match_output.prune_chains(
        chains,
        chain_prune_length=chain_prune_length,
        aggressive_pruning=aggressive_pruning,
    )
    # prune 2 remove overlapped chains, few chains are deleted
    chains = best_match_output.remove_duplicates(chains, ca_pos)
    if aggressive_pruning:
        # prune 3 sort chains, some are deleted here
        chains, best_match_output = sort_chains(best_match_output, chains, ca_pos,)
    # prune 4 sort chains by match, few are deleted
    chains, best_match_output = sort_chains_by_match(chains, best_match_output)
    return FixChainsOutput(
        chains=chains, best_match_output=best_match_output, unmodelled_sequences=None,
    )

