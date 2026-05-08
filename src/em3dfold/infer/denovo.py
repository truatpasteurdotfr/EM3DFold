import os
import json
import copy

import numpy as np
from scipy.spatial import cKDTree

from em3dfold.infer.fixing import fix_match
from em3dfold.infer.flood_fill import (
    beam_trace_with_edge_dual_protein,
    flood_fill_with_edge_dual,
)
from em3dfold.infer.gnn_inference_utils import sample_na_aa_logits
from em3dfold.infer.hmm.hmm_sequence_align import (
    MatchToSequence,
    best_match_to_sequences,
    fix_chains_pipeline,
    prune_and_connect_chains,
)
from em3dfold.io.pdbio import chains_atom_pos_to_pdb
from em3dfold.io.seqio import read_fasta
from em3dfold.polymer_utils.residue_constants import num_prot, select_torsion_angles
from em3dfold.utils.misc_utils import abspath, pjoin
from em3dfold.utils.to_all_atom import affines_and_torsion_angles_to_atomc_pos


FALLBACK_CONFIDENCE_SENTINEL = 0.0


def remove_overlapping_ca(
    ca_positions: np.ndarray,
    bfactors,
    existence_mask=None,
    radius_threshold: float = 1.5,
) -> np.ndarray:
    kdtree = cKDTree(ca_positions)
    bfactors_copy = np.copy(bfactors)
    sorted_indices = np.argsort(bfactors_copy)[::-1]
    if existence_mask is None:
        existence_mask = np.ones(len(ca_positions), dtype=bool)

    for i in sorted_indices:
        if existence_mask[i]:
            too_close = np.array(
                kdtree.query_ball_point(ca_positions[i], r=radius_threshold)
            )
            too_close = too_close[too_close != i]
            existence_mask[too_close] = False
    return existence_mask


def normalize_local_confidence_score(
    local_confidence_score: np.ndarray,
    best_value: float = 0.20,
    worst_value: float = 1.00,
) -> np.ndarray:
    normalized_score = (worst_value - local_confidence_score) / (
        worst_value - best_value
    )
    normalized_score = np.clip(normalized_score, 0, 1)
    return normalized_score


def _inverse_minmax_confidence(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-8:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return 1.0 - (values - min_value) / (max_value - min_value)


def _pred_rmsd_to_confidence(pred_rmsd: np.ndarray, prot_mask: np.ndarray) -> np.ndarray:
    pred_rmsd = np.asarray(pred_rmsd, dtype=np.float32)
    prot_mask = np.asarray(prot_mask, dtype=bool)
    confidence = np.zeros((len(pred_rmsd),), dtype=np.float32)

    if np.any(prot_mask):
        prot_conf = normalize_local_confidence_score(
            pred_rmsd[prot_mask],
            best_value=0.20,
            worst_value=1.20,
        )
        if float(np.max(prot_conf)) <= 1e-6 or float(np.std(prot_conf)) <= 1e-6:
            prot_conf = _inverse_minmax_confidence(pred_rmsd[prot_mask])
        confidence[prot_mask] = prot_conf

    na_mask = ~prot_mask
    if np.any(na_mask):
        na_conf = normalize_local_confidence_score(
            pred_rmsd[na_mask],
            best_value=0.40,
            worst_value=1.40,
        )
        if float(np.max(na_conf)) <= 1e-6 or float(np.std(na_conf)) <= 1e-6:
            na_conf = _inverse_minmax_confidence(pred_rmsd[na_mask])
        confidence[na_mask] = na_conf

    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _read_sequences(sequence_path, label):
    if sequence_path is None:
        return []
    print(f"# Read {label} sequence(s)")
    seqs = [seq.upper() for seq in read_fasta(sequence_path)]
    print(f"# Found {len(seqs)} {label} sequences")
    lengths = [len(seq) for seq in seqs]
    if len(lengths) > 0:
        print(f"# {label} sequence lengths = {lengths}")
    return seqs


def _sanitize_na_sequence(seq):
    seq = seq.upper().replace("T", "U")
    seq = "".join([s if s in ["A", "C", "G", "U"] else "N" for s in seq])
    return seq


def _reconstruct_chains(reordered_final_results, chains, chains_res_type, bfactor_key="plddt"):
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_idx = []
    chains_bfactor = []
    for chain, aatype in zip(chains, chains_res_type):
        pred_torsions = select_torsion_angles(
            reordered_final_results["pred_torsions"][chain],
            aatype=aatype,
            normalize=True,
        )
        pred_atom_pos, pred_atom_mask = affines_and_torsion_angles_to_atomc_pos(
            reordered_final_results["pred_affines"][chain],
            pred_torsions,
            aatype=aatype,
        )
        chains_atom_pos.append(pred_atom_pos.cpu().numpy())
        chains_atom_mask.append(pred_atom_mask.cpu().numpy())
        chains_res_idx.append(np.arange(len(chain), dtype=np.int32))
        chains_bfactor.append(reordered_final_results[bfactor_key][chain])
    return chains_atom_pos, chains_atom_mask, chains_res_idx, chains_bfactor


def _get_chain_stats(chains):
    if len(chains) == 0:
        return {
            "num_chains": 0,
            "total_residues": 0,
            "max_len": 0,
            "min_len": 0,
            "mean_len": 0.0,
        }
    lengths = np.array([len(chain) for chain in chains], dtype=np.int32)
    return {
        "num_chains": int(len(chains)),
        "total_residues": int(lengths.sum()),
        "max_len": int(lengths.max()),
        "min_len": int(lengths.min()),
        "mean_len": float(lengths.mean()),
    }


def _print_chain_stats(label, chains):
    stats = _get_chain_stats(chains)
    print(
        "# {}: chains={}, total_residues={}, longest={}, shortest={}, mean={:.2f}".format(
            label,
            stats["num_chains"],
            stats["total_residues"],
            stats["max_len"],
            stats["min_len"],
            stats["mean_len"],
        )
    )


def _write_chain_file(
    filename,
    reordered_final_results,
    chains,
    chains_res_type,
    bfactor_key="plddt",
):
    chains_atom_pos, chains_atom_mask, chains_res_idx, chains_bfactor = _reconstruct_chains(
        reordered_final_results,
        chains,
        chains_res_type,
        bfactor_key=bfactor_key,
    )
    chains_atom_pos_to_pdb(
        filename=filename,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        chains_bfactor=chains_bfactor,
        suffix="cif",
    )
    print(f"# Output chains to {filename}")


def _softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(np.sum(probs, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
    return probs


def _logsumexp_pair_numpy(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    m = np.maximum(x, y)
    return m + np.log(np.exp(x - m) + np.exp(y - m))


def _merge_na_logits_to_acgu(na_logits):
    na_logits = np.asarray(na_logits, dtype=np.float32)
    if na_logits.shape[-1] >= 8:
        return np.stack(
            [
                _logsumexp_pair_numpy(na_logits[..., 0], na_logits[..., 4]),
                _logsumexp_pair_numpy(na_logits[..., 1], na_logits[..., 5]),
                _logsumexp_pair_numpy(na_logits[..., 2], na_logits[..., 6]),
                _logsumexp_pair_numpy(na_logits[..., 3], na_logits[..., 7]),
            ],
            axis=-1,
        )
    if na_logits.shape[-1] >= 4:
        return na_logits[..., :4]
    return na_logits


def _compute_residue_type_scores(pred_aatype, prot_mask):
    pred_aatype = np.asarray(pred_aatype, dtype=np.float32)
    prot_mask = np.asarray(prot_mask, dtype=bool)

    entropy = np.zeros((len(pred_aatype),), dtype=np.float32)
    confidence = np.zeros((len(pred_aatype),), dtype=np.float32)

    if np.any(prot_mask):
        prot_probs = _softmax_numpy(pred_aatype[prot_mask, :num_prot])
        prot_entropy = -np.sum(prot_probs * np.log(np.clip(prot_probs, 1e-8, None)), axis=-1)
        prot_max_entropy = np.log(float(num_prot))
        entropy[prot_mask] = prot_entropy
        confidence[prot_mask] = 1.0 - prot_entropy / prot_max_entropy

    na_mask = ~prot_mask
    if np.any(na_mask):
        na_logits = _merge_na_logits_to_acgu(pred_aatype[na_mask, num_prot:])
        na_vocab = na_logits.shape[-1]
        if na_vocab <= 1:
            na_entropy = np.zeros((len(na_logits),), dtype=np.float32)
            na_conf = np.ones((len(na_logits),), dtype=np.float32)
        else:
            na_probs = _softmax_numpy(na_logits)
            na_entropy = -np.sum(na_probs * np.log(np.clip(na_probs, 1e-8, None)), axis=-1)
            na_max_entropy = np.log(float(na_vocab))
            na_conf = 1.0 - na_entropy / na_max_entropy
        entropy[na_mask] = na_entropy
        confidence[na_mask] = na_conf

    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    return entropy.astype(np.float32), confidence


def _expand_residue_scalar_to_atom_bfactor(scores, num_atoms=23):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
    return np.repeat(scores, num_atoms, axis=-1)


def _write_residue_type_score_npz(filename, final_results):
    np.savez_compressed(
        filename,
        residue_type_entropy=np.asarray(final_results["residue_type_entropy"], dtype=np.float32),
        residue_type_confidence=np.asarray(final_results["residue_type_confidence"], dtype=np.float32),
        pred_rmsd=np.asarray(final_results["pred_rmsd"], dtype=np.float32),
        prot_mask=np.asarray(final_results["prot_mask"], dtype=bool),
        pred_aatype=np.asarray(final_results["pred_aatype"], dtype=np.float32),
    )
    print(f"# Output residue type scores to {filename}")


def _build_chain_order_recycle_state(reordered_final_results, chains):
    if len(chains) == 0:
        return None

    flat_indices = np.concatenate(chains, axis=0)
    pred_aatype = np.asarray(
        reordered_final_results["pred_aatype"][flat_indices],
        dtype=np.float32,
    )
    prot_mask = np.asarray(reordered_final_results["prot_mask"][flat_indices], dtype=bool)

    prev_aa_probs = np.zeros_like(pred_aatype, dtype=np.float32)
    if np.any(prot_mask):
        prev_aa_probs[prot_mask, :num_prot] = _softmax_numpy(
            pred_aatype[prot_mask, :num_prot]
        )
    if np.any(~prot_mask):
        prev_aa_probs[~prot_mask, num_prot:] = _softmax_numpy(
            pred_aatype[~prot_mask, num_prot:]
        )

    prev_rmsd = np.asarray(
        reordered_final_results["pred_rmsd"][flat_indices],
        dtype=np.float32,
    ).reshape(-1, 1)

    recycle_state = {
        "prev_aa_probs": prev_aa_probs,
        "prev_rmsd": prev_rmsd,
    }
    if "recycle_node_state" in reordered_final_results:
        recycle_state["prev_node"] = np.asarray(
            reordered_final_results["recycle_node_state"][flat_indices],
            dtype=np.float32,
        )

    return recycle_state


def _write_residue_subset_file(
    filename,
    atom_pos,
    atom_mask,
    aatype,
    residue_index=None,
    bfactor=None,
):
    if len(atom_pos) == 0:
        return

    if residue_index is None:
        residue_index = np.arange(len(atom_pos), dtype=np.int32)
    if bfactor is None:
        bfactor = np.zeros_like(atom_mask, dtype=np.float32)

    chains_atom_pos_to_pdb(
        filename=filename,
        chains_atom_pos=[atom_pos],
        chains_atom_mask=[atom_mask],
        chains_res_type=[aatype],
        chains_res_idx=[residue_index],
        chains_bfactor=[bfactor],
        suffix="cif",
    )
    print(f"# Output raw residue subset to {filename}")


def _build_protein_chain_outputs(
    reordered_final_results,
    prot_chains,
    prot_seqs,
    hmm_temp_dir,
    dummy_atom_pos,
):
    if len(prot_chains) == 0:
        return [], [], [], []

    if len(prot_seqs) == 0:
        chains_res_type = [
            np.argmax(reordered_final_results["pred_aatype"][chain][..., :num_prot], axis=-1)
            for chain in prot_chains
        ]
        return prot_chains, chains_res_type, prot_chains, chains_res_type

    # Keep CA positions in the same global index space as `prot_chains`.
    ca_pos = dummy_atom_pos[..., 1, :]
    chains_prot_mask = [np.ones(len(chain), dtype=bool) for chain in prot_chains]
    chains_aa_logits = [reordered_final_results["pred_aatype"][chain] for chain in prot_chains]

    fix_chains_output = fix_chains_pipeline(
        prot_sequences=prot_seqs,
        rna_sequences=[],
        dna_sequences=[],
        chains=prot_chains,
        chain_aa_logits=chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=chains_prot_mask,
        base_dir=hmm_temp_dir,
        postprocess=False,
    )

    print("# Fix protein chain match")
    fix_chains_output = fix_chains_output._replace(
        best_match_output=fix_match(
            fix_chains_output.best_match_output,
            prot_seqs,
            match_score_cutoff=0.60,
            len_cutoff=10,
        )
    )

    before_chains = fix_chains_output.chains
    before_chain_types = fix_chains_output.best_match_output.new_sequences

    after_fix_chains_output = prune_and_connect_chains(
        chains=fix_chains_output.chains,
        best_match_output=fix_chains_output.best_match_output,
        ca_pos=ca_pos,
        aggressive_pruning=True,
        chain_prune_length=4,
    )

    return (
        before_chains,
        before_chain_types,
        after_fix_chains_output.chains,
        after_fix_chains_output.best_match_output.new_sequences,
    )


def _map_na_sequence_to_res_type(sequence, is_dna):
    if is_dna:
        mapping = {"A": 20, "C": 21, "G": 22, "U": 23, "N": 29}
    else:
        mapping = {"A": 24, "C": 25, "G": 26, "U": 27, "N": 30}
    return np.array([mapping.get(ch, mapping["N"]) for ch in sequence], dtype=np.int32)


def _filter_short_chains(chains, chains_res_type, min_len):
    if min_len <= 1:
        return chains, chains_res_type
    kept = [
        (chain, res_type)
        for chain, res_type in zip(chains, chains_res_type)
        if len(chain) >= min_len
    ]
    if len(kept) == 0:
        return [], []
    new_chains, new_chain_types = zip(*kept)
    return list(new_chains), list(new_chain_types)


def _filter_short_chains_with_fallback_masks(chains, chains_res_type, fallback_masks, min_len):
    if min_len <= 1:
        return chains, chains_res_type, fallback_masks
    kept = [
        (chain, res_type, fallback_mask)
        for chain, res_type, fallback_mask in zip(chains, chains_res_type, fallback_masks)
        if len(chain) >= min_len
    ]
    if len(kept) == 0:
        return [], [], []
    new_chains, new_chain_types, new_fallback_masks = zip(*kept)
    return list(new_chains), list(new_chain_types), list(new_fallback_masks)


def _predict_na_chain_types(reordered_final_results, na_chains):
    return [
        np.argmax(reordered_final_results["pred_aatype"][chain][..., num_prot:], axis=-1) + num_prot
        for chain in na_chains
    ]


def _pack_shared_na_logits(na_logits):
    na_logits = np.asarray(na_logits, dtype=np.float32)
    full_logits = np.full(
        na_logits.shape[:-1] + (num_prot + 4,),
        fill_value=-100.0,
        dtype=np.float32,
    )
    full_logits[..., num_prot:] = na_logits
    return full_logits


def _sample_vanilla_na_chain_logits(reordered_final_results, na_chains, na_aa_logits_data):
    if na_aa_logits_data is None:
        return None

    vanilla_chain_logits = []
    for chain in na_chains:
        c4_positions = reordered_final_results["_dummy_atom_pos"][chain][..., 1, :]
        chain_logits = sample_na_aa_logits(
            na_aa_logits_data["map"],
            c4_positions,
            na_aa_logits_data["origin"],
            na_aa_logits_data["voxel_size"],
        )
        vanilla_chain_logits.append(_pack_shared_na_logits(chain_logits))
    return vanilla_chain_logits


def _select_best_na_match_output(tta_output, vanilla_output, chains):
    if vanilla_output is None:
        return tta_output, None

    if len(tta_output.chains) != len(vanilla_output.chains):
        print("# WARN NA TTA/vanilla chain counts differ, keep TTA logits alignment")
        return tta_output, None

    use_tta = []
    selected = {
        "new_sequences": [],
        "residue_idxs": [],
        "sequence_idxs": [],
        "key_start_matches": [],
        "key_end_matches": [],
        "match_scores": [],
        "hmm_output_match_sequences": [],
        "exists_in_sequence_mask": [],
        "is_nucleotide": [],
    }

    tta_match = tta_output.best_match_output
    vanilla_match = vanilla_output.best_match_output
    tta_score_sum = float(np.sum(tta_match.match_scores))
    vanilla_score_sum = float(np.sum(vanilla_match.match_scores))
    print("# NA TTA score = {:.4f}".format(tta_score_sum))
    print("# NA vanilla score = {:.4f}".format(vanilla_score_sum))

    tta_residue_count = 0
    vanilla_residue_count = 0
    total_residue_count = 0
    for chain_idx in range(len(chains)):
        tta_score = tta_match.match_scores[chain_idx]
        vanilla_score = vanilla_match.match_scores[chain_idx]
        choose_tta = tta_score > vanilla_score
        use_tta.append(choose_tta)

        chain_len = len(chains[chain_idx])
        total_residue_count += chain_len
        source = tta_match if choose_tta else vanilla_match
        if choose_tta:
            tta_residue_count += chain_len
        else:
            vanilla_residue_count += chain_len

        selected["new_sequences"].append(source.new_sequences[chain_idx])
        selected["residue_idxs"].append(source.residue_idxs[chain_idx])
        selected["sequence_idxs"].append(source.sequence_idxs[chain_idx])
        selected["key_start_matches"].append(source.key_start_matches[chain_idx])
        selected["key_end_matches"].append(source.key_end_matches[chain_idx])
        selected["match_scores"].append(source.match_scores[chain_idx])
        selected["hmm_output_match_sequences"].append(
            source.hmm_output_match_sequences[chain_idx]
        )
        selected["exists_in_sequence_mask"].append(
            source.exists_in_sequence_mask[chain_idx]
        )
        selected["is_nucleotide"].append(source.is_nucleotide[chain_idx])

    if total_residue_count > 0:
        print(
            "# Using TTA ratio = {:.4f} using vanilla ratio = {:.4f}".format(
                tta_residue_count / total_residue_count,
                vanilla_residue_count / total_residue_count,
            )
        )

    merged_output = MatchToSequence(
        new_sequences=selected["new_sequences"],
        residue_idxs=selected["residue_idxs"],
        sequence_idxs=np.array(selected["sequence_idxs"]),
        key_start_matches=np.array(selected["key_start_matches"]),
        key_end_matches=np.array(selected["key_end_matches"]),
        match_scores=np.array(selected["match_scores"]),
        hmm_output_match_sequences=selected["hmm_output_match_sequences"],
        exists_in_sequence_mask=selected["exists_in_sequence_mask"],
        is_nucleotide=selected["is_nucleotide"],
    )
    return (
        tta_output._replace(best_match_output=merged_output),
        np.asarray(use_tta, dtype=bool),
    )


def _build_na_chain_outputs(
    reordered_final_results,
    na_chains,
    rna_seqs,
    dna_seqs,
    hmm_temp_dir,
    na_aa_logits_data=None,
    min_chain_len=1,
    fallback_to_predicted_na_types=True,
):
    if len(na_chains) == 0:
        return [], [], [], [], [], []

    if len(rna_seqs) + len(dna_seqs) == 0:
        before_chains = list(na_chains)
        before_chain_types = _predict_na_chain_types(reordered_final_results, before_chains)
        before_fallback_masks = [
            np.ones((len(chain),), dtype=bool) for chain in before_chains
        ]
        print(
            "# NA sequences not provided; keep predicted NA types and apply length-only filtering (min_len = {})".format(
                min_chain_len
            )
        )
        after_chains, after_chain_types, after_fallback_masks = _filter_short_chains_with_fallback_masks(
            before_chains,
            before_chain_types,
            before_fallback_masks,
            min_chain_len,
        )
        return (
            before_chains,
            before_chain_types,
            after_chains,
            after_chain_types,
            before_fallback_masks,
            after_fallback_masks,
        )

    seqs = [_sanitize_na_sequence(seq) for seq in (rna_seqs + dna_seqs)]
    seqs_is_dna = [False] * len(rna_seqs) + [True] * len(dna_seqs)
    first_character = seqs[0][0]
    seqs_is_all_same = np.all(
        np.asarray([all(ch == first_character for ch in seq) for seq in seqs], dtype=bool)
    )
    print(f"# NA seqs is all same = {seqs_is_all_same}")

    # Keep C1'/CA proxy positions in the same global index space as `na_chains`.
    ca_pos = reordered_final_results["_dummy_atom_pos"][..., 1, :]
    chains_prot_mask = [np.zeros(len(chain), dtype=bool) for chain in na_chains]
    chains_aa_logits = [reordered_final_results["pred_aatype"][chain] for chain in na_chains]

    fix_chains_output = fix_chains_pipeline(
        prot_sequences=[],
        rna_sequences=seqs,
        dna_sequences=[],
        chains=na_chains,
        chain_aa_logits=chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=chains_prot_mask,
        chain_confidences=None,
        base_dir=hmm_temp_dir,
        postprocess=False,
    )
    vanilla_chain_logits = _sample_vanilla_na_chain_logits(
        reordered_final_results,
        fix_chains_output.chains,
        na_aa_logits_data,
    )
    if vanilla_chain_logits is not None:
        vanilla_fix_chains_output = fix_chains_pipeline(
            prot_sequences=[],
            rna_sequences=seqs,
            dna_sequences=[],
            chains=na_chains,
            chain_aa_logits=vanilla_chain_logits,
            ca_pos=ca_pos,
            chain_prot_mask=chains_prot_mask,
            chain_confidences=None,
            base_dir=hmm_temp_dir,
            postprocess=False,
        )
        fix_chains_output, _ = _select_best_na_match_output(
            fix_chains_output,
            vanilla_fix_chains_output,
            fix_chains_output.chains,
        )
    predicted_chain_types = _predict_na_chain_types(
        reordered_final_results,
        fix_chains_output.chains,
    )

    print("# Fix NA chain match")
    fixed_match = fix_match(
        fix_chains_output.best_match_output,
        [seq.lower() for seq in seqs],
        match_score_cutoff=0.40,
        len_cutoff=6,
    )

    chains_res_type = []
    chain_fallback_masks = []
    num_fallback_chains = 0
    for chain_idx, aatype in enumerate(fixed_match.new_sequences):
        aatype = aatype.copy()
        chain = fix_chains_output.chains[chain_idx]
        seq_idx = fixed_match.sequence_idxs[chain_idx]
        has_valid_sequence = 0 <= seq_idx < len(seqs)
        if not has_valid_sequence:
            if fallback_to_predicted_na_types:
                chains_res_type.append(predicted_chain_types[chain_idx].astype(np.int32))
                chain_fallback_masks.append(np.ones((len(chain),), dtype=bool))
                num_fallback_chains += 1
            else:
                chains_res_type.append(aatype.astype(np.int32))
                chain_fallback_masks.append(np.zeros((len(chain),), dtype=bool))
            continue
        if seqs_is_dna[seq_idx]:
            aatype[aatype == 24] = 20
            aatype[aatype == 25] = 21
            aatype[aatype == 26] = 22
            aatype[aatype == 27] = 23
            aatype[aatype >= 28] = 21
        else:
            aatype[aatype >= 28] = 21 + 4
        chains_res_type.append(aatype.astype(np.int32))
        chain_fallback_masks.append(np.zeros((len(chain),), dtype=bool))

    if seqs_is_all_same:
        for chain_idx, chain_types in enumerate(chains_res_type):
            seq_idx = fixed_match.sequence_idxs[chain_idx]
            if not (0 <= seq_idx < len(seqs)):
                continue
            fill_value = _map_na_sequence_to_res_type(seqs[seq_idx], seqs_is_dna[seq_idx])[0]
            chain_types[:] = fill_value

    if fallback_to_predicted_na_types and num_fallback_chains > 0:
        print(
            "# Fallback to predicted NA types for {} unmatched chain(s)".format(
                num_fallback_chains
            )
        )

    before_chains = fix_chains_output.chains
    before_chain_types = chains_res_type
    before_fallback_masks = chain_fallback_masks
    print(
        "# Use HMM-aligned NA chain types, then apply length-only filtering (min_len = {})".format(
            min_chain_len
        )
    )
    after_chains, after_chain_types, after_fallback_masks = _filter_short_chains_with_fallback_masks(
        before_chains,
        before_chain_types,
        before_fallback_masks,
        min_chain_len,
    )
    return (
        before_chains,
        before_chain_types,
        after_chains,
        after_chain_types,
        before_fallback_masks,
        after_fallback_masks,
    )


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _histogram_summary(values, bins=10):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"bin_edges": [], "counts": []}
    bins = int(max(1, min(bins, max(1, values.size))))
    counts, bin_edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def _serializable_candidate(candidate):
    serializable = dict(candidate)
    if "overlap_metadata" in serializable:
        overlap_metadata = dict(serializable["overlap_metadata"])
        overlap_metadata["conflict_candidate_ids"] = [
            int(x) if isinstance(x, (np.integer, int)) else x
            for x in overlap_metadata.get("conflict_candidate_ids", [])
        ]
        overlap_metadata["dedup_removed_candidate_ids"] = [
            int(x) if isinstance(x, (np.integer, int)) else x
            for x in overlap_metadata.get("dedup_removed_candidate_ids", [])
        ]
        serializable["overlap_metadata"] = overlap_metadata
    return serializable


def _decorate_protein_beam_candidate(
    candidate,
    round_idx,
    protein_local_indices,
    protein_global_indices,
    protein_original_indices,
):
    local_path = np.asarray(candidate["node_indices"], dtype=np.int32)
    return {
        "candidate_uid": f"round{int(round_idx)}_cand{int(candidate['candidate_id'])}",
        "source_round_idx": int(round_idx),
        "source_candidate_id": int(candidate["candidate_id"]),
        "node_indices_local": [int(x) for x in local_path.tolist()],
        "node_indices_filtered": [
            int(x) for x in protein_global_indices[local_path].astype(np.int32).tolist()
        ],
        "node_indices_original": [
            int(x) for x in protein_original_indices[local_path].astype(np.int32).tolist()
        ],
        "path_length": int(candidate["path_length"]),
        "tail_idx_local": int(local_path[-1]),
        "tail_idx_filtered": int(protein_global_indices[local_path[-1]]),
        "tail_idx_original": int(protein_original_indices[local_path[-1]]),
        "score_total": float(candidate["score_total"]),
        "score_breakdown": {
            "edge": float(candidate["score_breakdown"]["edge"]),
            "cn": float(candidate["score_breakdown"]["cn"]),
            "ca": float(candidate["score_breakdown"]["ca"]),
            "confidence": float(candidate["score_breakdown"]["confidence"]),
        },
        "edge_sources": [str(x) for x in candidate.get("edge_sources", [])],
        "edge_scores": candidate.get("edge_scores", []),
        "num_ca_rescue_edges": int(candidate.get("num_ca_rescue_edges", 0)),
        "num_both_edges": int(candidate.get("num_both_edges", 0)),
        "overlap_metadata": {
            "duplicate_of": None,
            "dedup_removed_candidate_ids": [],
            "conflict_candidate_ids": [],
        },
    }


def _beam_candidate_overlap(a_nodes, b_nodes):
    a_set = set(int(x) for x in a_nodes)
    b_set = set(int(x) for x in b_nodes)
    if len(a_set) == 0 or len(b_set) == 0:
        return 0.0
    return len(a_set & b_set) / float(min(len(a_set), len(b_set)))


def _deduplicate_protein_beam_candidates(candidates, overlap_threshold=0.95):
    ranked = sorted(
        candidates,
        key=lambda cand: (cand["score_total"], cand["path_length"]),
        reverse=True,
    )
    kept = []
    duplicate_map = {}

    for candidate in ranked:
        candidate_nodes = candidate["node_indices_filtered"]
        candidate_set = set(candidate_nodes)
        duplicate_of = None

        for kept_candidate in kept:
            kept_nodes = kept_candidate["node_indices_filtered"]
            kept_set = set(kept_nodes)
            if candidate_nodes == kept_nodes:
                duplicate_of = kept_candidate["candidate_uid"]
                break

            overlap = _beam_candidate_overlap(candidate_nodes, kept_nodes)
            if overlap >= overlap_threshold:
                if candidate_nodes[0] == kept_nodes[0] and candidate_nodes[-1] == kept_nodes[-1]:
                    duplicate_of = kept_candidate["candidate_uid"]
                    break

            if candidate_set.issubset(kept_set) and len(candidate_nodes) < len(kept_nodes):
                if candidate["score_total"] <= kept_candidate["score_total"] + 1e-6:
                    duplicate_of = kept_candidate["candidate_uid"]
                    break

        if duplicate_of is not None:
            candidate["overlap_metadata"]["duplicate_of"] = duplicate_of
            duplicate_map.setdefault(duplicate_of, []).append(candidate["candidate_uid"])
            continue

        kept.append(candidate)

    for candidate in kept:
        candidate["overlap_metadata"]["dedup_removed_candidate_ids"] = duplicate_map.get(
            candidate["candidate_uid"], []
        )

    return kept, duplicate_map


def _attach_protein_beam_conflicts(candidates):
    node_to_candidate_ids = {}
    for candidate in candidates:
        candidate_nodes = set(int(x) for x in candidate["node_indices_filtered"])
        for node_idx in candidate_nodes:
            node_to_candidate_ids.setdefault(node_idx, []).append(candidate["candidate_uid"])

    conflict_map = {candidate["candidate_uid"]: set() for candidate in candidates}
    for candidate_ids in node_to_candidate_ids.values():
        if len(candidate_ids) < 2:
            continue
        for candidate_id in candidate_ids:
            conflict_map[candidate_id].update(
                other_id for other_id in candidate_ids if other_id != candidate_id
            )

    for candidate in candidates:
        candidate["overlap_metadata"]["conflict_candidate_ids"] = sorted(
            conflict_map[candidate["candidate_uid"]]
        )

    return candidates, conflict_map


def _summarize_protein_beam_candidates(candidates):
    if len(candidates) == 0:
        return {
            "raw_candidate_count": 0,
            "candidate_count": 0,
            "path_length_histogram": {"bin_edges": [], "counts": []},
            "score_histogram": {"bin_edges": [], "counts": []},
            "num_candidates_with_ca_rescue": 0,
            "fraction_candidates_with_ca_rescue": 0.0,
            "fraction_edges_from_ca_rescue": 0.0,
        }

    lengths = np.asarray([cand["path_length"] for cand in candidates], dtype=np.float32)
    scores = np.asarray([cand["score_total"] for cand in candidates], dtype=np.float32)
    rescue_candidates = [cand for cand in candidates if cand["num_ca_rescue_edges"] > 0]
    rescue_edges = sum(int(cand["num_ca_rescue_edges"]) for cand in candidates)
    total_edges = sum(max(int(cand["path_length"]) - 1, 0) for cand in candidates)

    return {
        "raw_candidate_count": int(len(candidates)),
        "candidate_count": int(len(candidates)),
        "path_length_histogram": _histogram_summary(lengths),
        "score_histogram": _histogram_summary(scores),
        "num_candidates_with_ca_rescue": int(len(rescue_candidates)),
        "fraction_candidates_with_ca_rescue": float(len(rescue_candidates) / len(candidates)),
        "fraction_edges_from_ca_rescue": float(rescue_edges / max(total_edges, 1)),
    }


def _annotate_protein_beam_candidates_with_hmm(
    candidates,
    reordered_final_results,
    prot_seqs,
    hmm_temp_dir,
):
    if len(candidates) == 0 or len(prot_seqs) == 0:
        return candidates, None

    chain_aa_logits = []
    chain_confidences = []
    chain_prot_mask = []
    for candidate in candidates:
        node_indices = np.asarray(candidate["node_indices_filtered"], dtype=np.int32)
        chain_aa_logits.append(
            np.asarray(reordered_final_results["pred_aatype"][node_indices], dtype=np.float32)
        )
        if "confidence" in reordered_final_results:
            chain_confidences.append(
                np.asarray(reordered_final_results["confidence"][node_indices][:, 0], dtype=np.float32)
            )
        else:
            chain_confidences.append(None)
        chain_prot_mask.append(np.ones((len(node_indices),), dtype=bool))

    hmm_output = best_match_to_sequences(
        prot_sequences=prot_seqs,
        rna_sequences=[],
        dna_sequences=[],
        chain_prot_mask=chain_prot_mask,
        chain_aa_logits=chain_aa_logits,
        chain_confidences=chain_confidences,
        base_dir=hmm_temp_dir,
        do_pp=False,
    )

    for candidate_idx, candidate in enumerate(candidates):
        seq_idx = int(hmm_output.sequence_idxs[candidate_idx])
        residue_idxs = np.asarray(hmm_output.residue_idxs[candidate_idx], dtype=np.int32)
        exists_mask = np.asarray(
            hmm_output.exists_in_sequence_mask[candidate_idx], dtype=np.int32
        )
        coverage = float(np.mean(exists_mask.astype(np.float32))) if len(exists_mask) > 0 else 0.0
        candidate["hmm_annotation"] = {
            "match_score": float(hmm_output.match_scores[candidate_idx]),
            "matched_sequence_id": int(seq_idx) if 0 <= seq_idx < len(prot_seqs) else -1,
            "matched_residue_span": [
                int(residue_idxs.min()) if len(residue_idxs) > 0 else -1,
                int(residue_idxs.max()) if len(residue_idxs) > 0 else -1,
            ],
            "aligned_coverage": coverage,
            "exists_in_sequence_mask_summary": {
                "count": int(np.sum(exists_mask > 0)),
                "length": int(len(exists_mask)),
            },
            "hmm_output_match_sequence": hmm_output.hmm_output_match_sequences[candidate_idx],
        }

    return candidates, hmm_output


def _rank_protein_beam_candidates(candidates):
    if len(candidates) == 0:
        return candidates

    def _sort_key(candidate):
        hmm_annotation = candidate.get("hmm_annotation")
        if hmm_annotation is None:
            return (candidate["score_total"], candidate["path_length"])
        return (
            float(hmm_annotation.get("match_score", 0.0)),
            float(hmm_annotation.get("aligned_coverage", 0.0)),
            float(candidate["score_total"]),
            int(candidate["path_length"]),
        )

    return sorted(candidates, key=_sort_key, reverse=True)


def _select_anchor_candidates_for_iterative_refine(candidates, max_anchors=32):
    if len(candidates) == 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.get("hmm_annotation", {}).get("match_score", candidate["score_total"]),
            candidate.get("hmm_annotation", {}).get("aligned_coverage", 0.0),
            candidate["score_total"],
            candidate["path_length"],
        ),
        reverse=True,
    )

    selected = []
    used_nodes = set()
    for candidate in ranked:
        candidate_nodes = set(int(x) for x in candidate["node_indices_local"])
        if len(candidate_nodes) == 0:
            continue
        if candidate_nodes & used_nodes:
            continue
        hmm_annotation = candidate.get("hmm_annotation")
        if hmm_annotation:
            if hmm_annotation.get("match_score", 0.0) < 0.55:
                continue
            if hmm_annotation.get("aligned_coverage", 0.0) < 0.4:
                continue
        elif candidate["score_total"] < 0.0:
            continue
        selected.append(candidate)
        used_nodes.update(candidate_nodes)
        if len(selected) >= max_anchors:
            break

    return selected


def _write_json(filename, payload):
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(f"# Output JSON to {filename}")


def final_results_align_to_sequence_beam_protein(
    final_results,
    protein_seq_dir=None,
    dna_seq_dir=None,
    rna_seq_dir=None,
    output_dir="output",
    flag_prune_and_connect_chains=True,
    protein_radius_threshold=1.5,
    na_radius_threshold=1.5,
    min_na_chain_len=1,
    fallback_to_predicted_na_types=True,
    na_aa_logits_data=None,
    beam_width_protein=16,
    trace_max_candidates=5000,
    cn_radius_protein=2.1,
    ca_rescue_radius_protein=4.8,
    beam_protein_trace_dump_candidates=True,
    beam_protein_trace_hmm_rerank=False,
    beam_protein_trace_iterative_refine=False,
    beam_protein_trace_max_refine_rounds=3,
):
    del dna_seq_dir
    del rna_seq_dir
    del flag_prune_and_connect_chains
    del na_radius_threshold
    del min_na_chain_len
    del fallback_to_predicted_na_types
    del na_aa_logits_data

    parent_out_dir = abspath(output_dir)
    out_dir = pjoin(parent_out_dir, "postprocess_beam_protein")
    hmm_temp_dir = pjoin(out_dir, "hmm")
    os.makedirs(hmm_temp_dir, exist_ok=True)
    print(f"# Beam protein postprocess directory = {out_dir}")
    print(f"# Beam protein HMM temp dir = {hmm_temp_dir}")

    prot_seqs = (
        _read_sequences(protein_seq_dir, "protein")
        if protein_seq_dir is not None
        else []
    )

    working_results = dict(final_results)
    prot_mask_full = np.asarray(final_results["prot_mask"], dtype=bool)
    if not np.any(prot_mask_full):
        summary_path = pjoin(out_dir, "beam_summary.json")
        candidates_path = pjoin(out_dir, "protein_candidates.json")
        empty_summary = {
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
            "truncated_candidate_count": 0,
            "beam_settings": {
                "beam_width_protein": int(beam_width_protein),
                "trace_max_candidates": int(trace_max_candidates),
                "cn_radius_protein": float(cn_radius_protein),
                "ca_rescue_radius_protein": float(ca_rescue_radius_protein),
                "beam_protein_trace_hmm_rerank": bool(beam_protein_trace_hmm_rerank),
                "beam_protein_trace_iterative_refine": bool(
                    beam_protein_trace_iterative_refine
                ),
                "beam_protein_trace_max_refine_rounds": int(
                    beam_protein_trace_max_refine_rounds
                ),
            },
            "round_summaries": [],
            "message": "No protein residues found in final_results.",
        }
        if beam_protein_trace_dump_candidates:
            _write_json(
                candidates_path,
                {
                    "raw_candidates": [],
                    "processed_candidates": [],
                    "dedup_removed_mapping": {},
                },
            )
        _write_json(summary_path, empty_summary)
        return {
            "output_dir": out_dir,
            "protein_candidates_json": candidates_path
            if beam_protein_trace_dump_candidates
            else None,
            "beam_summary_json": summary_path,
            "protein_candidates_topk_cif": None,
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
        }

    na_mask_full = ~prot_mask_full
    residue_type_entropy, residue_type_confidence = _compute_residue_type_scores(
        final_results["pred_aatype"],
        prot_mask_full,
    )
    working_results["residue_type_entropy"] = residue_type_entropy
    working_results["residue_type_confidence"] = residue_type_confidence
    working_results["entropy_score_bfactor"] = _expand_residue_scalar_to_atom_bfactor(
        residue_type_confidence
    )

    confidence = _pred_rmsd_to_confidence(final_results["pred_rmsd"], prot_mask_full)
    working_results["confidence"] = np.repeat(confidence[..., None], 23, axis=-1)
    working_results["plddt"] = working_results["confidence"]

    dummy_aatype = np.zeros((len(final_results["pred_aatype"]),), dtype=np.int32)
    if np.any(prot_mask_full):
        dummy_aatype[prot_mask_full] = np.argmax(
            final_results["pred_aatype"][prot_mask_full][..., :num_prot],
            axis=-1,
        )
    if np.any(na_mask_full):
        dummy_aatype[na_mask_full] = (
            np.argmax(final_results["pred_aatype"][na_mask_full][..., num_prot:], axis=-1)
            + num_prot
            + 4
        )

    pred_torsions = select_torsion_angles(
        final_results["pred_torsions"],
        aatype=dummy_aatype,
        normalize=True,
    )
    pred_atom_pos, pred_atom_mask = affines_and_torsion_angles_to_atomc_pos(
        final_results["pred_affines"],
        pred_torsions,
        aatype=dummy_aatype,
    )
    pred_atom_pos = pred_atom_pos.cpu().numpy()
    pred_atom_mask = pred_atom_mask.cpu().numpy()

    existence_mask = np.ones_like(confidence, dtype=bool)
    if np.any(prot_mask_full):
        existence_mask[prot_mask_full] = remove_overlapping_ca(
            final_results["pred_affines"][prot_mask_full][..., :, -1],
            bfactors=confidence[prot_mask_full],
            existence_mask=None,
            radius_threshold=protein_radius_threshold,
        )

    pred_atom_pos = pred_atom_pos[existence_mask]
    pred_atom_mask = pred_atom_mask[existence_mask]
    prot_mask = prot_mask_full[existence_mask]
    confidence = confidence[existence_mask]
    survived_original_indices = np.nonzero(existence_mask)[0].astype(np.int32)

    reordered_final_results = {}
    for key, value in working_results.items():
        if key == "pred_edge_existence_dict":
            continue
        reordered_final_results[key] = value[existence_mask]
    reordered_final_results["_dummy_atom_pos"] = pred_atom_pos

    if not np.any(prot_mask):
        summary_path = pjoin(out_dir, "beam_summary.json")
        candidates_path = pjoin(out_dir, "protein_candidates.json")
        empty_summary = {
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
            "truncated_candidate_count": 0,
            "beam_settings": {
                "beam_width_protein": int(beam_width_protein),
                "trace_max_candidates": int(trace_max_candidates),
                "cn_radius_protein": float(cn_radius_protein),
                "ca_rescue_radius_protein": float(ca_rescue_radius_protein),
                "beam_protein_trace_hmm_rerank": bool(beam_protein_trace_hmm_rerank),
                "beam_protein_trace_iterative_refine": bool(
                    beam_protein_trace_iterative_refine
                ),
                "beam_protein_trace_max_refine_rounds": int(
                    beam_protein_trace_max_refine_rounds
                ),
            },
            "round_summaries": [],
            "message": "No protein residues survived the existence filter.",
        }
        if beam_protein_trace_dump_candidates:
            _write_json(
                candidates_path,
                {
                    "raw_candidates": [],
                    "processed_candidates": [],
                    "dedup_removed_mapping": {},
                },
            )
        _write_json(summary_path, empty_summary)
        return {
            "output_dir": out_dir,
            "protein_candidates_json": candidates_path
            if beam_protein_trace_dump_candidates
            else None,
            "beam_summary_json": summary_path,
            "protein_candidates_topk_cif": None,
            "raw_candidate_count": 0,
            "processed_candidate_count": 0,
        }

    protein_global_indices = np.arange(len(pred_atom_pos), dtype=np.int32)[prot_mask]
    protein_original_indices = survived_original_indices[prot_mask]
    protein_atom_pos = pred_atom_pos[prot_mask]
    protein_confidence = confidence[prot_mask]

    all_raw_candidates = []
    round_summaries = []
    frozen_mask = np.zeros((len(protein_atom_pos),), dtype=bool)
    num_rounds = (
        max(int(beam_protein_trace_max_refine_rounds), 1)
        if beam_protein_trace_iterative_refine
        else 1
    )
    last_beam_settings = None

    for round_idx in range(num_rounds):
        remaining_budget = max(int(trace_max_candidates) - len(all_raw_candidates), 0)
        if remaining_budget <= 0:
            round_summaries.append(
                {
                    "round_idx": int(round_idx),
                    "num_candidates": 0,
                    "stopped_reason": "candidate_cap_reached",
                    "num_frozen_nodes": int(np.sum(frozen_mask)),
                }
            )
            break

        beam_output = beam_trace_with_edge_dual_protein(
            protein_atom_pos,
            b_factors=protein_confidence,
            bond_distance_threshold=cn_radius_protein,
            last_idx=2,
            next_idx=0,
            edge_existence_dict=final_results["pred_edge_existence_dict"],
            idx_exist_to_original=protein_original_indices,
            beam_width=beam_width_protein,
            max_candidates=remaining_budget,
            max_path_length=len(protein_atom_pos),
            ca_rescue_radius=ca_rescue_radius_protein,
            frozen_mask=frozen_mask,
        )
        last_beam_settings = dict(beam_output["settings"])
        round_candidates = [
            _decorate_protein_beam_candidate(
                candidate,
                round_idx=round_idx,
                protein_local_indices=np.arange(len(protein_atom_pos), dtype=np.int32),
                protein_global_indices=protein_global_indices,
                protein_original_indices=protein_original_indices,
            )
            for candidate in beam_output["candidates"]
        ]

        if len(round_candidates) == 0:
            round_summaries.append(
                {
                    "round_idx": int(round_idx),
                    "num_candidates": 0,
                    "stopped_reason": "no_candidates",
                    "num_frozen_nodes": int(np.sum(frozen_mask)),
                }
            )
            break

        if beam_protein_trace_iterative_refine and len(prot_seqs) > 0:
            round_candidates, _ = _annotate_protein_beam_candidates_with_hmm(
                round_candidates,
                reordered_final_results,
                prot_seqs,
                hmm_temp_dir,
            )

        all_raw_candidates.extend(round_candidates)
        round_summary = {
            "round_idx": int(round_idx),
            "num_candidates": int(len(round_candidates)),
            "num_frozen_nodes": int(np.sum(frozen_mask)),
            "num_candidates_with_ca_rescue": int(
                sum(cand["num_ca_rescue_edges"] > 0 for cand in round_candidates)
            ),
        }

        if not beam_protein_trace_iterative_refine:
            round_summary["stopped_reason"] = "single_round_mode"
            round_summaries.append(round_summary)
            break

        anchors = _select_anchor_candidates_for_iterative_refine(round_candidates)
        round_summary["num_selected_anchors"] = int(len(anchors))
        round_summary["selected_anchor_candidate_uids"] = [
            cand["candidate_uid"] for cand in anchors
        ]
        if len(anchors) == 0:
            round_summary["stopped_reason"] = "no_anchor_candidates"
            round_summaries.append(round_summary)
            break

        new_frozen_mask = frozen_mask.copy()
        for anchor_candidate in anchors:
            anchor_local_indices = np.asarray(
                anchor_candidate["node_indices_local"], dtype=np.int32
            )
            new_frozen_mask[anchor_local_indices] = True

        if np.array_equal(new_frozen_mask, frozen_mask):
            round_summary["stopped_reason"] = "no_new_frozen_nodes"
            round_summaries.append(round_summary)
            break

        frozen_mask = new_frozen_mask
        round_summary["stopped_reason"] = "continue_refine"
        round_summaries.append(round_summary)

    for candidate_idx, candidate in enumerate(all_raw_candidates):
        candidate["candidate_id"] = int(candidate_idx)

    raw_candidates_snapshot = copy.deepcopy(all_raw_candidates)
    processed_candidates = copy.deepcopy(all_raw_candidates)

    if beam_protein_trace_hmm_rerank and len(processed_candidates) > 0 and len(prot_seqs) > 0:
        processed_candidates, _ = _annotate_protein_beam_candidates_with_hmm(
            processed_candidates,
            reordered_final_results,
            prot_seqs,
            hmm_temp_dir,
        )

    processed_candidates = _rank_protein_beam_candidates(processed_candidates)
    processed_candidates, duplicate_map = _deduplicate_protein_beam_candidates(
        processed_candidates
    )
    processed_candidates, conflict_map = _attach_protein_beam_conflicts(
        processed_candidates
    )
    processed_candidates = _rank_protein_beam_candidates(processed_candidates)

    topk_candidates_path = None
    if len(raw_candidates_snapshot) > 0:
        topk_candidates = sorted(
            raw_candidates_snapshot,
            key=lambda cand: (cand["score_total"], cand["path_length"]),
            reverse=True,
        )[: min(64, len(raw_candidates_snapshot))]
        topk_chains = [
            np.asarray(cand["node_indices_filtered"], dtype=np.int32)
            for cand in topk_candidates
            if cand["path_length"] > 0
        ]
        topk_chain_types = [
            np.argmax(
                np.asarray(reordered_final_results["pred_aatype"][chain], dtype=np.float32)[
                    ..., :num_prot
                ],
                axis=-1,
            ).astype(np.int32)
            for chain in topk_chains
        ]
        if len(topk_chains) > 0:
            topk_candidates_path = pjoin(out_dir, "protein_candidates_topk.cif")
            _write_chain_file(
                topk_candidates_path,
                reordered_final_results,
                topk_chains,
                topk_chain_types,
            )

    candidates_json_path = pjoin(out_dir, "protein_candidates.json")
    if beam_protein_trace_dump_candidates:
        _write_json(
            candidates_json_path,
            {
                "raw_candidates": [
                    _serializable_candidate(candidate)
                    for candidate in raw_candidates_snapshot
                ],
                "processed_candidates": [
                    _serializable_candidate(candidate)
                    for candidate in processed_candidates
                ],
                "dedup_removed_mapping": duplicate_map,
                "conflict_map": {
                    candidate_id: sorted(list(conflict_ids))
                    for candidate_id, conflict_ids in conflict_map.items()
                },
            },
        )

    raw_summary = _summarize_protein_beam_candidates(raw_candidates_snapshot)
    processed_summary = _summarize_protein_beam_candidates(processed_candidates)
    summary_payload = {
        "raw_candidate_count": int(len(raw_candidates_snapshot)),
        "processed_candidate_count": int(len(processed_candidates)),
        "truncated_candidate_count": int(min(len(raw_candidates_snapshot), int(trace_max_candidates))),
        "beam_settings": {
            "beam_width_protein": int(beam_width_protein),
            "trace_max_candidates": int(trace_max_candidates),
            "cn_radius_protein": float(cn_radius_protein),
            "ca_rescue_radius_protein": float(ca_rescue_radius_protein),
            "beam_protein_trace_dump_candidates": bool(
                beam_protein_trace_dump_candidates
            ),
            "beam_protein_trace_hmm_rerank": bool(beam_protein_trace_hmm_rerank),
            "beam_protein_trace_iterative_refine": bool(
                beam_protein_trace_iterative_refine
            ),
            "beam_protein_trace_max_refine_rounds": int(
                beam_protein_trace_max_refine_rounds
            ),
            "backend_settings": last_beam_settings,
        },
        "round_summaries": round_summaries,
        "raw_path_length_histogram": raw_summary["path_length_histogram"],
        "raw_score_histogram": raw_summary["score_histogram"],
        "processed_path_length_histogram": processed_summary["path_length_histogram"],
        "processed_score_histogram": processed_summary["score_histogram"],
        "raw_fraction_candidates_with_ca_rescue": raw_summary[
            "fraction_candidates_with_ca_rescue"
        ],
        "raw_fraction_edges_from_ca_rescue": raw_summary[
            "fraction_edges_from_ca_rescue"
        ],
        "processed_fraction_candidates_with_ca_rescue": processed_summary[
            "fraction_candidates_with_ca_rescue"
        ],
        "processed_fraction_edges_from_ca_rescue": processed_summary[
            "fraction_edges_from_ca_rescue"
        ],
        "num_protein_nodes_after_existence_filter": int(len(protein_atom_pos)),
        "num_duplicate_groups": int(len(duplicate_map)),
        "num_conflicting_processed_candidates": int(
            sum(
                len(candidate["overlap_metadata"]["conflict_candidate_ids"]) > 0
                for candidate in processed_candidates
            )
        ),
    }
    summary_path = pjoin(out_dir, "beam_summary.json")
    _write_json(summary_path, summary_payload)

    return {
        "output_dir": out_dir,
        "protein_candidates_json": candidates_json_path
        if beam_protein_trace_dump_candidates
        else None,
        "beam_summary_json": summary_path,
        "protein_candidates_topk_cif": topk_candidates_path,
        "raw_candidate_count": int(len(raw_candidates_snapshot)),
        "processed_candidate_count": int(len(processed_candidates)),
    }


def final_results_align_to_sequence(
    final_results,
    protein_seq_dir=None,
    dna_seq_dir=None,
    rna_seq_dir=None,
    output_dir="output",
    flag_prune_and_connect_chains=True,
    protein_radius_threshold=1.5,
    na_radius_threshold=1.5,
    min_na_chain_len=1,
    fallback_to_predicted_na_types=True,
    na_aa_logits_data=None,
):
    out_dir = abspath(output_dir)
    hmm_temp_dir = pjoin(out_dir, "hmm")
    os.makedirs(hmm_temp_dir, exist_ok=True)
    print(f"# Setting output directory to {out_dir}")
    print(f"# Setting HMM temp dir to {hmm_temp_dir}")

    prot_seqs = _read_sequences(protein_seq_dir, "protein") if protein_seq_dir is not None else []
    dna_seqs = _read_sequences(dna_seq_dir, "DNA") if dna_seq_dir is not None else []
    rna_seqs = _read_sequences(rna_seq_dir, "RNA") if rna_seq_dir is not None else []

    has_protein_input = len(prot_seqs) > 0
    has_na_input = (len(dna_seqs) + len(rna_seqs)) > 0

    prot_mask = np.asarray(final_results["prot_mask"], dtype=bool)
    if has_protein_input and (not has_na_input):
        print("# Protein-only mode: disable nucleic-acid postprocess outputs")
        prot_mask = np.ones_like(prot_mask, dtype=bool)
        na_mask = np.zeros_like(prot_mask, dtype=bool)
    elif has_na_input and (not has_protein_input):
        print("# Nucleic-acid-only mode: disable protein postprocess outputs")
        prot_mask = np.zeros_like(prot_mask, dtype=bool)
        na_mask = np.ones_like(prot_mask, dtype=bool)
    else:
        na_mask = ~prot_mask

    residue_type_entropy, residue_type_confidence = _compute_residue_type_scores(
        final_results["pred_aatype"],
        prot_mask,
    )
    final_results["residue_type_entropy"] = residue_type_entropy
    final_results["residue_type_confidence"] = residue_type_confidence
    final_results["entropy_score_bfactor"] = _expand_residue_scalar_to_atom_bfactor(
        residue_type_confidence
    )

    confidence = _pred_rmsd_to_confidence(final_results["pred_rmsd"], prot_mask)

    existence_mask = np.ones_like(confidence).astype(bool)
    if np.any(prot_mask):
        existence_mask[prot_mask] = remove_overlapping_ca(
            final_results["pred_affines"][prot_mask][..., :, -1],
            bfactors=confidence[prot_mask],
            existence_mask=None,
            radius_threshold=protein_radius_threshold,
        )
    if np.any(na_mask):
        existence_mask[na_mask] = remove_overlapping_ca(
            final_results["pred_affines"][na_mask][..., :, -1],
            bfactors=confidence[na_mask],
            existence_mask=None,
            radius_threshold=na_radius_threshold,
        )

    final_results["confidence"] = np.repeat(confidence[..., None], 23, axis=-1)
    final_results["plddt"] = final_results["confidence"]

    dummy_aatype = np.zeros((len(final_results["pred_aatype"]),), dtype=np.int32)
    if np.any(prot_mask):
        dummy_aatype[prot_mask] = np.argmax(
            final_results["pred_aatype"][prot_mask][..., :num_prot],
            axis=-1,
        )
    if np.any(na_mask):
        dummy_aatype[na_mask] = (
            np.argmax(final_results["pred_aatype"][na_mask][..., num_prot:], axis=-1)
            + num_prot
            + 4
        )

    pred_torsions = select_torsion_angles(
        final_results["pred_torsions"],
        aatype=dummy_aatype,
        normalize=True,
    )
    pred_atom_pos, pred_atom_mask = affines_and_torsion_angles_to_atomc_pos(
        final_results["pred_affines"],
        pred_torsions,
        aatype=dummy_aatype,
    )
    pred_atom_pos = pred_atom_pos.cpu().numpy()
    pred_atom_mask = pred_atom_mask.cpu().numpy()

    raw_before_existence_filter_path = None
    raw_protein_before_existence_filter_path = None
    raw_na_before_existence_filter_path = None
    residue_type_scores_before_existence_filter_path = os.path.join(
        out_dir, "residue_type_scores_before_existence_filter.npz"
    )
    _write_residue_type_score_npz(
        residue_type_scores_before_existence_filter_path,
        final_results,
    )
    if np.any(prot_mask):
        raw_protein_before_existence_filter_path = os.path.join(
            out_dir, "protein_before_existence_filter.cif"
        )
        _write_residue_subset_file(
            raw_protein_before_existence_filter_path,
            pred_atom_pos[prot_mask],
            pred_atom_mask[prot_mask],
            dummy_aatype[prot_mask],
            residue_index=np.arange(int(np.sum(prot_mask)), dtype=np.int32),
            bfactor=final_results["plddt"][prot_mask],
        )
        print("# Raw protein residues before existence filter = {}".format(int(np.sum(prot_mask))))
    if np.any(na_mask):
        raw_na_before_existence_filter_path = os.path.join(
            out_dir, "na_before_existence_filter.cif"
        )
        _write_residue_subset_file(
            raw_na_before_existence_filter_path,
            pred_atom_pos[na_mask],
            pred_atom_mask[na_mask],
            dummy_aatype[na_mask],
            residue_index=np.arange(int(np.sum(na_mask)), dtype=np.int32),
            bfactor=final_results["plddt"][na_mask],
        )
        print("# Raw NA residues before existence filter = {}".format(int(np.sum(na_mask))))

    raw_before_existence_filter_path = os.path.join(out_dir, "before_existence_filter.cif")
    _write_residue_subset_file(
        raw_before_existence_filter_path,
        pred_atom_pos,
        pred_atom_mask,
        dummy_aatype,
        residue_index=np.arange(len(pred_atom_pos), dtype=np.int32),
        bfactor=final_results["plddt"],
    )
    print("# Raw all residues before existence filter = {}".format(len(pred_atom_pos)))

    pred_atom_pos = pred_atom_pos[existence_mask]
    pred_atom_mask = pred_atom_mask[existence_mask]

    prot_mask = prot_mask[existence_mask]
    na_mask = na_mask[existence_mask]
    confidence = confidence[existence_mask]

    reordered_final_results = {}
    for key, value in final_results.items():

        # Must not reorder pred_edge_existence_dict
        # Afterwards, we have idx_exist_to_original to indicate the correspondence between the original and the existed
        if key == "pred_edge_existence_dict":
            continue
        reordered_final_results[key] = value[existence_mask]
        
    reordered_final_results["_dummy_atom_pos"] = pred_atom_pos

    residue_type_scores_after_existence_filter_path = os.path.join(
        out_dir, "residue_type_scores_after_existence_filter.npz"
    )
    _write_residue_type_score_npz(
        residue_type_scores_after_existence_filter_path,
        reordered_final_results,
    )

    prot_chains = []
    if has_protein_input and np.any(prot_mask):
        idx_exist_to_original = np.arange(len(pred_atom_pos), dtype=np.int32)[prot_mask]
        traced = flood_fill_with_edge_dual(
            pred_atom_pos[prot_mask],
            confidence[prot_mask],
            bond_distance_threshold=2.1,
            last_idx=2,
            next_idx=0,
            edge_existence_dict=final_results["pred_edge_existence_dict"],
            idx_exist_to_original=idx_exist_to_original,
        )
        idxs = np.arange(len(pred_atom_pos))[prot_mask]
        prot_chains = [idxs[c] for c in traced]
        print(f"# Trace protein into {len(prot_chains)} dummy chains")

    na_chains = []
    if has_na_input and np.any(na_mask):
        idx_exist_to_original = np.arange(len(pred_atom_pos), dtype=np.int32)[na_mask]
        traced = flood_fill_with_edge_dual(
            pred_atom_pos[na_mask],
            confidence[na_mask],
            bond_distance_threshold=3.6,
            last_idx=6,
            next_idx=8,
            edge_existence_dict=final_results["pred_edge_existence_dict"],
            idx_exist_to_original=idx_exist_to_original,
        )
        idxs = np.arange(len(pred_atom_pos))[na_mask]
        na_chains = [idxs[c] for c in traced]
        print(f"# Trace NA into {len(na_chains)} dummy chains")

    (
        prot_before_chains,
        prot_before_chain_types,
        prot_after_chains,
        prot_after_chain_types,
    ) = _build_protein_chain_outputs(
        reordered_final_results,
        prot_chains,
        prot_seqs,
        hmm_temp_dir,
        pred_atom_pos,
    )
    (
        na_before_chains,
        na_before_chain_types,
        na_after_chains,
        na_after_chain_types,
        na_before_fallback_masks,
        na_after_fallback_masks,
    ) = _build_na_chain_outputs(
        reordered_final_results,
        na_chains,
        rna_seqs,
        dna_seqs,
        hmm_temp_dir,
        na_aa_logits_data=na_aa_logits_data,
        min_chain_len=min_na_chain_len,
        fallback_to_predicted_na_types=fallback_to_predicted_na_types,
    )

    before_prune_path = None
    after_prune_path = None
    protein_before_prune_path = None
    protein_after_prune_path = None
    na_before_prune_path = None
    na_after_prune_path = None
    if len(prot_before_chains) > 0:
        _print_chain_stats("Protein before prune", prot_before_chains)
        protein_before_prune_path = os.path.join(out_dir, "protein_before_prune.cif")
        _write_chain_file(
            protein_before_prune_path,
            reordered_final_results,
            prot_before_chains,
            prot_before_chain_types,
        )

    if len(prot_after_chains) > 0:
        _print_chain_stats("Protein after prune", prot_after_chains)
        protein_after_prune_path = os.path.join(out_dir, "protein_after_prune.cif")
        _write_chain_file(
            protein_after_prune_path,
            reordered_final_results,
            prot_after_chains,
            prot_after_chain_types,
        )

    if len(na_before_chains) > 0:
        _print_chain_stats("NA before prune", na_before_chains)
        na_before_prune_path = os.path.join(out_dir, "na_before_prune.cif")
        _write_chain_file(
            na_before_prune_path,
            reordered_final_results,
            na_before_chains,
            na_before_chain_types,
        )

    if len(na_after_chains) > 0:
        _print_chain_stats("NA after prune", na_after_chains)
        na_after_prune_path = os.path.join(out_dir, "na_after_prune.cif")
        _write_chain_file(
            na_after_prune_path,
            reordered_final_results,
            na_after_chains,
            na_after_chain_types,
        )

    all_before_chains = []
    all_before_chain_types = []
    if len(prot_before_chains) > 0:
        all_before_chains.extend(prot_before_chains)
        all_before_chain_types.extend(prot_before_chain_types)
    if len(na_before_chains) > 0:
        all_before_chains.extend(na_before_chains)
        all_before_chain_types.extend(na_before_chain_types)

    if len(all_before_chains) > 0:
        before_prune_path = os.path.join(out_dir, "before_prune.cif")
        _print_chain_stats("All before prune", all_before_chains)
        _write_chain_file(
            before_prune_path,
            reordered_final_results,
            all_before_chains,
            all_before_chain_types,
        )
    before_prune_state = _build_chain_order_recycle_state(
        reordered_final_results,
        all_before_chains,
    )

    all_after_chains = []
    all_after_chain_types = []
    if len(prot_after_chains) > 0:
        all_after_chains.extend(prot_after_chains)
        all_after_chain_types.extend(prot_after_chain_types)
    if len(na_after_chains) > 0:
        all_after_chains.extend(na_after_chains)
        all_after_chain_types.extend(na_after_chain_types)

    if len(all_after_chains) > 0:
        after_prune_path = os.path.join(out_dir, "after_prune.cif")
        _print_chain_stats("All after prune", all_after_chains)
        _write_chain_file(
            after_prune_path,
            reordered_final_results,
            all_after_chains,
            all_after_chain_types,
        )

    all_chains = []
    all_chain_types = []
    if len(prot_after_chains) > 0:
        all_chains.extend(prot_after_chains)
        all_chain_types.extend(prot_after_chain_types)
    if len(na_after_chains) > 0:
        all_chains.extend(na_after_chains)
        all_chain_types.extend(na_after_chain_types)

    fpdbout = os.path.join(out_dir, "output.cif")
    _write_chain_file(
        fpdbout,
        reordered_final_results,
        all_chains,
        all_chain_types,
    )
    entropy_final_results = dict(reordered_final_results)
    entropy_score_bfactor = np.array(
        reordered_final_results["entropy_score_bfactor"],
        copy=True,
    )
    fallback_entropy_mask = np.zeros((len(entropy_score_bfactor),), dtype=bool)
    for chain, fallback_mask in zip(na_after_chains, na_after_fallback_masks):
        if len(chain) == 0 or len(fallback_mask) == 0:
            continue
        fallback_entropy_mask[np.asarray(chain, dtype=np.int32)[np.asarray(fallback_mask, dtype=bool)]] = True
    if np.any(fallback_entropy_mask):
        entropy_score_bfactor[fallback_entropy_mask] = FALLBACK_CONFIDENCE_SENTINEL
    entropy_final_results["entropy_score_bfactor"] = entropy_score_bfactor
    entropy_output_path = os.path.join(out_dir, "output_entropy_score.cif")
    _write_chain_file(
        entropy_output_path,
        entropy_final_results,
        all_chains,
        all_chain_types,
        bfactor_key="entropy_score_bfactor",
    )
    output_state = _build_chain_order_recycle_state(
        reordered_final_results,
        all_chains,
    )
    next_round_state = before_prune_state if before_prune_path is not None else output_state
    protein_after_num_res = int(sum(len(chain) for chain in prot_after_chains))
    na_after_num_res = int(sum(len(chain) for chain in na_after_chains))
    return {
        "output_path": fpdbout,
        "output_entropy_score_path": entropy_output_path,
        "before_prune_path": before_prune_path,
        "after_prune_path": after_prune_path,
        "raw_before_existence_filter_path": raw_before_existence_filter_path,
        "raw_protein_before_existence_filter_path": raw_protein_before_existence_filter_path,
        "raw_na_before_existence_filter_path": raw_na_before_existence_filter_path,
        "residue_type_scores_before_existence_filter_path": residue_type_scores_before_existence_filter_path,
        "residue_type_scores_after_existence_filter_path": residue_type_scores_after_existence_filter_path,
        "protein_before_prune_path": protein_before_prune_path,
        "protein_after_prune_path": protein_after_prune_path,
        "na_before_prune_path": na_before_prune_path,
        "na_after_prune_path": na_after_prune_path,
        "protein_after_num_res": protein_after_num_res,
        "na_after_num_res": na_after_num_res,
        "before_prune_state": before_prune_state,
        "output_state": output_state,
        "next_round_state": next_round_state,
    }
