import argparse
import numpy as np

from em3dfold.utils.qscore.mrc_utils import load_mrc
from em3dfold.utils.qscore.pdb_utils import get_protein_from_file_path
from em3dfold.utils.qscore.q_score import calculate_q_score


def add_args(parser):
    parser.add_argument("--struct", "-s")
    parser.add_argument("--map", "-m")
    return parser


def _print_qscore_stats(name, q_scores):
    if q_scores.size == 0:
        print(f"{name}: no atoms")
        return
    print(
        f"{name}: Mean {np.mean(q_scores):.6f} "
        f"Min {np.min(q_scores):.6f} Max {np.max(q_scores):.6f}"
    )


def main(args):
    np.random.seed(42)

    prot = get_protein_from_file_path(args.struct)
    cryo_map = load_mrc(args.map)

    mask = prot.atom_mask.astype(bool)
    atoms = prot.atom_positions[mask]
    q_scores = calculate_q_score(atoms, cryo_map)

    q_score_per_atom = np.zeros_like(prot.atom_mask, dtype=np.float32)
    q_score_per_atom[mask] = q_scores

    q_score_per_residue = np.zeros_like(mask, dtype=np.float32)
    q_score_per_residue[mask] = q_scores
    q_score_per_residue = q_score_per_residue.sum(axis=1) / (1e-6 + mask.sum(axis=1))

    prot_res_mask = prot.aatype < 20
    na_res_mask = (prot.aatype >= 20) & (prot.aatype < 28)
    prot_atom_mask = mask[prot_res_mask]
    na_atom_mask = mask[na_res_mask]

    _print_qscore_stats("All atoms", q_scores)

    if np.any(prot_res_mask):
        _print_qscore_stats("Protein atoms", q_score_per_atom[prot_res_mask][prot_atom_mask])
        _print_qscore_stats(
            "Protein residues",
            q_score_per_residue[prot_res_mask][prot_atom_mask.any(axis=1)],
        )

    if np.any(na_res_mask):
        _print_qscore_stats("Nucleic acid atoms", q_score_per_atom[na_res_mask][na_atom_mask])
        _print_qscore_stats(
            "Nucleic acid residues",
            q_score_per_residue[na_res_mask][na_atom_mask.any(axis=1)],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

