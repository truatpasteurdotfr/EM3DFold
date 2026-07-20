import argparse
import numpy as np

from em3dfold.utils.qscore.mrc_utils import load_mrc
from em3dfold.utils.qscore.pdb_utils import get_protein_from_file_path
from em3dfold.utils.qscore.q_score import calculate_q_score


def add_args(parser):
    parser.add_argument("--struct", "-s")
    parser.add_argument("--map", "-m")
    parser.add_argument(
        "--min-atom-qscore", "--min-residue-qscore",
        dest="min_atom_qscore",
        type=float,
        default=-100.0,
        help="Ignore atoms with qscore below this cutoff before computing residue statistics.",
    )
    return parser


def _print_qscore_stats(name, q_scores):
    if q_scores.size == 0:
        print(f"{name}: no values")
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

    q_score_per_atom = np.full(prot.atom_mask.shape, np.nan, dtype=np.float32)
    q_score_per_atom[mask] = q_scores

    atom_cutoff = float(args.min_atom_qscore)
    valid_atom_mask = np.logical_and(mask & np.isfinite(q_score_per_atom), q_score_per_atom >= atom_cutoff)
    q_score_per_residue = np.full((mask.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid_atom_mask):
        atom_sum = np.where(valid_atom_mask, q_score_per_atom, 0.0).sum(axis=1, dtype=np.float64)
        atom_count = valid_atom_mask.sum(axis=1)
        nonzero = atom_count > 0
        q_score_per_residue[nonzero] = (atom_sum[nonzero] / atom_count[nonzero]).astype(np.float32)

    prot_res_mask = prot.aatype < 20
    na_res_mask = (prot.aatype >= 20) & (prot.aatype < 28)
    prot_atom_mask = mask[prot_res_mask]
    na_atom_mask = mask[na_res_mask]

    print(f"Atom qscore cutoff: >= {atom_cutoff:.6f}")

    _print_qscore_stats("All atoms", q_scores)

    if np.any(prot_res_mask):
        _print_qscore_stats("Protein atoms", q_score_per_atom[prot_res_mask][prot_atom_mask])
        prot_residue_scores = q_score_per_residue[prot_res_mask][prot_atom_mask.any(axis=1)]
        prot_residue_scores = prot_residue_scores[np.isfinite(prot_residue_scores)]
        _print_qscore_stats("Protein residues", prot_residue_scores)

    if np.any(na_res_mask):
        _print_qscore_stats("Nucleic acid atoms", q_score_per_atom[na_res_mask][na_atom_mask])
        na_residue_scores = q_score_per_residue[na_res_mask][na_atom_mask.any(axis=1)]
        na_residue_scores = na_residue_scores[np.isfinite(na_residue_scores)]
        _print_qscore_stats("Nucleic acid residues", na_residue_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

