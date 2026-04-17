import argparse

import numpy as np
import torch

from em3dfold.io.pdbio import read_pdb
from em3dfold.polymer_utils import residue_constants as rc
from em3dfold.train.losses.violation import find_structural_violations


def add_args(parser):
    parser.add_argument("--struct", "-s", required=True, help="Input PDB/mmCIF structure")
    parser.add_argument(
        "--violation-tolerance-factor",
        type=float,
        default=12.0,
        help="Tolerance factor for bond and angle violations",
    )
    parser.add_argument(
        "--clash-overlap-tolerance",
        type=float,
        default=1.5,
        help="Overlap tolerance for steric clash detection",
    )
    return parser


def _to_scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _build_batch(struct_path: str):
    atom_pos, atom_mask, aatype, residue_index, chain_index, _ = read_pdb(struct_path)
    batch = {
        "atom23_atom_exists": torch.tensor(atom_mask, dtype=torch.float32),
        "residue_index": torch.tensor(residue_index, dtype=torch.long),
        "aatype": torch.tensor(aatype, dtype=torch.long),
        "residx_atomc_to_atomf": torch.tensor(
            rc.residx_atomc_to_atomf[aatype], dtype=torch.long
        ),
    }
    pred_atom_positions = torch.tensor(atom_pos, dtype=torch.float32)
    return batch, pred_atom_positions, np.asarray(chain_index), np.asarray(aatype)


def _print_structure_summary(chain_index, aatype):
    num_res = int(len(chain_index))
    num_chain = int(chain_index.max() + 1) if num_res > 0 else 0
    prot_mask = aatype < 20
    na_mask = (aatype >= 20) & (aatype < 28)
    print(f"# Residues = {num_res}")
    print(f"# Chains = {num_chain}")
    print(f"# Protein residues = {int(prot_mask.sum())}")
    print(f"# Nucleic acid residues = {int(na_mask.sum())}")


def main(args):
    batch, pred_atom_positions, chain_index, aatype = _build_batch(args.struct)
    _print_structure_summary(chain_index, aatype)

    violations = find_structural_violations(
        batch=batch,
        atom23_pred_positions=pred_atom_positions,
        violation_tolerance_factor=args.violation_tolerance_factor,
        clash_overlap_tolerance=args.clash_overlap_tolerance,
    )

    between = violations["between_residues"]
    within = violations["within_residues"]
    total_res_violation_mask = violations["total_per_residue_violations_mask"]

    per_atom_between_num = between["clashes_per_atum_num_clash"]
    per_atom_within_num = within["per_atom_num_clash"]
    per_atom_total_num = per_atom_between_num + per_atom_within_num

    per_atom_between_mask = between["clashes_per_atom_clash_mask"].bool()
    per_atom_within_mask = within["per_atom_violations"].bool()
    per_atom_total_mask = per_atom_between_mask | per_atom_within_mask

    per_res_between_mask = torch.any(per_atom_between_mask, dim=-1)
    per_res_within_mask = torch.any(per_atom_within_mask, dim=-1)
    per_res_total_atom_mask = torch.any(per_atom_total_mask, dim=-1)

    prot_mask = batch["aatype"] < 20
    na_mask = (batch["aatype"] >= 20) & (batch["aatype"] < 28)

    print("# Violation statistics")
    print(f"bond_loss_mean = {_to_scalar(between['bonds_loss_mean']):.6f}")
    print(f"angle_ca_c_n_loss_mean = {_to_scalar(between['angles_ca_c_n_loss_mean']):.6f}")
    print(f"angle_c_n_ca_loss_mean = {_to_scalar(between['angles_c_n_ca_loss_mean']):.6f}")
    print(f"between_residue_clash_mean_loss = {_to_scalar(between['clashes_mean_loss']):.6f}")
    print(
        f"within_residue_loss_mean = "
        f"{_to_scalar(torch.mean(within['per_atom_loss_sum'])):.6f}"
    )

    print("# Counts")
    print(f"residues_with_any_violation = {int(total_res_violation_mask.sum().item())}")
    print(
        f"residue_violation_fraction = "
        f"{_to_scalar(total_res_violation_mask.float().mean()):.6f}"
    )
    print(f"residues_with_between_residue_clash = {int(per_res_between_mask.sum().item())}")
    print(f"residues_with_within_residue_violation = {int(per_res_within_mask.sum().item())}")
    print(f"residues_with_any_atom_clash = {int(per_res_total_atom_mask.sum().item())}")
    print(f"atoms_with_any_clash = {int(per_atom_total_mask.sum().item())}")
    print(f"total_atom_clash_count = {_to_scalar(torch.sum(per_atom_total_num)):.0f}")

    if torch.any(prot_mask):
        print("# Protein-only")
        print(
            f"protein_residues_with_any_violation = "
            f"{int(total_res_violation_mask[prot_mask].sum().item())}"
        )
        print(
            f"protein_atoms_with_any_clash = "
            f"{int(per_atom_total_mask[prot_mask].sum().item())}"
        )
        print(
            f"protein_total_atom_clash_count = "
            f"{_to_scalar(torch.sum(per_atom_total_num[prot_mask])):.0f}"
        )

    if torch.any(na_mask):
        print("# Nucleic-acid-only")
        print(
            f"na_residues_with_any_violation = "
            f"{int(total_res_violation_mask[na_mask].sum().item())}"
        )
        print(
            f"na_atoms_with_any_clash = "
            f"{int(per_atom_total_mask[na_mask].sum().item())}"
        )
        print(
            f"na_total_atom_clash_count = "
            f"{_to_scalar(torch.sum(per_atom_total_num[na_mask])):.0f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

