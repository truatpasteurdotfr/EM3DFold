import argparse
import csv
import os
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import ALL_MOL_TYPES, _filter_structure, _print_eval_result, eval_local

warnings.filterwarnings("ignore")

LABELS = ("interface", "surface", "core")
VALID_RESNAME_3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "A", "G", "C", "U", "DA", "DG", "DC", "DT",
}


def add_args(parser):
    parser.add_argument("--target", "-t", required=True, help="Target structure")
    parser.add_argument("--query", "-q", required=True, help="Query structure")
    parser.add_argument("--labels", required=True, help="TSV output from precompute_interface_labels.py")
    parser.add_argument("--eval-chain", action="store_true", help="Eval in chain level")
    parser.add_argument("-d", type=float, default=3.0, help="Distance cutoff")
    parser.add_argument("--align", action="store_true", default=False)
    parser.add_argument(
        "--mol",
        choices=ALL_MOL_TYPES,
        default=None,
        help="Optional compatibility selector. If omitted, evaluate protein, nucleic and complex together.",
    )
    return parser


def _build_parser_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return PDBParser(QUIET=True)
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path}")


def _extract_residue_keys(structure_path: str):
    path = Path(structure_path)
    parser = _build_parser_for_path(path)
    structure = parser.get_structure(path.stem or "model", str(path))
    model = structure[0]

    residue_keys = []
    for chain in model:
        prev_residue_number = None
        for residue in chain:
            residue_number = residue.get_id()[1]
            if prev_residue_number is None or residue_number != prev_residue_number:
                resname_3 = residue.get_resname().strip()
                if resname_3 not in VALID_RESNAME_3:
                    continue
                prev_residue_number = residue_number
            hetfield, resseq, icode = residue.get_id()
            if hetfield != " ":
                continue
            residue_keys.append((str(chain.id).strip(), str(resseq), str(icode).strip(), residue.get_resname().strip()))
    return residue_keys


def _load_label_map(label_path: str):
    label_map = {}
    with Path(label_path).expanduser().resolve().open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            key = (row["chain"], row["resid"], row["icode"], row["resname"])
            label_map[key] = row["label"]
    return label_map


def _label_mask_for_mol(atom_mask, res_type, label_values, mol, target_label):
    has_anchor = atom_mask[..., 1].astype(bool)
    if mol == "protein":
        mol_mask = np.logical_and(has_anchor, res_type < 20)
    elif mol == "nucleic":
        mol_mask = np.logical_and(has_anchor, res_type >= 20)
    else:
        mol_mask = has_anchor
    label_mask = np.asarray([x == target_label for x in label_values], dtype=bool)
    return np.logical_and(mol_mask, label_mask)


def _print_label_eval_result(scope_name, label, metrics, query_len, target_len):
    if metrics is None:
        print(
            f"# {scope_name} interface {label} target_len = {target_len} query_len = {query_len} "
            "CX_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = -"
        )
        return
    print(
        "# {} interface {} target_len = {} query_len = {} CX_RMSD = {:.4f} BB_RMSD = {:.4f} Cov = {:.4f} "
        "Seq_Match = {:.4f} Seq_Recall = {:.4f}".format(
            scope_name,
            label,
            target_len,
            query_len,
            metrics["c4_rmsd"],
            metrics["bb_rmsd"],
            metrics["cov"],
            metrics["seq_match"],
            metrics["seq_recall"],
        )
    )


def _evaluate_one_mol_interface(
    mol,
    tgt_atom_pos,
    tgt_atom_mask,
    tgt_res_type,
    tgt_res_idx,
    tgt_chain_idx,
    tgt_label_values,
    qry_atom_pos,
    qry_atom_mask,
    qry_res_type,
    qry_res_idx,
    qry_chain_idx,
    args,
):
    target_path = os.path.abspath(args.target)
    query_path = os.path.abspath(args.query)
    labels_path = os.path.abspath(args.labels)

    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, tgt_res_idx_m, tgt_chain_idx_m = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, qry_res_idx_m, qry_chain_idx_m = _filter_structure(
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )
    if mol == "protein":
        keep_mask = np.logical_and(tgt_atom_mask[..., 1].astype(bool), tgt_res_type < 20)
    elif mol == "nucleic":
        keep_mask = np.logical_and(tgt_atom_mask[..., 1].astype(bool), tgt_res_type >= 20)
    else:
        keep_mask = tgt_atom_mask[..., 1].astype(bool)
    tgt_labels_m = [tgt_label_values[i] for i in np.where(keep_mask)[0]]

    print("#" + "-" * 72)
    print(f"# Eval {mol}")
    print(f"# Read {len(tgt_atom_pos_m)} valid residues from {target_path}")
    print(f"# Read {len(qry_atom_pos_m)} valid residues from {query_path}")
    print(f"# Read interface labels from {labels_path}")

    if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
        print(f"# Skip {mol} evaluation because target or query has no valid residues")
        return

    print(f"# Eval {mol} all len = {len(qry_atom_pos_m)}")
    metrics = eval_local(
        tgt_atom_pos_m,
        tgt_atom_mask_m,
        tgt_res_type_m,
        qry_atom_pos_m,
        qry_atom_mask_m,
        qry_res_type_m,
        r=args.d,
        mol=mol,
        align=args.align,
    )
    _print_eval_result(f"{mol} all", metrics, len(qry_atom_pos_m))

    if args.eval_chain:
        qry_n_chain = qry_chain_idx_m.max() + 1
        for i in range(qry_n_chain):
            sel_idxs = np.arange(0, len(qry_atom_pos_m))[qry_chain_idx_m == i]
            print(f"# Eval {mol} chain {i} len = {len(sel_idxs)}")
            metrics = eval_local(
                tgt_atom_pos_m,
                tgt_atom_mask_m,
                tgt_res_type_m,
                qry_atom_pos_m[sel_idxs],
                qry_atom_mask_m[sel_idxs],
                qry_res_type_m[sel_idxs],
                r=args.d,
                mol=mol,
                align=args.align,
            )
            _print_eval_result(f"{mol} chain {i}", metrics, len(sel_idxs))

    for label in LABELS:
        target_mask = np.asarray([x == label for x in tgt_labels_m], dtype=bool)
        target_len = int(np.sum(target_mask))
        print(f"# Eval {mol} interface category {label} target_len = {target_len}")
        if target_len == 0:
            _print_label_eval_result(f"{mol} all", label, None, len(qry_atom_pos_m), 0)
            if args.eval_chain:
                qry_n_chain = qry_chain_idx_m.max() + 1
                for i in range(qry_n_chain):
                    sel_idxs = np.arange(0, len(qry_atom_pos_m))[qry_chain_idx_m == i]
                    _print_label_eval_result(f"{mol} chain {i}", label, None, len(sel_idxs), 0)
            continue

        tgt_atom_pos_lab = tgt_atom_pos_m[target_mask]
        tgt_atom_mask_lab = tgt_atom_mask_m[target_mask]
        tgt_res_type_lab = tgt_res_type_m[target_mask]

        metrics = eval_local(
            tgt_atom_pos_lab,
            tgt_atom_mask_lab,
            tgt_res_type_lab,
            qry_atom_pos_m,
            qry_atom_mask_m,
            qry_res_type_m,
            r=args.d,
            mol=mol,
            align=args.align,
        )
        _print_label_eval_result(f"{mol} all", label, metrics, len(qry_atom_pos_m), target_len)

        if args.eval_chain:
            qry_n_chain = qry_chain_idx_m.max() + 1
            for i in range(qry_n_chain):
                sel_idxs = np.arange(0, len(qry_atom_pos_m))[qry_chain_idx_m == i]
                metrics = eval_local(
                    tgt_atom_pos_lab,
                    tgt_atom_mask_lab,
                    tgt_res_type_lab,
                    qry_atom_pos_m[sel_idxs],
                    qry_atom_mask_m[sel_idxs],
                    qry_res_type_m[sel_idxs],
                    r=args.d,
                    mol=mol,
                    align=args.align,
                )
                _print_label_eval_result(f"{mol} chain {i}", label, metrics, len(sel_idxs), target_len)


def main(args):
    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(args.target, return_bfactor=False)
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(args.query, return_bfactor=False)
    tgt_residue_keys = _extract_residue_keys(args.target)
    if len(tgt_residue_keys) != len(tgt_atom_pos):
        raise RuntimeError(
            f"target residue-key count mismatch: keys={len(tgt_residue_keys)} read_pdb={len(tgt_atom_pos)}"
        )
    label_map = _load_label_map(args.labels)
    tgt_label_values = []
    for key in tgt_residue_keys:
        if key not in label_map:
            raise KeyError(f"Missing interface label for target residue {key} in {args.labels}")
        tgt_label_values.append(label_map[key])

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Interface categories = {}".format(", ".join(LABELS)))
    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES
    for mol in eval_mols:
        _evaluate_one_mol_interface(
            mol,
            tgt_atom_pos,
            tgt_atom_mask,
            tgt_res_type,
            tgt_res_idx,
            tgt_chain_idx,
            tgt_label_values,
            qry_atom_pos,
            qry_atom_mask,
            qry_res_type,
            qry_res_idx,
            qry_chain_idx,
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
