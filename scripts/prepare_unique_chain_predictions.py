#!/usr/bin/env python3
"""Report identical protein-chain SEQRES groups from a list of mmCIF files."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def _ensure_repo_paths_on_path() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    src_dir = repo_root / "src"
    for path in (script_dir, src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_paths_on_path()
from get_entity_uniprot_ids import _ensure_list, _normalize_token, _parse_mmcif_fields  # noqa: E402


def _find_structure_path(structure_dir: Path, pdbid: str) -> Path:
    candidates = [
        structure_dir / f"{pdbid}.cif",
        structure_dir / f"{pdbid}.mmcif",
        structure_dir / pdbid / f"{pdbid}.cif",
        structure_dir / pdbid / f"{pdbid}.mmcif",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot find mmCIF file for {pdbid} under {structure_dir}")


def _normalize_seqres(text: str | None) -> str | None:
    token = _normalize_token(text)
    if token is None:
        return None
    compact = "".join(str(token).split()).upper()
    return compact or None


def _protein_entity_ids(mmcif: dict) -> set[str]:
    entity_ids = _ensure_list(mmcif.get("_entity_poly.entity_id"))
    poly_types = _ensure_list(mmcif.get("_entity_poly.type"))

    protein_ids: set[str] = set()
    row_count = max(len(entity_ids), len(poly_types))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        poly_type = _normalize_token(poly_types[idx] if idx < len(poly_types) else None)
        if entity_id is None or poly_type is None:
            continue
        if "polypeptide" in poly_type.lower():
            protein_ids.add(entity_id)
    return protein_ids


def _entity_to_chain_ids(mmcif: dict, protein_ids: set[str]) -> dict[str, list[str]]:
    entity_ids = _ensure_list(mmcif.get("_entity_poly.entity_id"))
    strand_ids = _ensure_list(mmcif.get("_entity_poly.pdbx_strand_id"))

    mapping: dict[str, list[str]] = {}
    row_count = max(len(entity_ids), len(strand_ids))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        strand_id_text = _normalize_token(strand_ids[idx] if idx < len(strand_ids) else None)
        if entity_id is None or entity_id not in protein_ids or strand_id_text is None:
            continue
        chain_ids = [
            chain_id.strip()
            for chain_id in str(strand_id_text).split(",")
            if chain_id.strip() and chain_id.strip() not in {"?", "."}
        ]
        if chain_ids:
            mapping[entity_id] = chain_ids
    return mapping


def _entity_to_seqres(mmcif: dict, protein_ids: set[str]) -> dict[str, str]:
    entity_ids = _ensure_list(mmcif.get("_entity_poly.entity_id"))
    seqres_can = _ensure_list(mmcif.get("_entity_poly.pdbx_seq_one_letter_code_can"))
    seqres_raw = _ensure_list(mmcif.get("_entity_poly.pdbx_seq_one_letter_code"))

    mapping: dict[str, str] = {}
    row_count = max(len(entity_ids), len(seqres_can), len(seqres_raw))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        if entity_id is None or entity_id not in protein_ids:
            continue
        sequence = _normalize_seqres(seqres_can[idx] if idx < len(seqres_can) else None)
        if sequence is None:
            sequence = _normalize_seqres(seqres_raw[idx] if idx < len(seqres_raw) else None)
        if sequence is not None:
            mapping[entity_id] = sequence
    return mapping


def _collect_chain_seqres(cif_path: Path, pdbid: str) -> list[tuple[str, str]]:
    mmcif = _parse_mmcif_fields(cif_path)
    protein_ids = _protein_entity_ids(mmcif)
    entity_to_chains = _entity_to_chain_ids(mmcif, protein_ids)
    entity_to_seqres = _entity_to_seqres(mmcif, protein_ids)

    rows: list[tuple[str, str]] = []
    for entity_id in sorted(protein_ids, key=lambda value: (len(value), value)):
        sequence = entity_to_seqres.get(entity_id)
        chain_ids = entity_to_chains.get(entity_id, [])
        if sequence is None:
            print(f"skip\t{pdbid}\tentity={entity_id}\treason=no_seqres")
            continue
        if not chain_ids:
            print(f"skip\t{pdbid}\tentity={entity_id}\treason=no_chain_ids")
            continue
        for chain_id in chain_ids:
            rows.append((f"{pdbid}.chain.{chain_id}", sequence))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a list of PDB IDs and report which protein chains have identical SEQRES.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--list",
        help="Text file; the first column is pdbid",
    )
    input_group.add_argument(
        "--pdb",
        help="Single pdbid to process",
    )
    parser.add_argument(
        "--structure-dir",
        required=True,
        help="Directory containing input mmCIF files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="seqres_out",
        help="Directory to write deduplicated protein SEQRES fasta files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    structure_dir = Path(args.structure_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not structure_dir.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {structure_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = output_dir / "unique_protein_seqres"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    seq_to_members: dict[str, list[str]] = defaultdict(list)
    total_chain_count = 0

    pdbids: list[str] = []
    if args.list is not None:
        list_path = Path(args.list).expanduser().resolve()
        if not list_path.is_file():
            raise FileNotFoundError(f"List file not found: {list_path}")
        with list_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                pdbids.append(stripped.split()[0])
    else:
        pdbid = str(args.pdb).strip()
        if not pdbid:
            raise SystemExit("--pdb must not be empty")
        pdbids.append(pdbid)

    for pdbid in pdbids:
        cif_path = _find_structure_path(structure_dir, pdbid)
        chain_rows = _collect_chain_seqres(cif_path, pdbid)
        total_chain_count += len(chain_rows)
        for chain_name, sequence in chain_rows:
            seq_to_members[sequence].append(chain_name)

    duplicate_group_count = 0
    unique_sequence_count = len(seq_to_members)
    for group_index, (sequence, members) in enumerate(
        sorted(
            seq_to_members.items(),
            key=lambda item: (-len(item[1]), -len(item[0]), item[1][0]),
        ),
        start=1,
    ):
        if len(members) > 1:
            duplicate_group_count += 1
        representative = members[0]
        for member in members:
            print(
                "{:04d} {} {} {}".format(
                    group_index,
                    len(members),
                    len(sequence),
                    member,
                )
            )
        fasta_name = f"{group_index:04d}"
        fasta_path = fasta_dir / f"{fasta_name}.fa"
        fasta_path.write_text(f">{fasta_name}\n{sequence}\n", encoding="utf-8")

    print(
        "summary\tprotein_chains={}\tunique_seqres={}\tduplicate_groups={}\tchains_saved={}".format(
            total_chain_count,
            unique_sequence_count,
            duplicate_group_count,
            total_chain_count - unique_sequence_count,
        )
    )
    print(f"write\t{fasta_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
