#!/usr/bin/env python3
"""Download AlphaFold DB structures for each protein chain in an mmCIF file."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from download_afdb import (
    AFDB_ENTRY_URL,
    _download_file,
    _fallback_download_url,
    _fetch_prediction_metadata,
    _pick_download_url,
)
from get_entity_uniprot_ids import (
    UNIPROT_DB_NAMES,
    _ensure_list,
    _normalize_db_name,
    _normalize_token,
    _parse_mmcif_fields,
)


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _sanitize_chain_id(chain_id: str) -> str:
    chain_id = chain_id.strip()
    if not chain_id:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in chain_id)


def _entity_to_chain_ids(mmcif: dict) -> dict[str, list[str]]:
    entity_ids = _ensure_list(mmcif.get("_entity_poly.entity_id"))
    strand_ids = _ensure_list(mmcif.get("_entity_poly.pdbx_strand_id"))

    mapping: dict[str, list[str]] = {}
    row_count = max(len(entity_ids), len(strand_ids))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        strand_id_text = _normalize_token(strand_ids[idx] if idx < len(strand_ids) else None)
        if entity_id is None or strand_id_text is None:
            continue
        chain_ids = [
            chain_id.strip()
            for chain_id in strand_id_text.split(",")
            if chain_id.strip() and chain_id.strip() not in {"?", "."}
        ]
        if chain_ids:
            mapping[entity_id] = chain_ids
    return mapping


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


def _entity_to_uniprot_ids(mmcif: dict, protein_ids: set[str]) -> dict[str, list[str]]:
    entity_ids = _ensure_list(mmcif.get("_struct_ref.entity_id"))
    db_names = _ensure_list(mmcif.get("_struct_ref.db_name"))
    accessions = _ensure_list(mmcif.get("_struct_ref.pdbx_db_accession"))
    db_codes = _ensure_list(mmcif.get("_struct_ref.db_code"))

    mapping: dict[str, list[str]] = {}
    row_count = max(len(entity_ids), len(db_names), len(accessions), len(db_codes))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        db_name = _normalize_db_name(db_names[idx] if idx < len(db_names) else None)
        accession = _normalize_token(accessions[idx] if idx < len(accessions) else None)
        db_code = _normalize_token(db_codes[idx] if idx < len(db_codes) else None)
        if entity_id is None or entity_id not in protein_ids or db_name not in UNIPROT_DB_NAMES:
            continue
        uniprot_id = accession or db_code
        if uniprot_id is None:
            continue
        mapping.setdefault(entity_id, [])
        if uniprot_id not in mapping[entity_id]:
            mapping[entity_id].append(uniprot_id)
    return mapping


def _download_afdb_pdb(uniprot_id: str, output_path: Path) -> None:
    try:
        metadata = _fetch_prediction_metadata(uniprot_id)
        download_url = _pick_download_url(metadata, "pdb")
    except Exception as exc:
        _stderr(
            "Warning: failed to query AlphaFold DB API for {} ({}). Falling back to {}.".format(
                uniprot_id,
                exc,
                AFDB_ENTRY_URL.format(uniprot_id=uniprot_id),
            )
        )
        download_url = _fallback_download_url(uniprot_id, "pdb")
    _download_file(download_url, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download AlphaFold DB structures for each protein chain in an mmCIF file.",
    )
    parser.add_argument("cif", help="Input mmCIF file")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--pdbid",
        default=None,
        help="PDB ID prefix used in output filenames. Defaults to the input mmCIF stem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cif_path = Path(args.cif).expanduser().resolve()
    if not cif_path.is_file():
        raise FileNotFoundError(f"mmCIF file not found: {cif_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdbid = (args.pdbid or cif_path.stem).strip()
    if not pdbid:
        raise SystemExit("PDB ID prefix must not be empty")

    mmcif = _parse_mmcif_fields(cif_path)
    protein_ids = _protein_entity_ids(mmcif)
    entity_to_chains = _entity_to_chain_ids(mmcif)
    entity_to_uniprot = _entity_to_uniprot_ids(mmcif, protein_ids)

    downloaded = 0
    downloaded_by_uniprot: dict[str, Path] = {}
    for entity_id in sorted(protein_ids, key=lambda value: (len(value), value)):
        chain_ids = entity_to_chains.get(entity_id, [])
        uniprot_ids = entity_to_uniprot.get(entity_id, [])
        if not chain_ids:
            print(f"skip\tentity={entity_id}\treason=no_chain_ids")
            continue
        if not uniprot_ids:
            print(f"skip\tentity={entity_id}\treason=no_uniprot_id")
            continue

        uniprot_id = uniprot_ids[0]
        for chain_id in chain_ids:
            output_path = output_dir / f"{pdbid}.chain.{_sanitize_chain_id(chain_id)}.pdb"
            cached_path = downloaded_by_uniprot.get(uniprot_id)
            if cached_path is not None:
                try:
                    if cached_path.resolve() != output_path.resolve():
                        shutil.copyfile(cached_path, output_path)
                except OSError as exc:
                    print(
                        f"skip\tchain={chain_id}\tuniprot={uniprot_id}\treason=copy_failed\tmessage={exc}"
                    )
                    continue
                print(f"{chain_id}\t{uniprot_id}\t{output_path}\t(copy)")
                downloaded += 1
                continue

            try:
                _download_afdb_pdb(uniprot_id, output_path)
            except (HTTPError, URLError, OSError, RuntimeError) as exc:
                print(
                    f"skip\tchain={chain_id}\tuniprot={uniprot_id}\treason=download_failed\tmessage={exc}"
                )
                continue
            downloaded_by_uniprot[uniprot_id] = output_path
            print(f"{chain_id}\t{uniprot_id}\t{output_path}\t(download)")
            downloaded += 1

    if downloaded == 0:
        raise SystemExit(f"No AlphaFold DB structure was downloaded for {cif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
