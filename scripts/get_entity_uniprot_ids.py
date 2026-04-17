import argparse
import csv
from collections import defaultdict
from pathlib import Path


UNIPROT_DB_NAMES = {
    "UNP",
    "UNIPROT",
    "UNIPROTKB",
    "SP",
    "SWISS-PROT",
    "TREMBL",
}


def _tokenize_mmcif(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if line.startswith(";"):
            block_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                block_lines.append(lines[i])
                i += 1
            yield "\n".join(block_lines)
            if i < len(lines) and lines[i].startswith(";"):
                i += 1
            continue

        current = []
        in_quote = None
        j = 0
        while j < len(line):
            ch = line[j]
            if in_quote is not None:
                if ch == in_quote:
                    yield "".join(current)
                    current = []
                    in_quote = None
                else:
                    current.append(ch)
                j += 1
                continue

            if ch in {"'", '"'}:
                if current:
                    current.append(ch)
                else:
                    in_quote = ch
                j += 1
                continue

            if ch.isspace():
                if current:
                    yield "".join(current)
                    current = []
                j += 1
                continue

            current.append(ch)
            j += 1

        if current:
            yield "".join(current)
        i += 1


def _parse_mmcif_fields(cif_path):
    tokens = list(_tokenize_mmcif(Path(cif_path).read_text(encoding="utf-8")))
    fields = {}
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "loop_":
            i += 1
            tags = []
            while i < len(tokens) and tokens[i].startswith("_"):
                tags.append(tokens[i])
                i += 1

            if not tags:
                continue

            values = []
            while i < len(tokens):
                next_token = tokens[i]
                if next_token == "loop_" or next_token.startswith("data_") or next_token.startswith("save_"):
                    break
                if next_token.startswith("_") and len(values) % len(tags) == 0:
                    break
                values.append(next_token)
                i += 1

            usable_count = (len(values) // len(tags)) * len(tags)
            values = values[:usable_count]
            for col_idx, tag in enumerate(tags):
                tag_values = values[col_idx::len(tags)]
                if tag_values:
                    fields.setdefault(tag, []).extend(tag_values)
            continue

        if token.startswith("_"):
            if i + 1 < len(tokens):
                value = tokens[i + 1]
                existing = fields.get(token)
                if existing is None:
                    fields[token] = value
                elif isinstance(existing, list):
                    existing.append(value)
                else:
                    fields[token] = [existing, value]
                i += 2
                continue

        i += 1

    return fields


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_token(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in {"?", "."}:
        return None
    return value


def _normalize_db_name(value):
    token = _normalize_token(value)
    if token is None:
        return None
    return token.upper()


def _get_protein_entity_ids(mmcif):
    entity_ids = _ensure_list(mmcif.get("_entity_poly.entity_id"))
    poly_types = _ensure_list(mmcif.get("_entity_poly.type"))

    protein_entity_ids = set()
    row_count = max(len(entity_ids), len(poly_types))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        poly_type = _normalize_token(poly_types[idx] if idx < len(poly_types) else None)
        if entity_id is None or poly_type is None:
            continue
        if "polypeptide" in poly_type.lower():
            protein_entity_ids.add(entity_id)

    return protein_entity_ids


def extract_entity_uniprot_ids(cif_path):
    mmcif = _parse_mmcif_fields(cif_path)
    protein_entity_ids = _get_protein_entity_ids(mmcif)

    entity_ids = _ensure_list(mmcif.get("_struct_ref.entity_id"))
    db_names = _ensure_list(mmcif.get("_struct_ref.db_name"))
    accessions = _ensure_list(mmcif.get("_struct_ref.pdbx_db_accession"))
    db_codes = _ensure_list(mmcif.get("_struct_ref.db_code"))

    row_count = max(len(entity_ids), len(db_names), len(accessions), len(db_codes))
    entity_to_ids = defaultdict(list)

    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        db_name = _normalize_db_name(db_names[idx] if idx < len(db_names) else None)
        accession = _normalize_token(accessions[idx] if idx < len(accessions) else None)
        db_code = _normalize_token(db_codes[idx] if idx < len(db_codes) else None)

        if entity_id is None or entity_id not in protein_entity_ids or db_name not in UNIPROT_DB_NAMES:
            continue

        uniprot_id = accession or db_code
        if uniprot_id is None:
            continue

        if uniprot_id not in entity_to_ids[entity_id]:
            entity_to_ids[entity_id].append(uniprot_id)

    rows = []
    for entity_id in sorted(protein_entity_ids, key=lambda value: (len(value), value)):
        uniprot_ids = entity_to_ids.get(entity_id, [])
        if not uniprot_ids:
            rows.append({"entity_id": entity_id, "uniprotkb_id": ""})
            continue
        for uniprot_id in uniprot_ids:
            rows.append({"entity_id": entity_id, "uniprotkb_id": uniprot_id})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract UniProtKB accession IDs for protein entities from an mmCIF file."
    )
    parser.add_argument("cif", help="Input mmCIF file path")
    args = parser.parse_args()

    cif_path = Path(args.cif)
    if not cif_path.is_file():
        raise FileNotFoundError(f"mmCIF file not found: {cif_path}")

    rows = extract_entity_uniprot_ids(cif_path)
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=["entity_id", "uniprotkb_id"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
