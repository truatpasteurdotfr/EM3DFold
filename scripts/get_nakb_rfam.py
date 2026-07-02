#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request


SOLR_URL = "https://www.nakb.org/node/solr/nakb/select"
MISSING = "-"


def _parse_atlas_id(text: str) -> str:
    value = text.strip()
    if not value:
        raise ValueError("empty atlas id")
    if "nakb.org" not in value:
        return value.upper()

    parsed = urllib.parse.urlparse(value)
    tail = parsed.path.strip("/")
    if tail.startswith("atlas="):
        atlas_id = tail.split("=", 1)[1]
        if atlas_id:
            return atlas_id.upper()
    if parsed.query:
        query = urllib.parse.parse_qs(parsed.query)
        for key in ("atlas", "id", "q"):
            items = query.get(key)
            if items and items[0]:
                return items[0].upper()
    raise ValueError(f"cannot parse atlas id from URL: {text}")


def _fetch_doc(atlas_id: str) -> dict:
    params = urllib.parse.urlencode(
        {
            "q": f"allids:{atlas_id}",
            "wt": "json",
            "rows": 1,
        }
    )
    url = f"{SOLR_URL}?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    docs = payload.get("response", {}).get("docs", [])
    if not docs:
        raise RuntimeError(f"no naKB entry found for {atlas_id}")
    return docs[0]


def _safe_list(doc: dict, key: str) -> list:
    value = doc.get(key)
    if isinstance(value, list):
        return value
    return []


def _norm(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else MISSING


def _build_rows(doc: dict) -> list[dict]:
    entity_to_desc = {
        str(entity): desc
        for entity, desc in zip(
            _safe_list(doc, "polymer.entity_id"),
            _safe_list(doc, "polymer.description_nakb"),
        )
    }
    entity_to_type = {
        str(entity): poly_type
        for entity, poly_type in zip(
            _safe_list(doc, "polymer.entity_id"),
            _safe_list(doc, "polymer.type"),
        )
    }

    rows: list[dict] = []
    rfam_entities = _safe_list(doc, "rfamentity")
    rfam_ids = _safe_list(doc, "rfamids")
    rfam_descr = _safe_list(doc, "rfamdescr")
    rfam_chains = _safe_list(doc, "rfamchains")
    rfam_labels = _safe_list(doc, "rfamlabel")

    n = max(
        len(rfam_entities),
        len(rfam_ids),
        len(rfam_descr),
        len(rfam_chains),
        len(rfam_labels),
    )
    if n == 0:
        return [
            {
                "atlas_id": _norm(doc.get("id")),
                "pdb_id": _norm(doc.get("pdbid")),
                "entity_id": MISSING,
                "chain": MISSING,
                "polymer_type": MISSING,
                "description": MISSING,
                "rfam_id": MISSING,
                "rfam_label": MISSING,
                "rfam_description": MISSING,
            }
        ]

    for i in range(n):
        entity_id = str(rfam_entities[i]) if i < len(rfam_entities) else ""
        rows.append(
            {
                "atlas_id": _norm(doc.get("id")),
                "pdb_id": _norm(doc.get("pdbid")),
                "entity_id": _norm(entity_id),
                "chain": _norm(rfam_chains[i] if i < len(rfam_chains) else ""),
                "polymer_type": _norm(entity_to_type.get(entity_id, "")),
                "description": _norm(entity_to_desc.get(entity_id, "")),
                "rfam_id": _norm(rfam_ids[i] if i < len(rfam_ids) else ""),
                "rfam_label": _norm(rfam_labels[i] if i < len(rfam_labels) else ""),
                "rfam_description": _norm(rfam_descr[i] if i < len(rfam_descr) else ""),
            }
        )
    return rows


def _print_tsv(rows: list[dict]) -> None:
    fields = [
        "atlas_id",
        "pdb_id",
        "entity_id",
        "chain",
        "polymer_type",
        "description",
        "rfam_id",
        "rfam_label",
        "rfam_description",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join(_norm(row.get(field, MISSING)) for field in fields))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Rfam annotations for a naKB atlas entry."
    )
    parser.add_argument(
        "atlas",
        help="naKB atlas id like 9OWO or full URL like https://www.nakb.org/atlas=9OWO",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of TSV.",
    )
    args = parser.parse_args()

    atlas_id = _parse_atlas_id(args.atlas)
    doc = _fetch_doc(atlas_id)
    rows = _build_rows(doc)
    if args.json:
        json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        _print_tsv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
