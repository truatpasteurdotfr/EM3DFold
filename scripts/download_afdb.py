#!/usr/bin/env python3
"""Download an AlphaFold DB structure by UniProt accession."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve


AFDB_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
AFDB_ENTRY_URL = "https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
AFDB_FALLBACK_FILE_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.{suffix}"


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _fetch_prediction_metadata(uniprot_id: str) -> list[dict]:
    api_url = AFDB_API_URL.format(uniprot_id=uniprot_id)
    with urlopen(api_url) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No AlphaFold DB prediction metadata returned for {uniprot_id}")
    return data


def _pick_download_url(metadata: list[dict], file_format: str) -> str:
    key = f"{file_format}Url"
    for item in metadata:
        url = item.get(key)
        if isinstance(url, str) and url.strip():
            return url.strip()
    raise RuntimeError(f"Could not find {file_format} download URL in AlphaFold DB metadata")


def _fallback_download_url(uniprot_id: str, file_format: str) -> str:
    return AFDB_FALLBACK_FILE_URL.format(uniprot_id=uniprot_id, suffix=file_format)


def _resolve_output_path(output_dir: Path, download_url: str) -> Path:
    name = Path(urlparse(download_url).path).name
    if not name:
        raise RuntimeError(f"Could not derive output filename from URL: {download_url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def _download_file(download_url: str, output_path: Path) -> None:
    urlretrieve(download_url, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download an AlphaFold DB structure by UniProt accession.",
    )
    parser.add_argument(
        "uniprot_ids",
        nargs="+",
        help="One or more UniProt accessions, e.g. P29972 Q9BZE2",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to save the downloaded structure",
    )
    parser.add_argument(
        "--format",
        choices=("cif", "pdb"),
        default="cif",
        help="Coordinate file format to download",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    for raw_uniprot_id in args.uniprot_ids:
        uniprot_id = raw_uniprot_id.strip()
        if not uniprot_id:
            raise SystemExit("UniProt ID must not be empty")

        download_url = None
        try:
            metadata = _fetch_prediction_metadata(uniprot_id)
            download_url = _pick_download_url(metadata, args.format)
        except (HTTPError, URLError, json.JSONDecodeError, RuntimeError) as exc:
            _stderr(
                "Warning: failed to query AlphaFold DB API for {} ({}). "
                "Falling back to the default file URL pattern inferred from {}.".format(
                    uniprot_id,
                    exc,
                    AFDB_ENTRY_URL.format(uniprot_id=uniprot_id),
                )
            )
            download_url = _fallback_download_url(uniprot_id, args.format)

        output_path = _resolve_output_path(output_dir, download_url)

        try:
            _download_file(download_url, output_path)
        except HTTPError as exc:
            raise SystemExit(
                "Failed to download AlphaFold DB structure for {} from {}: HTTP {}".format(
                    uniprot_id,
                    download_url,
                    exc.code,
                )
            ) from exc
        except URLError as exc:
            raise SystemExit(
                "Failed to download AlphaFold DB structure for {} from {}: {}".format(
                    uniprot_id,
                    download_url,
                    exc,
                )
            ) from exc

        print(f"AlphaFold DB entry: {AFDB_ENTRY_URL.format(uniprot_id=uniprot_id)}")
        print(f"Downloaded: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
