"""Wrappers around usalign output parsing for template-guided workflows."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

from em3dfold.io.fileio import getlines


def read_usalign_transform(filename):
    lines = getlines(filename)
    rot = []
    trans = []
    for line in lines:
        if line[:1] in ["0", "1", "2"]:
            x0, x1, x2, x3 = [float(x) for x in line.strip().split()[1:5]]
            trans.append(x0)
            rot.append([x1, x2, x3])
    rot = np.asarray(rot, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    return rot, trans


def read_usalign_stdout(output):
    return output.split("\n")


def extract_alignment_lines(lines):
    marker_idx = -1
    for i, line in enumerate(lines):
        if "denotes residue pairs" in line and "denotes other aligned residues" in line:
            marker_idx = i
            break

    if marker_idx == -1:
        return None

    align_lines = []
    for line in lines[marker_idx + 1 :]:
        if not line.strip():
            continue
        if "Total CPU time" in line:
            break
        align_lines.append(line.rstrip("\n"))
        if len(align_lines) == 3:
            break

    if len(align_lines) != 3:
        return None
    return align_lines


def resolve_usalign_executable(lib_dir="./"):
    lib_dir = os.path.abspath(lib_dir)
    candidates = []
    if os.name == "nt":
        candidates.extend(
            [
                os.path.join(lib_dir, "bin", "USalign_windows.exe"),
                os.path.join(lib_dir, "bin", "usalign_windows.exe"),
                os.path.join(lib_dir, "bin", "USalign.exe"),
                os.path.join(lib_dir, "bin", "USalign"),
            ]
        )
    else:
        candidates.extend(
            [
                os.path.join(lib_dir, "bin", "USalign"),
                os.path.join(lib_dir, "bin", "USalign.exe"),
            ]
        )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def _build_windows_usalign_env():
    env = os.environ.copy()

    path_parts = []
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        path_parts.append(os.path.join(conda_prefix, "Library", "bin"))

    env_root = os.path.dirname(sys.executable)
    if env_root:
        path_parts.append(os.path.join(env_root, "Library", "bin"))

    existing_path = env.get("PATH", "")
    if existing_path:
        path_parts.append(existing_path)

    seen = set()
    merged = []
    for path in path_parts:
        norm = os.path.normcase(os.path.abspath(path))
        if os.path.isdir(path) and norm not in seen:
            seen.add(norm)
            merged.append(path)
    env["PATH"] = os.pathsep.join(merged)
    return env


def run_usalign(a, b, lib_dir="./", temp_dir=None, d=3.0, verbose=True, description=""):
    if temp_dir is None:
        with tempfile.TemporaryDirectory() as __temp_dir:
            return run_usalign(
                a,
                b,
                lib_dir=lib_dir,
                temp_dir=__temp_dir,
                d=d,
                verbose=verbose,
                description=description,
            )

    os.makedirs(temp_dir, exist_ok=True)
    matrix_file = os.path.join(temp_dir, f"{description}_MTX.txt")
    exe = resolve_usalign_executable(lib_dir)
    cmd = [
        exe,
        "-mol", "prot",
        "-mm", "0",
        a,
        b,
        "-d", f"{d:.2f}",
        "-m", matrix_file,
    ]
    if verbose:
        print("# Running command {}".format(" ".join(cmd)))
    run_kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if os.name == "nt":
        run_kwargs["env"] = _build_windows_usalign_env()
    result = subprocess.run(
        cmd,
        **run_kwargs,
    )
    if result.returncode == 0 and os.path.isfile(matrix_file):
        tm_output = read_usalign_stdout(result.stdout.decode("utf-8"))
        rot, trans = read_usalign_transform(matrix_file)
    else:
        tm_output, rot, trans = None, None, None
    return tm_output, rot, trans
