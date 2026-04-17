#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${ROOT_DIR}/em3dfold/bin"
SRC_DIR="${BIN_DIR}/src"
NPROC="${NPROC:-4}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

run_step() {
    local msg="$1"
    shift
    "$@" || fail "$msg"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        fail "required command not found: $cmd"
    fi
}

check_gxx_version() {
    local ver major minor patch
    ver="$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)"

    IFS=. read -r major minor patch <<< "$ver"
    patch="${patch:-0}"

    if (( major > 4 )) || (( major == 4 && minor > 8 )) || (( major == 4 && minor == 8 && patch >= 5 )); then
        echo "OK: g++ >= 4.8.5 (current: $ver)"
    else
        fail "g++ < 4.8.5 (current: $ver)"
    fi
}

copy_binary() {
    local src="$1"
    local dst="$2"
    cp "$src" "$dst"
    chmod +x "$dst"
    echo "Installed $(basename "$dst") -> $dst"
}

build_getp() {
    echo "Compiling getp"
    pushd "${SRC_DIR}/getp" >/dev/null
    make clean || true
    run_step "Failed to compile getp" make -j"${NPROC}" all
    copy_binary "${SRC_DIR}/getp/getp" "${BIN_DIR}/getp"
    popd >/dev/null
}

build_stride() {
    echo "Compiling stride"
    pushd "${SRC_DIR}/stride/src" >/dev/null
    make clean || true
    run_step "Failed to compile stride" make
    copy_binary "${SRC_DIR}/stride/src/stride" "${BIN_DIR}/stride"
    popd >/dev/null
}

build_unidoc() {
    echo "Compiling unidoc_frag"
    pushd "${SRC_DIR}/unidoc/src" >/dev/null
    rm -f unidoc_frag
    run_step "Failed to compile unidoc_frag" \
        g++ -std=c++0x -O2 -ffast-math -o unidoc_frag UniDoc_struct.cpp -lm
    copy_binary "${SRC_DIR}/unidoc/src/unidoc_frag" "${BIN_DIR}/unidoc_frag"
    popd >/dev/null
}

build_usalign_suite() {
    echo "Compiling USalign"
    pushd "${SRC_DIR}/usalign" >/dev/null
    make clean || true
    run_step "Failed to compile USalign" make -j"${NPROC}" USalign
    copy_binary "${SRC_DIR}/usalign/USalign" "${BIN_DIR}/USalign"
    popd >/dev/null
}

main() {
    require_cmd make
    require_cmd gcc
    require_cmd g++
    check_gxx_version

    mkdir -p "${BIN_DIR}"

    build_getp
    build_stride
    build_unidoc
    build_usalign_suite

    echo
    echo "EM3DFold binaries have been compiled successfully."
    echo "Installed programs:"
    echo "  ${BIN_DIR}/getp"
    echo "  ${BIN_DIR}/stride"
    echo "  ${BIN_DIR}/unidoc_frag"
    echo "  ${BIN_DIR}/USalign"
}

main "$@"
