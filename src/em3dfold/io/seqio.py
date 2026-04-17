from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np

from em3dfold.polymer_utils.residue_constants import (
    restype_3to1, index_to_restype_1,
    restype_1to3, index_to_restype_3,
)
from em3dfold.io.fileio import extract_lines_by_ca

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def read_lines(filename):
    lines = readlines(filename)
    return lines

def read_fasta(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs

def std_aa_seq(seq, outlier_filling='X'):
    seq0 = []
    for i in range(len(seq)):
        ch = seq[i].upper()
        if ch in index_to_restype_1[:20]:
            seq0.append(ch)
        else:
            seq0.append(outlier_filling)
    return "".join(seq0)

def write_lines_to_file(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line.strip())
            f.write("\n")

def filter_non_acgut(seq):
    new_seq = []
    for ch in seq:
        if ch in ["A", "C", "G", "U", "T"]:
            new_seq.append(ch)
    new_seq = "".join(new_seq)
    return new_seq

def write_lines_to_file(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line.strip("\n") + "\n")

def write_seq_to_file(seq, filename):
    lines = [">seq"]
    lines.append(seq.strip())
    write_lines_to_file(lines, filename)

def write_seqs_to_file(seqs, filename):
    lines = []
    for k, seq in enumerate(seqs):
        lines.append(f">Seq_{k}")
        lines.append(seq.strip())
    write_lines_to_file(lines, filename)

def read_secstr(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs

def toupper(seq):
    return seq.upper()

def tolower(seq):
    return seq.lower()

def is_na(seq):
    if len(seq) == 0:
        return False
    nucs = ['A', 'G', 'C', 'U', 'T', 'I', 'N', 'X', 'a', 'g', 'c', 'u', 't', 'i', 'n', 'x']
    for s in seq:
        if s not in nucs:
            return False
    return True

def is_dna(seq):
    return is_na(seq) and ('T' in seq or 't' in seq)

def is_rna(seq):
    return is_na(seq) and 'T' not in seq and 't' not in seq


def seq_identity(a, b):
    assert len(a) == len(b)
    s = 0
    for i in range(len(a)):
        if a[i] == '-' or b[i] == '-':
            continue
        if a[i] == b[i]:
            s += 1
    return s / len(a)


def get_sequence_from_pdb_lines(lines):
    seq = []
    for line in lines:
        if line.startswith("ATOM") and line[12:16] == " CA ":
            try:
                resname3 = line[17:20]
                resname1 = restype_3to1[resname3]
            except Exception:
                resname1 = "X"
            if resname1 not in index_to_restype_1[:20]:
                resname1 = "X"
            seq.append(resname1)
    return "".join(seq)


def update_sequence_to_pdb_lines(lines, seq):
    ca_lines = extract_lines_by_ca(lines)
    assert len(ca_lines) == len(seq)
    new_lines = []
    for i, line in enumerate(ca_lines):
        resname1 = seq[i]
        if resname1 not in index_to_restype_1[:20]:
            resname1 = "X"
        resname3 = restype_1to3[resname1]
        new_lines.append(line[:17] + "{:>3s}".format(resname3) + line[20:])
    return new_lines


def format_alignment(a, align, b):
    new_a = []
    new_align = []
    new_b = []
    for i, ch in enumerate(b):
        if ch != '-':
            new_a.append(a[i])
            new_align.append(align[i])
            new_b.append(ch)
    return "".join(new_a), "".join(new_align), "".join(new_b)


def _get_pairwise_aligner(seq_a, seq_b):
    aligner = PairwiseAligner()
    aligner.mode = "global"

    if is_na(seq_a) and is_na(seq_b):
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5
        score_fn = lambda x, y: 2.0 if x == y else -1.0
    else:
        matrix = substitution_matrices.load("BLOSUM62")
        aligner.substitution_matrix = matrix
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5

        def score_fn(x, y):
            try:
                return float(matrix[x, y])
            except Exception:
                return -4.0

    return aligner, score_fn


def _expand_alignment_strings(seq_a, seq_b, alignment):
    coords = np.asarray(alignment.coordinates, dtype=np.int32)
    seqA = []
    seqB = []

    for idx in range(coords.shape[1] - 1):
        a0, a1 = coords[0, idx], coords[0, idx + 1]
        b0, b1 = coords[1, idx], coords[1, idx + 1]
        da = a1 - a0
        db = b1 - b0

        if da > 0 and db > 0:
            step = min(da, db)
            seqA.append(seq_a[a0 : a0 + step])
            seqB.append(seq_b[b0 : b0 + step])
            a0 += step
            b0 += step
            da -= step
            db -= step

        if da > 0:
            seqA.append(seq_a[a0:a1])
            seqB.append("-" * da)

        if db > 0:
            seqA.append("-" * db)
            seqB.append(seq_b[b0:b1])

    return "".join(seqA), "".join(seqB)


def _build_alignment_markup(seqA, seqB, score_fn):
    markup = []
    for aa, bb in zip(seqA, seqB):
        if aa == "-" or bb == "-":
            markup.append(" ")
        elif aa == bb:
            markup.append(":")
        elif score_fn(aa, bb) > 0:
            markup.append(".")
        else:
            markup.append(" ")
    return "".join(markup)


def nwalign_fast(a, b, lib_dir="./", temp_dir=None, verbose=False, namea=None, nameb=None, fmt=False, debug=False):
    del lib_dir, temp_dir
    try:
        aligner, score_fn = _get_pairwise_aligner(a, b)
        alignments = aligner.align(a, b)
        if len(alignments) == 0:
            if debug:
                raise RuntimeError(
                    "No pairwise alignment returned for {} vs {} (lenA={}, lenB={})".format(
                        namea or "<seqA>",
                        nameb or "<seqB>",
                        len(a),
                        len(b),
                    )
                )
            return (None, None, None, 0.0, 0.0, 0.0, 0.0)

        if verbose:
            print("# Running in-process pairwise alignment")

        seqA, seqB = _expand_alignment_strings(a, b, alignments[0])
        align = _build_alignment_markup(seqA, seqB, score_fn)

        if fmt:
            seqA, align, seqB = format_alignment(seqA, align, seqB)

        seqid = 0
        seqcov = 0
        for i in range(len(seqA)):
            if seqA[i] != '-' and seqA[i] == seqB[i]:
                seqid += 1
            if seqA[i] != '-' and seqB[i] != '-':
                seqcov += 1
        denoma = len(a) if len(a) > 0 else 1e-8
        denomb = len(b) if len(b) > 0 else 1e-8
        seqAid = seqid / denoma
        seqAcov = seqcov / denoma
        seqBid = seqid / denomb
        seqBcov = seqcov / denomb
        return (seqA, align, seqB, seqAid, seqAcov, seqBid, seqBcov)
    except Exception as e:
        if debug:
            raise RuntimeError(
                "Pairwise alignment failed for {} vs {} (lenA={}, lenB={}, is_na_A={}, is_na_B={})".format(
                    namea or "<seqA>",
                    nameb or "<seqB>",
                    len(a),
                    len(b),
                    is_na(a),
                    is_na(b),
                )
            ) from e
        if verbose:
            print(
                "# WARNING pairwise alignment failed for {} vs {}: {}".format(
                    namea or "<seqA>",
                    nameb or "<seqB>",
                    e,
                )
            )
        return (None, None, None, 0.0, 0.0, 0.0, 0.0)


if __name__ == '__main__':
    pass
