import os
import sys
import tempfile
import subprocess
import contextlib
import numpy as np

from em3dfold.io.fileio import getlines

def read_USalign_transform(filename):
    lines = getlines(filename)
    R = []
    t = []
    for line in lines:
        if line[:1] in ["0", "1", "2"]:
            x0, x1, x2, x3 = [float(x) for x in line.strip().split()[1:5]]
            t.append(x0)
            R.append([x1, x2, x3])
    R = np.asarray(R, dtype=np.float32) # (3, 3)
    t = np.asarray(t, dtype=np.float32) # (3,  )
    return R, t

def read_USalign_stdout(output):
    # output is a string of USalign's stdout
    """
    Name of Chain_1: AF.C.pdb (to be superimposed onto Chain_2)
    Name of Chain_2: denovo_chain_4.pdb
    Length of Chain_1: 55 residues
    Length of Chain_2: 44 residues
    
    Aligned length= 41, RMSD=   1.76, Seq_ID=n_identical/n_aligned= 0.902
    TM-score= 0.59349 (if normalized by length of Chain_1, i.e., LN=55, d0=2.44)
    TM-score= 0.69650 (if normalized by length of Chain_2, i.e., LN=44, d0=2.01)
    (You should use TM-score normalized by length of the reference structure)
    
    (":" denotes residue pairs of d <  5.0 Angstrom, "." denotes other aligned residues)
    -LCGGELVDTLQFVCGDRGFYFSR--PASRVSRRSRGIVEECCFRSCDLALLETYCAT
     :: :::::::::::::::::     ..        ::::::::::::::::::::
    LCG-GELVDTLQFVCGDRGFY---FSRP--------GIVEECCFRSCDLALLETYC--
    
    Total CPU time is  0.00 seconds
    """
    lines = output.split("\n")
    return lines

def extract_alignment_lines(lines):
    marker_idx = -1
    for i, line in enumerate(lines):
        if "denotes residue pairs" in line and "denotes other aligned residues" in line:
            marker_idx = i
            break

    if marker_idx == -1:
        return None

    align_lines = []
    for line in lines[marker_idx + 1:]:
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

def run_USalign(a, b, lib_dir='./', temp_dir=None, d=3.0, verbose=True, description=""):
    # align a onto b using USalign
    with tempfile.TemporaryDirectory() as __temp_dir:
        # make a temp dir
        if temp_dir is None:
            temp_dir = __temp_dir

        matrix_file = temp_dir + "/" + description + "_MTX.txt"
        cmd = [
            lib_dir + "/bin/USalign",
            "-mol", "prot",
            "-mm", "0",
            a,
            b,
            "-d", f"{d:.2f}",
            "-m", matrix_file,
        ]
        if verbose:
            print("# Running command {}".format(" ".join(cmd)))
        # run
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # parse result
        if result.returncode == 0:
            tm_output = result.stdout.decode('utf-8')
            tm_output = read_USalign_stdout(tm_output)
            R, t = read_USalign_transform(matrix_file)
        else:
            tm_output, R, t = None, None, None

    return tm_output, R, t



