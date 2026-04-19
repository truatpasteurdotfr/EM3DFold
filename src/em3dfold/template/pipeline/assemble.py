import os
import re
import time
import builtins
import argparse
import numpy as np

from em3dfold.io.pdbio import read_pdb, convert_to_chains, chains_atom_pos_to_pdb
from em3dfold.io.seqio import read_fasta, std_aa_seq, nwalign_fast

from em3dfold.utils.clash_utils import get_clash
from em3dfold.utils.cpsolver import solve
from em3dfold.utils.misc_utils import pjoin, abspath
from em3dfold.polymer_utils.residue_constants import index_to_restype_1
from em3dfold.utils.cryo_utils import read_mrc
from em3dfold.utils.grid_utils import grid_value_interp


def print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    message = sep.join(str(arg) for arg in args)
    if not message.startswith("# "):
        message = f"# {message}"
    builtins.print(message, **kwargs)

def split_matched_frag(input_string, tolerance=10):
    pattern = re.compile(r"[:]+(?: {1," + str(tolerance) + r"}[:]+)*")
    matches = pattern.finditer(input_string)
    matches = list(matches)
    #print(matches)
    return matches


def get_idxs_a2o(seq):
    idxs = []
    idx = 0
    for i in range(len(seq)):
        if seq[i] != '-':
            idxs.append(idx)
            idx += 1
        else:
            idxs.append(-1)
    idxs = np.asarray(idxs, dtype=np.int32)
    return idxs

def _append_fragment(
    isel,
    built_type,
    atom14_pos,
    atom14_mask,
    res_type,
    res_idx,
    new_chain_built_type,
    new_atom14_pos,
    new_atom14_mask,
    new_res_type,
    new_res_idx,
    new_chain_idx,
    new_n_chain,
    idxs=None,
):
    if idxs is None:
        idxs = np.arange(np.count_nonzero(isel), dtype=np.int32)
    idxs = np.asarray(idxs, dtype=np.int32)
    if len(idxs) == 0:
        return new_n_chain

    new_chain_built_type.append(built_type)
    new_atom14_pos.append(atom14_pos[isel][idxs])
    new_atom14_mask.append(atom14_mask[isel][idxs])
    new_res_type.append(res_type[isel][idxs])
    new_res_idx.append(res_idx[isel][idxs])
    new_chain_idx.append(np.asarray([new_n_chain] * len(idxs), dtype=np.int32))
    return new_n_chain + 1

def _is_protein_fragment(fragment_res_type):
    if len(fragment_res_type) == 0:
        return False
    fragment_res_type = np.asarray(fragment_res_type)
    return bool(np.all(fragment_res_type < 20))

def main(args):
    fseq = args.seq
    fpdbs = args.pdb
    if fpdbs is None:
        raise Exception("Please provide the fragments")
    lib_dir = abspath(args.lib)
    out_dir = abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    
    # read seqs
    seqs = read_fasta(fseq)
    seqs = [std_aa_seq(x) for x in seqs]
    for seq in seqs:
        print("Found seq")
        print(seq)

    # Read fragments/domains.
    # built_type: 0 for denovo, 1 for template-fit/template-guided.
    chain_built_type = []
    atom14_pos = []
    atom14_mask = []
    res_type = []
    res_idx = []
    chain_idx = []
    n_chain = 0
    for fpdb in fpdbs:
        print("Intend to read structure from {}".format(fpdb))
        try:
            __atom14_pos, __atom14_mask, __res_type, __res_idx, __chain_idx = read_pdb(fpdb, keep_valid=False)

            atom14_pos.append(__atom14_pos)
            atom14_mask.append(__atom14_mask)
            res_type.append(__res_type)
            res_idx.append(__res_idx)
           
            built_type = 1 if ("imp" in fpdb or "fix" in fpdb) else 0
            built_type = [built_type] * (__chain_idx.max() + 1)
            built_type = np.asarray(built_type, dtype=np.int32)

            chain_built_type.append(built_type)

            chain_idx.append(__chain_idx + n_chain)
            n_chain += (__chain_idx.max() + 1)
            print("Read {} chains from {}".format(__chain_idx.max() + 1, fpdb))
        except Exception as e:
            if getattr(args, "debug", False):
                raise
            print("Error occurs -> {}".format(e))
            print("Failed to read any structure from {}".format(fpdb))   

    # if no input structure
    if n_chain == 0 or len(atom14_pos) == 0:
        raise Exception("Cannot find any fragments in the input structures")

    chain_built_type = np.concatenate(chain_built_type, axis=0)

    atom14_pos = np.concatenate(atom14_pos, axis=0)
    atom14_mask = np.concatenate(atom14_mask, axis=0)
    res_type = np.concatenate(res_type, axis=0)
    res_idx = np.concatenate(res_idx, axis=0)
    chain_idx = np.concatenate(chain_idx, axis=0)
    print("Found {:d} input fragments".format(n_chain))


    if args.no_split:
        print("# No split")
    else:
        # align and split
        new_chain_built_type = []

        new_atom14_pos = []
        new_atom14_mask = []
        new_res_type = []
        new_res_idx = []
        new_chain_idx = []
        new_n_chain = 0
        n_max_res_gap_ratio = 0.01
        n_max_res_gap = 10
        n_min_res_frag = 10
        # query for the best match to determine the residue index
        for i in range(n_chain):
            # select current frag
            isel = chain_idx == i
            built_type = chain_built_type[i]
            frag_res_type = res_type[isel]

            if not _is_protein_fragment(frag_res_type):
                print(f"WARNING Frag {i} is non-protein, keep without sequence-based split")
                new_n_chain = _append_fragment(
                    isel, built_type, atom14_pos, atom14_mask, res_type, res_idx,
                    new_chain_built_type, new_atom14_pos, new_atom14_mask,
                    new_res_type, new_res_idx, new_chain_idx, new_n_chain,
                )
                continue

            pdb_seq = "".join([index_to_restype_1[x] for x in frag_res_type])
            if len(pdb_seq) == 0:
                print(f"WARNING Frag {i} has empty protein sequence, keep without sequence-based split")
                new_n_chain = _append_fragment(
                    isel, built_type, atom14_pos, atom14_mask, res_type, res_idx,
                    new_chain_built_type, new_atom14_pos, new_atom14_mask,
                    new_res_type, new_res_idx, new_chain_idx, new_n_chain,
                )
                continue
            # truncate coords according to input sequence by
            # removing the unmatched residues and splitting the matched residues into fragments if have large gaps
            best_seqid = -1.0
            best_k = -1
            best_align = None
            for k in range(len(seqs)):
                seq = seqs[k]
                align = nwalign_fast(pdb_seq, seq, lib_dir=lib_dir, debug=getattr(args, "debug", False))
                if align[0] is None or align[1] is None or align[2] is None:
                    print(f"WARNING Frag {i} failed to align to seq {k}")
                    continue
                _, _, _, seqid0, cov0, seqid1, cov1 = align

                if seqid0 > best_seqid:
                    best_seqid = seqid0
                    best_align = align
                    best_k = k

            if best_align is None:
                print(f"WARNING Frag {i} has no valid protein-sequence alignment, keep without sequence-based split")
                new_n_chain = _append_fragment(
                    isel, built_type, atom14_pos, atom14_mask, res_type, res_idx,
                    new_chain_built_type, new_atom14_pos, new_atom14_mask,
                    new_res_type, new_res_idx, new_chain_idx, new_n_chain,
                )
                continue

            print(f"Frag {i} best matches to seq {best_k}")

            # remove the unmatched residues and split
            matches = split_matched_frag(best_align[1], n_max_res_gap)
            if len(matches) == 0:
                print(f"WARNING Frag {i} has no matched protein sub-fragments, keep without sequence-based split")
                new_n_chain = _append_fragment(
                    isel, built_type, atom14_pos, atom14_mask, res_type, res_idx,
                    new_chain_built_type, new_atom14_pos, new_atom14_mask,
                    new_res_type, new_res_idx, new_chain_idx, new_n_chain,
                )
                continue
            #print(matches)
            idxs_a2o_pdb = get_idxs_a2o(best_align[0])
            idxs_a2o_seq = get_idxs_a2o(best_align[2])

            n_added = 0
            for match in matches:
                frag = match.group(0)
                # ignore too short frags
                if len(frag) < n_min_res_frag:
                    continue
                start_idx = match.start()
                end_idx = start_idx + len(frag)
                # select frags
                idxs0 = idxs_a2o_pdb[start_idx : end_idx]
                idxs1 = idxs_a2o_seq[start_idx : end_idx]

                idxs0 = idxs0[idxs0 != -1]
                if isinstance(idxs0, int):
                    idxs0 = np.asarray([idxs0], dtype=np.int32)
                idxs1 = idxs1[idxs1 != -1]
                if isinstance(idxs1, int):
                    idxs1 = np.asarray([idxs1], dtype=np.int32)


                # append
                new_n_chain = _append_fragment(
                    isel, built_type, atom14_pos, atom14_mask, res_type, res_idx,
                    new_chain_built_type, new_atom14_pos, new_atom14_mask,
                    new_res_type, new_res_idx, new_chain_idx, new_n_chain,
                    idxs=idxs0,
                )
                n_added += 1

            if n_added == 0:
                print(f"WARNING Frag {i} produced no retained protein sub-fragments, keep without sequence-based split")
                new_n_chain = _append_fragment(
                    isel, built_type, atom14_pos, atom14_mask, res_type, res_idx,
                    new_chain_built_type, new_atom14_pos, new_atom14_mask,
                    new_res_type, new_res_idx, new_chain_idx, new_n_chain,
                )
        
        # concatenate
        new_chain_built_type = np.asarray(new_chain_built_type, dtype=np.int32)
        new_atom14_pos = np.concatenate(new_atom14_pos, axis=0)
        new_atom14_mask = np.concatenate(new_atom14_mask, axis=0)
        new_res_type = np.concatenate(new_res_type, axis=0)
        new_res_idx = np.concatenate(new_res_idx, axis=0)
        new_chain_idx = np.concatenate(new_chain_idx, axis=0)
  
        chain_built_type = new_chain_built_type
        atom14_pos = new_atom14_pos
        atom14_mask = new_atom14_mask
        res_type = new_res_type
        res_idx = new_res_idx
        chain_idx = new_chain_idx
        n_chain = new_n_chain 

        print(f"Extract {new_n_chain} new frags") 


    # convert to individual fragments
    #chains_atom14_pos, \
    #chains_atom14_mask, \
    #chains_res_type, \
    #chains_res_idx = convert_to_chains(chain_idx, atom14_pos, atom14_mask, res_type, res_idx)
    #print("Found {:d} domains".format(len(chains_atom14_pos)))


    # detect clash
    # split to frags when have clash
    d_clash = 2.5
    frags = []
    frags_built_type = []
    for i in range(n_chain):
        isel = chain_idx == i
        isel_idxs = np.arange(len(atom14_pos), dtype=np.int32)[isel]
        frags.append(isel_idxs)
        frags_built_type.append(chain_built_type[i])
    print("Split to {} frags".format(len(frags)))


    # 0. reading ca map
    print("Reading map")
    fmap = args.map
    data, origin, vsize = read_mrc(fmap)
    print("Normalize map")
    dmax = np.percentile(data, q=99.9)
    data = np.clip(data, 0.0, dmax)
    data /= (dmax + 1e-6)

    data *= 10.0

    # 1. compute score for each frag
    # the score is the main-chain match score
    print("Compute the score for each frag")
    score = np.ones(len(frags), dtype=np.float32)
    for k, frag in enumerate(frags):
        ca_pos = atom14_pos[frag][..., 1, :]
        dens = grid_value_interp(ca_pos, data, origin=origin, vsize=vsize)
        score[k] = dens.sum()

    # 2. compute rst between each pair of frags
    # accelerated using cpp
    print("Compute the rst between frags")
    r_clash_cutoff = 0.10
    rst = np.zeros((len(frags), len(frags)), dtype=np.int32)
    ts = time.time()

    for i in range(len(frags)):
        for k in range(i + 1, len(frags)):
            i_clash, k_clash = get_clash(
                atom14_pos[frags[i]][..., 1, :], 
                np.ones(len(atom14_pos[frags[i]])), 
                atom14_pos[frags[k]][..., 1, :],
                np.ones(len(atom14_pos[frags[k]])), 
            )

            i_clash /= len(atom14_pos[frags[i]])
            k_clash /= len(atom14_pos[frags[k]])

            i_clash = i_clash.sum()
            k_clash = k_clash.sum()

            #print("{} {} clash {:.4f} {:.4f}".format(i, k, i_clash, k_clash))

            if i_clash > r_clash_cutoff  or k_clash > r_clash_cutoff:
                rst[i, k] = 1
                rst[k, i] = 1


    te = time.time()
    print("Done calculating frags collision matrix shape = {} time = {:.4f} s".format(rst.shape, te - ts))
    r = rst.sum() / (rst.shape[0] * rst.shape[1])
    print("Collision rate = {:.4f}".format(r))

    # solve
    #print(score)

    result, score, status = solve(score, rst, log_search_progress=False)
    print("Result is {}".format(result))
    print("Result has {} nodes".format(len(result)))

    # TODO
    # remove repeated residues and connect frags to chains (sort_chains)
    pass


    # Output denovo-built and template-fit parts independently.
    chains0_atom_pos = []
    chains0_atom_mask = []
    chains0_res_type = []
    chains0_res_idx = []
    chains1_atom_pos = []
    chains1_atom_mask = []
    chains1_res_type = []
    chains1_res_idx = []

    # output all fragment
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []
    #print(frags_built_type)

    for node in result:
        frag = frags[node]
        chains_atom_pos.append(atom14_pos[frag])
        chains_atom_mask.append(atom14_mask[frag])
        chains_res_type.append(res_type[frag])
        chains_res_idx.append(res_idx[frag])

        if frags_built_type[node] == 0:
            chains0_atom_pos.append(atom14_pos[frag])
            chains0_atom_mask.append(atom14_mask[frag])
            chains0_res_type.append(res_type[frag])
            chains0_res_idx.append(res_idx[frag])
        else:
            chains1_atom_pos.append(atom14_pos[frag])
            chains1_atom_mask.append(atom14_mask[frag])
            chains1_res_type.append(res_type[frag])
            chains1_res_idx.append(res_idx[frag])

    # type0
    fout = pjoin(out_dir, "assemble_denovo.cif")
    chains_atom_pos_to_pdb(
        fout,
        chains_atom_pos=chains0_atom_pos,
        chains_atom_mask=chains0_atom_mask,
        chains_res_type=chains0_res_type,
        chains_res_idx=chains0_res_idx,
        suffix=fout.split('.')[-1],
    )
    print("Write denovo part to {}".format(fout))

    # type1
    fout = pjoin(out_dir, "assemble_fit.cif")
    chains_atom_pos_to_pdb(
        fout,
        chains_atom_pos=chains1_atom_pos,
        chains_atom_mask=chains1_atom_mask,
        chains_res_type=chains1_res_type,
        chains_res_idx=chains1_res_idx,
        suffix=fout.split('.')[-1],
    )
    print("Write template-fit part to {}".format(fout))

    # all
    fout = pjoin(out_dir, "assemble.cif")
    chains_atom_pos_to_pdb(
        fout,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        suffix=fout.split('.')[-1],
    )
    print("Write assembled fragments to {}".format(fout))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", "-s", help="Input sequences")
    parser.add_argument("--pdb", "-p", nargs='+', help="Input structure domains/fragments")
    parser.add_argument("--map", "-m", help="Input CA map")
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to print log to stdout")
    parser.add_argument("--lib", "-l", help="Library directory", default=os.path.join(script_dir, ".."))
    parser.add_argument("--out", "-o", help="Output directory", default='./')
    #
    parser.add_argument("--no_split", help="No split frags", action='store_true')
    parser.add_argument("--debug", action="store_true", help="Raise exceptions directly")
    args = parser.parse_args()
    main(args)


