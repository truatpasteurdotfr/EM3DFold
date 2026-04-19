import os
import re
import time
import builtins
import argparse
import numpy as np

from scipy.spatial import KDTree
from collections import deque

from em3dfold.io.fileio import (
    writelines,
    extract_lines_by_ca,
)

from em3dfold.io.seqio import (
    seq_identity,
    get_sequence_from_pdb_lines,
)

from em3dfold.utils.misc_utils import (
    abspath,
    pjoin,
    find_first_not_of,
    find_last_not_of,
)

from em3dfold.utils.tm_utils import (
    run_usalign,
    extract_alignment_lines,
)

from em3dfold.template.pipeline.template_refine import (
    build_template_refine_context,
    bundle_from_models,
    ChainStructureBundle,
    FixPassResult,
    load_structure_models,
    write_fix_outputs,
)

from em3dfold.utils.geometry import (
    apply,
    kabsch,
)

def print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    message = sep.join(str(arg) for arg in args)
    if not message.startswith("# "):
        message = f"# {message}"
    builtins.print(message, **kwargs)

def find_matched_frag(seq1, align, seq2, tolerance=2, min_sub_num=5, min_score=0.80, min_seq_id=0.80, verbose=False):
    assert tolerance >= 1
    assert min_sub_num >= 1
    assert min_score >= 0.0
    assert min_seq_id >= 0.0
    assert len(seq1) == len(align) == len(seq2)

    # find fragments and score each fragment
    data = seq2
    pattern = re.compile(r"[^-]+(?:-{1," + str(tolerance) + r"}[^-]+)*")
    matches = list(re.finditer(pattern, data))
    # filter
    matches = [match for match in matches if len(match.group(0)) >= min_sub_num]
   
    seqids = [] 
    scores = []
    for match in matches:
        substr = match.group(0)
        idx = match.start()

        n = 0
        seqid = 0
        score = 0.0
        for k in range(idx, idx + len(substr)):
            if align[k] == ':':
                score += 1.0
            elif align[k] == '.':
                score += 0.2
            if seq1[k] != '-' and seq2[k] != '-':
                n += 1
                if seq1[k] == seq2[k]:
                    seqid += 1
        if n == 0:
            seqid = 0.0
        else:
            seqid = seqid / n
        score = score / len(substr)
        scores.append(score)
        seqids.append(seqid)

        if verbose:
            print(f"sub-frag '{match.group(0)}' start position {match.start()} len {len(substr):d} seqid {seqid:.4f} score {score:.4f}")

    # merge well-matched fragments
    idxs = np.argsort(scores, kind='stable')
    idxs = idxs[::-1]
    idxs = [idx for idx in idxs if scores[idx] >= min_score and seqids[idx] >= min_seq_id]

    # find left fragments
    left_idxs = [i for i in range(len(matches)) if i not in idxs]
    #print(left_idxs)

    # return all fragments
    return [
        (matches[idx].group(0), matches[idx].start(), seqids[idx], scores[idx]) for idx in idxs
    ], [
        (matches[idx].group(0), matches[idx].start(), seqids[idx], scores[idx]) for idx in left_idxs
    ]


def idx_aligned_to_original(seq):
    n = 0
    idxs = np.full(len(seq), -1, dtype=np.int32)
    for i, s in enumerate(seq):
        if s != '-':
            idxs[i] = n
            n += 1
    return idxs

def run_fix_pass(args, context) -> FixPassResult:
    out_dir = abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    verbose = args.verbose
    temp_dir = pjoin(out_dir, "alignments")
    os.makedirs(temp_dir, exist_ok=True)
    lib_dir = context.lib_dir
    fchains = [model.path for model in context.chains]
    ftempls = [model.path for model in context.templates]
    print(f"Found {len(context.chains)} chains")
    print(f"Found {len(context.templates)} templates")

    for i, chain_model in enumerate(context.chains):
        if len(chain_model.align_seq) == 0:
            print("WARNING skip chain {} because it has no protein CA sequence".format(chain_model.path))
            continue

        match = context.best_matches.get(i)
        if match is None or match.template_index < 0:
            print("WARNING no valid protein template fit for chain {}".format(chain_model.path))
            continue

        template_model = context.templates[match.template_index]
        if len(template_model.align_seq) == 0:
            print("WARNING skip template {} because it has no protein CA sequence".format(template_model.path))
            continue

        print("Best sequence fit for chain {} is {}".format(chain_model.path, template_model.path))

    fix_chains_atom_pos = []
    fix_chains_atom_mask = []
    fix_chains_res_type = []
    fix_chains_res_idx = []

    max_layer = 20
    for cidx in range(len(fchains)):
            print("-"*80)
            print(f"Start searching for chain {cidx}")
            print("-"*80)

            # find best fit template
            match = context.best_matches.get(cidx)
            tidx = -1 if match is None else match.template_index
            if tidx is None or tidx < 0:
                print("Skip chain {} because no valid protein template fit is available".format(cidx))
                continue

            template_domain_info = context.template_domains[tidx]
            templ_domains = template_domain_info.domains
            templ_domains_1d = template_domain_info.domains_1d

            template_model = context.templates[tidx]
            chain_model = context.chains[cidx]
            templ_atom_pos = template_model.atom_pos
            templ_atom_mask = template_model.atom_mask
            templ_res_type = template_model.res_type
            templ_res_idx = template_model.res_idx

            # iterative alignment
            # init lines
            templ_lines0 = template_model.raw_lines
            ca_templ_lines0 = template_model.ca_lines
            chain_lines0 = chain_model.raw_lines
            ca_chain_lines0 = chain_model.ca_lines
            if len(ca_chain_lines0) == 0:
                print("Skip chain {} because it has no CA atoms".format(cidx))
                continue
            if len(ca_templ_lines0) == 0:
                print("Skip template {} because it has no CA atoms".format(tidx))
                continue
            q = deque()
            q.append((ca_templ_lines0, ca_chain_lines0))

            layer = 0
            while q:
                N = len(q)
                for n in range(N):
                    if args.verbose:
                        print("Start search at layer {} node {}".format(layer, n))

                    # pop element in queue
                    e = q.pop()
                    templ_lines = e[0]
                    chain_lines = e[1]

                    # dump chains to files
                    templ_prefix = f"templ_l_{layer}_n_{n}"
                    chain_prefix = f"chain_l_{layer}_n_{n}"
                    ftempl = pjoin(temp_dir, templ_prefix + ".pdb")
                    fchain = pjoin(temp_dir, chain_prefix + ".pdb")
                    writelines(ftempl, templ_lines)
                    writelines(fchain, chain_lines)

                    # run initial usalign
                    result, R, t = run_usalign(ftempl, fchain, lib_dir=lib_dir, d=2.0, description=templ_prefix + "_onto_" + chain_prefix, temp_dir=temp_dir, verbose=verbose)
                    # check if we have run usalign succesfully
                    # if not continue on next node
                    if result is None:
                        print("No usalign result for {} and {}".format(templ_prefix, chain_prefix))
                        continue

                    align_result = extract_alignment_lines(result)
                    if align_result is None:
                        print("Unable to parse alignment text from usalign output for {} and {}".format(templ_prefix, chain_prefix))
                        continue
                    #print(align_result)

                    # save idxs
                    idxs0_a2o = np.asarray(idx_aligned_to_original(align_result[0]), dtype=np.int32)
                    idxs1_a2o = np.asarray(idx_aligned_to_original(align_result[2]), dtype=np.int32)

                    # score each fragments
                    frags_good, frags_bad = find_matched_frag(align_result[0], align_result[1], align_result[2], verbose=args.verbose)
                    # if no good fragments, continue on next node
                    if not frags_good:
                        print(f"No good fragments on layer {layer} node {n}")
                        continue

                    matched_template_pos = []
                    for frag in frags_good:
                        length = len(frag[0])
                        start_idx = frag[1]
                        idxs0 = idxs0_a2o[start_idx : start_idx + length]

                        # find start and end for structure 0
                        s = find_first_not_of(idxs0, -1)
                        e = find_last_not_of(idxs0, -1)
                        if s == -1 or e == -1:
                            continue
                        matched_template_pos.extend([int(x) for x in idxs0[s : e + 1] if int(x) != -1])
                        
                    # get the matched minimum "domain"
                    select_doms = []
                    if matched_template_pos:
                        matched_template_pos = np.asarray(matched_template_pos, dtype=np.int32)
                        matched_res_idx = templ_res_idx[matched_template_pos]
                        valid_mask = np.logical_and(matched_res_idx >= 0, matched_res_idx < len(templ_domains_1d))
                        matched_domain_ids = templ_domains_1d[matched_res_idx[valid_mask]]
                        matched_domain_ids = matched_domain_ids[matched_domain_ids >= 0]
                        doms, counts = np.unique(matched_domain_ids, return_counts=True)
                        for dom, count in zip(doms, counts):
                            if count / max(len(matched_domain_ids), 1) > 0.10:
                                select_doms.append(int(dom))
                    print("Select domains {}".format(select_doms))


                    # Refine the transform directly from the matched CA pairs
                    # instead of spawning a second usalign process.
                    sel_idxs = []
                    for frag in frags_good:
                        length = len(frag[0])
                        start_idx = frag[1]
                        sel_idxs.extend(list(range(start_idx, start_idx + length)))
                    origial_idxs0 = [idxs0_a2o[x] for x in sel_idxs]
                    origial_idxs1 = [idxs1_a2o[x] for x in sel_idxs]
                    # should not have -1 idx
                    sel_idxs = [i for i in range(len(sel_idxs)) if origial_idxs0[i] != -1 and origial_idxs1[i] != -1]
                    original_idxs0 = [origial_idxs0[i] for i in sel_idxs]
                    original_idxs1 = [origial_idxs1[i] for i in sel_idxs]
                    sel_templ_lines = [templ_lines[x] for x in original_idxs0]
                    sel_chain_lines = [chain_lines[x] for x in original_idxs1]
                    sel_ca_templ_lines = extract_lines_by_ca(sel_templ_lines)
                    sel_ca_chain_lines = extract_lines_by_ca(sel_chain_lines)
                    sel_templ_seq = get_sequence_from_pdb_lines(sel_ca_templ_lines)
                    sel_chain_seq = get_sequence_from_pdb_lines(sel_ca_chain_lines)

                    seq_id = seq_identity(sel_templ_seq, sel_chain_seq)

                    if args.verbose:
                        print("Templ seq {}".format(sel_templ_seq))
                        print("Chain seq {}".format(sel_chain_seq))
                        print("Seq id {:.4f}".format(seq_id))

                    if len(original_idxs0) >= 3:
                        sel_templ_ca = np.asarray(
                            [[float(line[k:k+8]) for k in [30, 38, 46]] for line in sel_ca_templ_lines],
                            dtype=np.float32,
                        )
                        sel_chain_ca = np.asarray(
                            [[float(line[k:k+8]) for k in [30, 38, 46]] for line in sel_ca_chain_lines],
                            dtype=np.float32,
                        )
                        if len(sel_templ_ca) == len(sel_chain_ca) and len(sel_templ_ca) >= 3:
                            print("Refine transform with matched CA Kabsch")
                            R, t = kabsch(sel_templ_ca, sel_chain_ca)

                    # save the corresponding domain structure
                    if select_doms:
                        # 2025-04-20 reorder domain
                        select_templ_mask = np.zeros_like(templ_res_type).astype(bool)
                        templ_domain_ids = np.full(len(templ_res_idx), -1, dtype=np.int32)
                        valid_res_mask = np.logical_and(templ_res_idx >= 0, templ_res_idx < len(templ_domains_1d))
                        templ_domain_ids[valid_res_mask] = templ_domains_1d[templ_res_idx[valid_res_mask]]

                        for dom in select_doms:
                            mask = templ_domain_ids == dom
                            select_templ_mask[mask] = True

                        select_atom_pos = templ_atom_pos[select_templ_mask]
                        select_atom_mask = templ_atom_mask[select_templ_mask]
                        select_res_type = templ_res_type[select_templ_mask]
                        select_res_idx = templ_res_idx[select_templ_mask]

                        #print(select_res_idx)
                        #exit()

                        # Previous
                        #select_atom_pos = []
                        #select_atom_mask = []
                        #select_res_type = []
                        #select_res_idx = []
                        #for dom in select_doms:
                        #    mask = templ_domains_1d == dom
                        #    select_atom_pos.append(templ_atom_pos[mask])
                        #    select_atom_mask.append(templ_atom_mask[mask])
                        #    select_res_type.append(templ_res_type[mask])
                        #    select_res_idx.append(templ_res_idx[mask])
                        #select_atom_pos = np.concatenate(select_atom_pos, axis=0)
                        #select_atom_mask = np.concatenate(select_atom_mask, axis=0)
                        #select_res_type = np.concatenate(select_res_type, axis=0)
                        #select_res_idx = np.concatenate(select_res_idx, axis=0)

                        # apply
                        if len(select_atom_pos) > 0:
                            select_atom_pos = apply(select_atom_pos, R, t)
                            fix_chains_atom_pos.append(select_atom_pos)
                            fix_chains_atom_mask.append(select_atom_mask)
                            fix_chains_res_type.append(select_res_type)
                            fix_chains_res_idx.append(select_res_idx)

                    # exclude the matched frags and put into queue
                    #print("Bad fragments")
                    bad_idxs = []
                    for frag in frags_bad:
                        length = len(frag[0])
                        start_idx = frag[1]
                        idxs1 = idxs1_a2o[start_idx : start_idx + length]
                        bad_idxs.extend(idxs1)
                    bad_idxs = [idx for idx in bad_idxs if idx != -1]

                    # usalign requires at least >= 3 residues
                    # we further tighten this restraint
                    if len(bad_idxs) >= 5:
                        bad_lines = [chain_lines[x] for x in bad_idxs]
                        q.append((templ_lines, bad_lines))

                # add up layer
                layer += 1

                # to avoid to stuck in dead loop
                # nearly impossible to happen
                if layer > max_layer:
                    print("WARNING Too deep layer end search now")
                    break

            print("-"*80)
            print(f"End searching for chain {cidx}")
            print("-"*80)

    if not len(fix_chains_atom_pos) > 0:
        raise Exception("Unable to fix any chains by fix")

    templ_bundle = ChainStructureBundle(
        atom_pos=[coords.copy() for coords in fix_chains_atom_pos],
        atom_mask=[mask.copy() for mask in fix_chains_atom_mask],
        res_type=[rtype.copy() for rtype in fix_chains_res_type],
        res_idx=[ridx.copy() for ridx in fix_chains_res_idx],
    )

    chain_ca_pos = []
    for chain_model in context.chains:
        for line in chain_model.ca_lines:
            chain_ca_pos.append([float(line[k:k+8]) for k in [30, 38, 46]])
    chain_ca_pos = np.asarray(chain_ca_pos)
    print("Total {} denovo coords".format(len(chain_ca_pos)))
    d0 = 1.0

    visited = set()
    tree = KDTree(chain_ca_pos)
    for i in range(len(fix_chains_atom_pos)):
        idxs = tree.query_ball_point(fix_chains_atom_pos[i][..., 1, :], d0)
        for k, idx in enumerate(idxs):
            if len(idx) == 0 or len(idx) >= 2:
                continue
            if idx[0] in visited:
                continue
            visited.add(idx[0])
            v = chain_ca_pos[idx[0]] - fix_chains_atom_pos[i][k][1]
            fix_chains_atom_pos[i][k] += v

    return FixPassResult(
        templ_bundle=templ_bundle,
        chain_templ_bundle=ChainStructureBundle(
            atom_pos=fix_chains_atom_pos,
            atom_mask=fix_chains_atom_mask,
            res_type=fix_chains_res_type,
            res_idx=fix_chains_res_idx,
        ),
    )


def run_with_context(args, context):
    ts = time.time()
    out_dir = abspath(args.output)
    try:
        result = run_fix_pass(args, context)
        write_fix_outputs(
            out_dir=out_dir,
            templ_bundle=result.templ_bundle,
            chain_templ_bundle=None,
            fallback=False,
        )
        write_fix_outputs(
            out_dir=out_dir,
            templ_bundle=None,
            chain_templ_bundle=result.chain_templ_bundle,
            fallback=False,
        )
    except Exception as e:
        if getattr(args, "debug", False):
            raise
        print("Error occurs -> {}".format(e))
        print("WARNING cannot fix chains by templates")
        print("WARNING will write denovo built chains instead")
        chain_models = context.chains if context is not None else load_structure_models(args.chain or [])
        fallback_bundle = bundle_from_models(chain_models)
        write_fix_outputs(
            out_dir=out_dir,
            templ_bundle=fallback_bundle,
            chain_templ_bundle=fallback_bundle,
            fallback=True,
        )


    te = time.time()
    #print("Time consuming = {:.4f}".format(te - ts))


def main(args):
    temp_context_dir = pjoin(abspath(args.output), "alignments")
    context = build_template_refine_context(
        chain_paths=args.chain or [],
        template_paths=args.template or [],
        lib_dir=args.lib,
        work_dir=temp_context_dir,
        seq_path=args.seq,
        verbose=args.verbose,
        debug=getattr(args, "debug", False),
        prepare_domains_flag=True,
    )
    return run_with_context(args, context)


if __name__ == '__main__':
    script_dir = abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", "-s", help="Input sequence")
    parser.add_argument("--chain", "-c", nargs='+', help="Denovo built chains")
    parser.add_argument("--template", "-t", type=str, nargs='+', help="Input template, could be models predicted by AlphaFold/ESMFold")
    parser.add_argument("--lib", "-l", help="Lib directory", default=pjoin(script_dir, ".."))
    parser.add_argument("--output", "-o", help="Output directory", default="./")
    parser.add_argument("--debug", action="store_true", help="Raise exceptions directly")
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to print log to stdout")
    args = parser.parse_args()
    main(args)


