import argparse
import os
import shutil

import numpy as np

from em3dfold.io.fileio import writelines
from em3dfold.io.pdbio import chains_atom_pos_to_pdb, read_pdb
from em3dfold.io.seqio import read_fasta
from em3dfold.polymer_utils.residue_constants import index_to_restype_1
from em3dfold.utils.misc_utils import abspath, pjoin


def _protein_mask(res_type):
    res_type = np.asarray(res_type, dtype=np.int32)
    return res_type < 20


def _reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _prepare_output_dirs(out_dir):
    out_dir = abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    directories = {
        "root": out_dir,
        "chain": pjoin(out_dir, "chain_templs"),
        "complex": pjoin(out_dir, "complex_templs"),
        "templs": pjoin(out_dir, "templs"),
    }
    for path in directories.values():
        if path != out_dir:
            _reset_dir(path)
    return directories


def _write_formatted_sequence(seq_path, out_dir):
    output_path = pjoin(out_dir, "format_seq.fasta")
    if seq_path is None:
        print("WARNING you do not input sequence(s)")
        writelines(output_path, [">dummy_chain_0", "AAA"])
        return output_path

    seqs = read_fasta(seq_path)
    seq_lines = []
    for i, seq in enumerate(seqs):
        seq0 = "".join(residue for residue in seq if residue in index_to_restype_1[:20])
        seq_lines.append(f">chain_{i}")
        seq_lines.append(seq0)
    writelines(output_path, seq_lines)
    print(f"Write formatted seq to {output_path}")
    return output_path


def _write_single_template(output_path, atom_pos, atom_mask, res_type, bfactor):
    chains_atom_pos_to_pdb(
        filename=output_path,
        chains_atom_pos=[atom_pos],
        chains_atom_mask=[atom_mask],
        chains_res_type=[res_type],
        chains_res_idx=[np.arange(0, len(atom_pos), dtype=np.int32)],
        chains_bfactor=[bfactor],
        suffix="pdb",
    )


def _rewrite_complex_template(complex_path, output_dir, debug=False):
    if complex_path is None:
        return []

    print("User input 1 complex structure template")
    try:
        atom_pos, atom_mask, res_type, _res_idx, chain_idx, bfactor = read_pdb(
            complex_path,
            keep_valid=False,
            return_bfactor=True,
        )
        if len(chain_idx) == 0:
            raise ValueError("no atoms were parsed from the complex template")

        output_paths = []
        for chain_local_idx in range(int(chain_idx.max()) + 1):
            sel_mask = chain_idx == chain_local_idx
            prot_mask = _protein_mask(res_type[sel_mask])
            if not np.any(prot_mask):
                print(f"Skip complex chain {chain_local_idx} because it has no protein residues")
                continue

            output_path = pjoin(output_dir, f"templ_{chain_local_idx}.pdb")
            _write_single_template(
                output_path,
                atom_pos=atom_pos[sel_mask][prot_mask],
                atom_mask=atom_mask[sel_mask][prot_mask],
                res_type=res_type[sel_mask][prot_mask],
                bfactor=bfactor[sel_mask][prot_mask],
            )
            output_paths.append(output_path)
            print(f"Rewrite chain {chain_local_idx} from complex to {output_path}")
        return output_paths
    except Exception as exc:
        if debug:
            raise
        print(f"Error occurs -> {exc}")
        print("WARNING cannot rewrite complex template")
        if os.path.exists(complex_path):
            print("WARNING the template file exists, maybe no ATOM is recorded")
        else:
            print("WARNING the template file not exists")
        return []


def _rewrite_chain_templates(chain_paths, output_dir, debug=False):
    chain_paths = chain_paths or []
    if len(chain_paths) == 0:
        print("User input no single-chain structure templates")
        print("Using no structure templates")
        return []

    print(f"User input {len(chain_paths)} single-chain structure templates")
    print("Using single-chain structure as templates")

    output_paths = []
    for chain_template_index, chain_path in enumerate(chain_paths):
        try:
            atom_pos, atom_mask, res_type, _res_idx, _chain_idx, bfactor = read_pdb(
                chain_path,
                keep_valid=False,
                return_bfactor=True,
            )
            prot_mask = _protein_mask(res_type)
            if not np.any(prot_mask):
                print(f"WARNING template {chain_template_index} has no protein residues, skip it")
                continue

            output_path = pjoin(output_dir, f"templ_{chain_template_index}.pdb")
            _write_single_template(
                output_path,
                atom_pos=atom_pos[prot_mask],
                atom_mask=atom_mask[prot_mask],
                res_type=res_type[prot_mask],
                bfactor=bfactor[prot_mask],
            )
            output_paths.append(output_path)
            print(f"Rewrite template {chain_template_index} from single-chain to {output_path}")
        except Exception as exc:
            if debug:
                raise
            print(f"Error occurs -> {exc}")
            print(f"WARNING cannot rewrite template {chain_template_index}")
            if os.path.exists(chain_path):
                print("WARNING the template file exists, maybe no ATOM is recorded")
            else:
                print("WARNING the template file not exists")
    return output_paths


def _write_template_type(templs_out_dir, template_type):
    output_path = pjoin(templs_out_dir, "type.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"{template_type}\n")
    return output_path


def _select_active_templates(chain_output_paths, complex_output_paths, directories):
    has_chain_template = len(chain_output_paths) > 0
    has_complex_template = len(complex_output_paths) > 0

    print(f"Has chain   template = {has_chain_template}")
    print(f"has complex template = {has_complex_template}")

    if has_complex_template:
        print("Has complex template, use chain from complex as template")
        for src_path in complex_output_paths:
            shutil.copy2(src_path, pjoin(directories["templs"], os.path.basename(src_path)))
        template_type = "complex"
    elif has_chain_template:
        print("No complex template, use chain from single as template")
        for src_path in chain_output_paths:
            shutil.copy2(src_path, pjoin(directories["templs"], os.path.basename(src_path)))
        template_type = "chain"
    else:
        print("No template found")
        template_type = "none"

    _write_template_type(directories["templs"], template_type)
    return template_type


def _write_formatted_map(map_path, out_dir):
    if map_path is None:
        return None

    from em3dfold.utils.cryo_utils import parse_map, write_map

    data, origin, _nxyz, _vsize = parse_map(map_path, False, 1.0)
    output_path = pjoin(out_dir, "format_map.mrc")
    write_map(
        output_path,
        data,
        origin=origin,
        voxel_size=[1.0, 1.0, 1.0],
    )
    print(f"Write formated map to {output_path}")
    return output_path


def main(args):
    directories = _prepare_output_dirs(args.output)
    format_seq_path = _write_formatted_sequence(args.seq, directories["root"])
    complex_output_paths = _rewrite_complex_template(
        args.complex,
        directories["complex"],
        debug=getattr(args, "debug", False),
    )
    chain_output_paths = _rewrite_chain_templates(
        args.chain,
        directories["chain"],
        debug=getattr(args, "debug", False),
    )
    template_type = _select_active_templates(
        chain_output_paths=chain_output_paths,
        complex_output_paths=complex_output_paths,
        directories=directories,
    )
    format_map_path = _write_formatted_map(args.map, directories["root"])

    return {
        "output_dir": directories["root"],
        "format_seq_path": format_seq_path,
        "format_map_path": format_map_path,
        "template_type": template_type,
        "chain_template_paths": chain_output_paths,
        "complex_template_paths": complex_output_paths,
        "active_template_dir": directories["templs"],
    }


def add_args(parser):
    parser.add_argument("--seq", "-s", help="Input sequence")
    parser.add_argument("--chain", nargs="*", help="Input single chain structure predicted by AF2", required=False)
    parser.add_argument("--complex", help="Input complex structure predicted by AF3", required=False)
    parser.add_argument("--map", "-m", help="Input map")
    parser.add_argument("--output", "-o", help="Output directory of processed models", default="./")
    parser.add_argument("--debug", action="store_true", help="Raise exceptions directly")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(add_args(parser).parse_args())
