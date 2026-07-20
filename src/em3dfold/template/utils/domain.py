"""Legacy template-domain helpers backed by the main EM3DFold parser."""

from __future__ import annotations

import numpy as np

from em3dfold.pipeline.unidoc_domain import parse_unidoc_domains


def _normalize_domain_type(domain_type: str) -> str:
    domain_type = str(domain_type).strip().lower()
    if domain_type in {"unmerged", "fragment", "fragments", "small"}:
        return "fragment"
    return "large_domain"


def run_unidoc(
    pdb_dir,
    chain="A",
    lib_dir="./",
    temp_dir=None,
    verbose=True,
    domain_type="merged",
):
    del lib_dir, temp_dir, verbose
    result = parse_unidoc_domains(pdb_dir, chain_id=chain)[chain]
    key = _normalize_domain_type(domain_type)
    return str(result.get(key, "")).strip()


def parse_unidoc_result(line):
    if line is None:
        return []
    line = str(line).strip()
    if not line:
        return []

    result = []
    for domain in line.split("/"):
        domain = domain.strip()
        if not domain:
            continue
        pieces = []
        for sub_domain in domain.split(","):
            sub_domain = sub_domain.strip()
            if not sub_domain:
                continue
            start, end = [int(x) for x in sub_domain.split("~", 1)]
            pieces.append([start, end])
        if pieces:
            result.append(pieces)
    return result


def domains_to_unidoc_result(domains):
    if not domains:
        return ""
    return "/".join(
        ",".join(f"{int(start)}~{int(end)}" for start, end in domain)
        for domain in domains
    )


def convert_domains_to_1d_repr(domains):
    if not domains:
        return np.zeros((0,), dtype=np.int32)
    max_res_idx = max(int(end) for domain in domains for _start, end in domain)
    d = np.full((max_res_idx + 1,), -1, dtype=np.int32)
    for domain_idx, domain in enumerate(domains):
        for start, end in domain:
            d[int(start) : int(end) + 1] = domain_idx
    return d


def convert_1d_repr_to_domains(d):
    d = np.asarray(d, dtype=np.int32)
    if d.size == 0:
        return []
    domains = []
    for domain_idx in sorted(int(x) for x in np.unique(d) if int(x) >= 0):
        idxs = np.flatnonzero(d == domain_idx)
        if idxs.size == 0:
            continue
        boundaries = np.where(np.diff(idxs) > 1)[0]
        starts = np.concatenate(([0], boundaries + 1))
        stops = np.concatenate((boundaries, [len(idxs) - 1]))
        domain = []
        for start_pos, stop_pos in zip(starts, stops, strict=True):
            domain.append([int(idxs[start_pos]), int(idxs[stop_pos])])
        domains.append(domain)
    return domains


def annotate_pdb_with_domains(lines, domains):
    domain = convert_domains_to_1d_repr(domains)
    new_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            res_idx = int(line[22:26])
            domain_idx = domain[res_idx] if res_idx < len(domain) else -1
            new_lines.append(line[:60] + "{:>6.2f}".format(domain_idx) + line[66:])
    return new_lines


def annotate_pdb_cif_with_domains(lines, domains):
    del lines, domains
    raise NotImplementedError("Legacy CIF-domain annotation is no longer maintained.")


def split_pdb_with_domains(lines, domains):
    domain = convert_domains_to_1d_repr(domains)
    outputs = [[] for _ in domains]
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        res_idx = int(line[22:26])
        if res_idx >= len(domain):
            continue
        domain_idx = int(domain[res_idx])
        if domain_idx < 0:
            continue
        outputs[domain_idx].append(line)
    return outputs


def merge_intervals(dom):
    new_dom = []
    for current in sorted(dom, key=lambda x: x[0]):
        if not new_dom:
            new_dom.append(list(current))
            continue
        last = new_dom[-1]
        if last[1] + 1 >= current[0]:
            new_dom[-1] = [last[0], max(last[1], current[1])]
        else:
            new_dom.append(list(current))
    return new_dom


def merge_domains_simple(domains, n_min_res=50):
    doms = [[list(x) for x in domain] for domain in domains]
    n_iter = 0
    max_iter = 20
    while n_iter < max_iter:
        need_merge = False
        for i in range(len(doms)):
            if not doms[i]:
                continue
            length = sum(abs(end - start) for start, end in doms[i])
            ters = [value for interval in doms[i] for value in interval]
            if length < n_min_res:
                need_merge = True
                neighbors = []
                for k in range(len(doms)):
                    if k == i or not doms[k]:
                        continue
                    k_is_neighbor = any(
                        interval[0] - 1 in ters or interval[1] + 1 in ters
                        for interval in doms[k]
                    )
                    if k_is_neighbor:
                        neighbors.append(k)
                neighbors.sort(key=lambda x: len(doms[x]))
                if neighbors:
                    doms[i].extend(doms[neighbors[0]])
                    doms[neighbors[0]] = []
        if not need_merge:
            break
        n_iter += 1

    return [merge_intervals(dom) for dom in doms if dom]


def merge_domains_dmap(dmap, domains, n_min_res=50):
    del dmap, domains, n_min_res
    raise NotImplementedError
