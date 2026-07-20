import contextlib

import torch
import torch.nn as nn

from em3dfold.models.modules import SinusoidalPositionalEncoding
from em3dfold.models.v2.model_refresh import Output
from em3dfold.models.v3x2.model import Model as V3X2Model


class Model(V3X2Model):
    def __init__(
        self,
        *args,
        pred_na_type=False,
        use_edge_pairing_input=True,
        use_index_pos=True,
        node_pos_enc_dim=16,
        edge_pos_enc_dim=16,
        max_rel_res_offset=64,
        residue_pos_scale=32.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pred_na_type = bool(pred_na_type)
        self.use_edge_pairing_input = bool(use_edge_pairing_input)
        self.use_index_pos = bool(use_index_pos)
        self.node_pos_enc_dim = int(node_pos_enc_dim)
        self.edge_pos_enc_dim = int(edge_pos_enc_dim)
        self.max_rel_res_offset = int(max_rel_res_offset)
        self.residue_pos_scale = float(residue_pos_scale)

        if self.pred_na_type:
            self.na_type_predictor = nn.Sequential(
                nn.Linear(self.d_node, self.d_node),
                nn.ReLU(),
                nn.Linear(self.d_node, self.d_node),
                nn.ReLU(),
                nn.Linear(self.d_node, 2),
            )

        if self.use_index_pos:
            self.node_residue_pos_enc = SinusoidalPositionalEncoding(self.node_pos_enc_dim)
            self.node_chain_pos_enc = SinusoidalPositionalEncoding(self.node_pos_enc_dim)
            self.edge_relative_pos_enc = SinusoidalPositionalEncoding(self.edge_pos_enc_dim)
            self.node_index_embed = nn.Linear(self.node_pos_enc_dim * 2 + 1, self.d_node, bias=False)
            self.edge_index_embed = nn.Linear(self.edge_pos_enc_dim + 3, self.d_edge, bias=False)

        if self.use_edge_pairing_input:
            self.edge_pairing_embed = nn.Linear(2, self.d_edge, bias=False)

    def _remap_chain_index(self, chain_index: torch.Tensor):
        chain_index = chain_index.long()
        remapped = torch.zeros_like(chain_index)
        valid = chain_index >= 0
        if valid.any():
            unique_chain_ids = torch.unique(chain_index[valid], sorted=True)
            for new_idx, chain_id in enumerate(unique_chain_ids.tolist(), start=1):
                remapped[chain_index == int(chain_id)] = int(new_idx)
        return remapped, valid

    def _build_index_condition_features(self, chain_index, residue_index, edge_index, dtype, device):
        if (not self.use_index_pos) or chain_index is None or residue_index is None:
            n, k = edge_index.shape
            return (
                torch.zeros((n, self.d_node), device=device, dtype=dtype),
                torch.zeros((n, k, self.d_edge), device=device, dtype=dtype),
            )

        chain_index = chain_index.to(device=device)
        residue_index = residue_index.to(device=device)
        remapped_chain, chain_valid = self._remap_chain_index(chain_index)

        residue_scalar = residue_index.float().unsqueeze(-1) / self.residue_pos_scale
        chain_scalar = remapped_chain.float().unsqueeze(-1)
        residue_pos = self.node_residue_pos_enc(residue_scalar).flatten(1)
        chain_pos = self.node_chain_pos_enc(chain_scalar).flatten(1)
        node_cond = torch.cat(
            [
                residue_pos.to(dtype=dtype),
                chain_pos.to(dtype=dtype),
                chain_valid.float().unsqueeze(-1).to(dtype=dtype),
            ],
            dim=-1,
        )
        node_cond = self.node_index_embed(node_cond)

        src_chain = remapped_chain[:, None]
        dst_chain = remapped_chain[edge_index]
        src_valid = chain_valid[:, None]
        dst_valid = chain_valid[edge_index]
        same_chain = src_valid & dst_valid & (src_chain == dst_chain)

        rel_residue = residue_index[:, None] - residue_index[edge_index]
        rel_residue = torch.where(same_chain, rel_residue, torch.zeros_like(rel_residue))
        rel_residue = rel_residue.clamp(-self.max_rel_res_offset, self.max_rel_res_offset)
        rel_residue_scalar = rel_residue.float().unsqueeze(-1) / self.residue_pos_scale
        rel_residue_pos = self.edge_relative_pos_enc(rel_residue_scalar).reshape(rel_residue.shape[0], rel_residue.shape[1], -1)
        edge_cond = torch.cat(
            [
                rel_residue_pos.to(dtype=dtype),
                same_chain.float().unsqueeze(-1).to(dtype=dtype),
                src_valid.float().expand_as(rel_residue).unsqueeze(-1).to(dtype=dtype),
                dst_valid.float().unsqueeze(-1).to(dtype=dtype),
            ],
            dim=-1,
        )
        edge_cond = self.edge_index_embed(edge_cond)
        return node_cond, edge_cond

    def _build_edge_pairing_features(self, edge_pairing, edge_pairing_label, edge_index, dtype, device):
        if (not self.use_edge_pairing_input) or edge_pairing is None:
            n, k = edge_index.shape
            return torch.zeros((n, k, self.d_edge), device=device, dtype=dtype)

        edge_pairing = edge_pairing.to(device=device, dtype=dtype)
        n, k = edge_index.shape
        if edge_pairing.ndim == 1 and edge_pairing.shape[0] == n:
            partner_index = edge_pairing.long()
            valid_partner = partner_index >= 0
            pairing_binary = (
                (edge_index == partner_index[:, None]) & valid_partner[:, None]
            ).to(dtype=dtype).unsqueeze(-1)
            pairing_score = pairing_binary
        elif edge_pairing.ndim == 2 and tuple(edge_pairing.shape) == (n, k):
            pairing_score = edge_pairing.unsqueeze(-1)
            if edge_pairing_label is None:
                pairing_binary = (pairing_score > 0.0).to(dtype=dtype)
            else:
                pairing_binary = edge_pairing_label.to(device=device, dtype=dtype).unsqueeze(-1)
        else:
            row_idx = torch.arange(n, device=device)[:, None]
            pairing_score = edge_pairing[row_idx, edge_index].unsqueeze(-1)
            if edge_pairing_label is None:
                pairing_binary = (pairing_score > 0.0).to(dtype=dtype)
            elif edge_pairing_label.ndim == 2 and tuple(edge_pairing_label.shape) == (n, k):
                pairing_binary = edge_pairing_label.to(device=device, dtype=dtype).unsqueeze(-1)
            else:
                pairing_binary = edge_pairing_label.to(device=device, dtype=dtype)[row_idx, edge_index].unsqueeze(-1)
        pairing_feat = torch.cat([pairing_score, pairing_binary], dim=-1)
        return self.edge_pairing_embed(pairing_feat)

    def forward(
        self,
        affines: torch.Tensor,
        prot_mask: torch.Tensor,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        prot_seq_embed: torch.Tensor = None,
        prot_seq_embed_mask: torch.Tensor = None,
        na_seq_embed: torch.Tensor = None,
        na_seq_embed_mask: torch.Tensor = None,
        prev_aa_probs: torch.Tensor = None,
        prev_rmsd: torch.Tensor = None,
        prev_node: torch.Tensor = None,
        edge_pairing: torch.Tensor = None,
        edge_pairing_label: torch.Tensor = None,
        chain_index: torch.Tensor = None,
        residue_index: torch.Tensor = None,
        batch=None,
        run_iters=1,
        **kwargs,
    ):
        if batch is None:
            batch = torch.zeros(len(affines), device=affines.device, dtype=torch.long)

        if cryo_grids is None or cryo_global_origins is None or cryo_voxel_sizes is None:
            raise ValueError("Cryo grid inputs must be provided.")

        init_affines = affines
        init_node = torch.zeros((len(affines), self.d_node), device=affines.device)
        max_k = min(self.k, len(affines) - 1)
        init_edge = torch.zeros((len(affines), max_k, self.d_edge), device=affines.device)
        init_aa_logits = torch.zeros((len(affines), self.num_res_types), device=affines.device)
        init_edge_aa_logits = torch.zeros((len(affines), max_k, self.num_res_types * self.num_res_types), device=affines.device)
        init_rmsd = torch.zeros((len(affines), 1), device=affines.device)
        if prev_node is not None:
            if prev_node.shape != init_node.shape:
                raise ValueError(f"prev_node must have shape {tuple(init_node.shape)}, got {tuple(prev_node.shape)}")
            init_node = prev_node.to(device=affines.device, dtype=init_node.dtype)
        if prev_aa_probs is not None:
            if prev_aa_probs.shape != init_aa_logits.shape:
                raise ValueError(f"prev_aa_probs must have shape {tuple(init_aa_logits.shape)}, got {tuple(prev_aa_probs.shape)}")
            init_aa_logits = prev_aa_probs.to(device=affines.device, dtype=init_aa_logits.dtype)
        if prev_rmsd is not None:
            if prev_rmsd.shape == (len(affines),):
                prev_rmsd = prev_rmsd[..., None]
            if prev_rmsd.shape != init_rmsd.shape:
                raise ValueError(f"prev_rmsd must have shape {tuple(init_rmsd.shape)}, got {tuple(prev_rmsd.shape)}")
            init_rmsd = prev_rmsd.to(device=affines.device, dtype=init_rmsd.dtype)

        fixed_edge_index = None
        fixed_full_edge_index = None

        pred_aatype = None
        pred_edge_aa_logits = None
        pred_node_existence = None
        pred_edge_existence = None
        pred_pairing = None

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                node_density, edge_density, bde_out = self._run_cryo_init(
                    affines=init_affines,
                    cryo_grids=cryo_grids,
                    cryo_global_origins=cryo_global_origins,
                    cryo_voxel_sizes=cryo_voxel_sizes,
                    edge_index=fixed_edge_index,
                    full_edge_index=fixed_full_edge_index,
                    batch=batch,
                )
                if fixed_edge_index is None:
                    fixed_edge_index = bde_out.edge_index
                    fixed_full_edge_index = bde_out.full_edge_index

                node = node_density
                edge = edge_density
                node_density_state = node_density
                edge_density_state = edge_density

                node_type, edge_type = self.embed_mol_type(prot_mask, edge_index=bde_out.edge_index)
                node = node + node_type
                edge = edge + edge_type

                node = node + self.embed_cycle_node(init_node)
                edge = edge + self.embed_cycle_edge(init_edge)
                node = node + self.embed_cycle_aa(init_aa_logits)
                edge = edge + self.embed_cycle_edge_aa(init_edge_aa_logits)
                node_conf_feat, edge_conf_feat = self._build_confidence_features(init_rmsd, bde_out.edge_index)
                node = node + self.embed_cycle_conf(node_conf_feat)
                edge = edge + self.embed_cycle_edge_conf(edge_conf_feat)

                index_node_feat, index_edge_feat = self._build_index_condition_features(
                    chain_index=chain_index,
                    residue_index=residue_index,
                    edge_index=bde_out.edge_index,
                    dtype=node.dtype,
                    device=node.device,
                )
                node = node + index_node_feat
                edge = edge + index_edge_feat
                edge = edge + self._build_edge_pairing_features(
                    edge_pairing=edge_pairing,
                    edge_pairing_label=edge_pairing_label,
                    edge_index=bde_out.edge_index,
                    dtype=edge.dtype,
                    device=edge.device,
                )

                torsion_angles = None
                edge = edge + self._compute_edge_bias(init_affines, torsion_angles, prot_mask, bde_out.edge_index)

                affines_list = []
                torsion_list = []
                affines = init_affines
                seq_attn_idx = 0
                for i in range(self.n_block):
                    do_seq_attn = self.use_seq_attn and self.seq_attn_every > 0 and ((i + 1) % self.seq_attn_every == 0)
                    do_geometry_update = self.blocks[i].enable_geometry_update

                    if do_seq_attn:
                        node = self.seq_attn_blocks[seq_attn_idx](
                            node,
                            prot_mask=prot_mask.bool(),
                            batch=batch,
                            prot_seq_emb=prot_seq_embed,
                            prot_seq_mask=prot_seq_embed_mask,
                            na_seq_emb=na_seq_embed,
                            na_seq_mask=na_seq_embed_mask,
                        )
                        seq_attn_idx += 1

                    node, edge, affines, torsion_angles = self.blocks[i](
                        node,
                        edge,
                        affines,
                        pos_emb=bde_out.pos3d_emb,
                        edge_index=bde_out.edge_index,
                        use_checkpoint=self.use_checkpoint,
                    )

                    if do_geometry_update and i != self.n_block - 1:
                        node_density, edge_density, _ = self._run_cryo_init(
                            affines=affines,
                            cryo_grids=cryo_grids,
                            cryo_global_origins=cryo_global_origins,
                            cryo_voxel_sizes=cryo_voxel_sizes,
                            edge_index=fixed_edge_index,
                            full_edge_index=fixed_full_edge_index,
                            batch=batch,
                        )
                        if not self.has_intermediate_refresh:
                            raise RuntimeError("Intermediate density refresh triggered without refresh modules.")
                        node_density_state = node_density_state + self.refresh_node_head_embed(node_density)
                        edge_density_state = edge_density_state + self.refresh_edge_head_embed(edge_density)
                        node = node + self.refresh_node_embed(node_density)
                        edge = edge + self.refresh_edge_embed(edge_density)
                        edge = edge + self._compute_edge_bias(affines, torsion_angles, prot_mask, bde_out.edge_index)

                    if do_geometry_update:
                        affines_list.append(affines)
                        torsion_list.append(torsion_angles)

                pred_aatype = self.aa_predictor(node_density_state, node)
                pred_edge_aa_logits = self.edge_aa_predictor(edge_density_state, edge)
                rmsd = self.rmsd_predictor(node)

                if self.pred_node_exist:
                    pred_node_existence = self.node_existence_predictor(node)
                if self.pred_edge_exist:
                    pred_edge_existence = self.edge_existence_predictor(edge)
                if self.pred_pairing:
                    pred_pairing = self.pairing_predictor(edge)

                init_affines = affines
                init_node = node
                init_edge = edge

                with torch.no_grad():
                    prot_probs = torch.softmax(pred_aatype[..., :20], dim=-1)
                    na_probs = torch.softmax(pred_aatype[..., 20:], dim=-1)
                    init_aa_logits = torch.zeros_like(pred_aatype)
                    prot_mask_bool = prot_mask.bool().unsqueeze(-1)
                    init_aa_logits[..., :20] = torch.where(prot_mask_bool, prot_probs, torch.zeros_like(prot_probs))
                    init_aa_logits[..., 20:] = torch.where(~prot_mask_bool, na_probs, torch.zeros_like(na_probs))
                    init_edge_aa_logits = torch.softmax(
                        pred_edge_aa_logits.reshape(pred_edge_aa_logits.shape[0], pred_edge_aa_logits.shape[1], -1),
                        dim=-1,
                    )
                    init_rmsd = rmsd

        output = Output(
            prot_mask=prot_mask,
            pred_aatype=pred_aatype,
            pred_aa_logits=pred_aatype,
            pred_edge_aa_logits=pred_edge_aa_logits,
            pred_torsions=torsion_list,
            pred_affines=affines_list,
            pred_positions=[x[..., :3, -1] for x in affines_list],
            pred_rmsd=rmsd,
            recycle_node_state=node,
            pred_node_existence=pred_node_existence if self.pred_node_exist else None,
            pred_edge_existence=pred_edge_existence if self.pred_edge_exist else None,
            edge_index=bde_out.edge_index,
            full_edge_index=bde_out.full_edge_index,
            pred_pairing=pred_pairing if self.pred_pairing else None,
        )
        if self.pred_ss:
            output["pred_ss_logits"] = self.ss_predictor(output["recycle_node_state"])
        else:
            output["pred_ss_logits"] = None
        if self.pred_na_type:
            output["pred_na_type_logits"] = self.na_type_predictor(output["recycle_node_state"])
        else:
            output["pred_na_type_logits"] = None
        return output
