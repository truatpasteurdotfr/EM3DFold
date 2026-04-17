import contextlib

import torch
import torch.nn as nn

from em3dfold.models.mol import MolTypeEmbedder
from em3dfold.models.v4.cryo_init import CryoFeatureInitializer
from em3dfold.models.v4.layer import GraphEncoderLayer
from em3dfold.models.v4.heads import (
    FusedPairResidueTypeHead,
    FusedResidueTypeHead,
    PairPredictionHead,
    ResidualStateHead,
)
from em3dfold.models.v4.structure import StructureRefinementBlock
from em3dfold.polymer_utils import residue_constants as rc


class Output:
    def __init__(self, **kwargs):
        self.data = dict(kwargs)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def to(self, device_or_dtype):
        for key, value in self.data.items():
            if torch.is_tensor(value):
                self.data[key] = value.to(device_or_dtype)
            elif isinstance(value, list):
                self.data[key] = [item.to(device_or_dtype) for item in value]
        return self
class Model(nn.Module):
    def __init__(
        self,
        d_node=256,
        d_edge=128,
        d_head=48,
        d_seq=1280,
        d_seq_na=1280,
        n_qk_point=4,
        n_v_point=8,
        n_head=8,
        n_block=16,
        n_structure_block=4,
        k=32,
        p_drop=0.10,
        c_grid=1,
        pred_node_exist=False,
        pred_edge_exist=True,
        pred_pairing=False,
        use_checkpoint=False,
        use_seq_attn=True,
        seq_attn_every=2,
        d_cryo_emb=None,
        cube_size=23,
        rectangle_length=15,
    ):
        super().__init__()
        if use_seq_attn and seq_attn_every <= 0:
            raise ValueError(f"seq_attn_every must be positive, got {seq_attn_every}")

        self.d_node = d_node
        self.d_edge = d_edge
        self.d_seq = d_seq
        self.d_seq_na = d_seq if d_seq_na is None else d_seq_na
        self.n_head = n_head
        self.n_block = n_block
        self.n_structure_block = n_structure_block
        self.k = k
        self.use_checkpoint = use_checkpoint
        self.use_seq_attn = use_seq_attn
        self.seq_attn_every = seq_attn_every
        self.num_residue_types = 24
        self.num_torsion_outputs = rc.canonical_num_residues * 10
        self.pred_node_exist = pred_node_exist
        self.pred_edge_exist = pred_edge_exist
        self.pred_pairing = pred_pairing

        self.feature_initializer = CryoFeatureInitializer(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node if d_cryo_emb is None else d_cryo_emb,
            c_grid=c_grid,
            cube_size=cube_size,
            rectangle_length=rectangle_length,
            k=k,
            checkpoint=use_checkpoint,
        )
        self.mol_type_embedder = MolTypeEmbedder(
            d_node=d_node,
            d_edge=d_edge,
            num_classes=2,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                GraphEncoderLayer(
                    d_node=d_node,
                    d_edge=d_edge,
                    d_head=d_head,
                    n_head=n_head,
                    use_sequence_attention=(
                        self.use_seq_attn
                        and self.seq_attn_every > 0
                        and ((block_index + 1) % self.seq_attn_every == 0)
                    ),
                    d_seq=d_seq,
                    d_seq_na=self.d_seq_na,
                    checkpoint=use_checkpoint,
                )
                for block_index in range(n_block)
            ]
        )
        self.structure_blocks = nn.ModuleList(
            [
                StructureRefinementBlock(
                    d_node=d_node,
                    d_edge=d_edge,
                    n_head=n_head,
                    d_head=d_head,
                    n_qk_point=n_qk_point,
                    n_v_point=n_v_point,
                )
                for _ in range(n_structure_block)
            ]
        )

        self.residue_type_head = FusedResidueTypeHead(
            d_node=d_node,
            num_classes=self.num_residue_types,
        )
        self.pair_residue_type_head = FusedPairResidueTypeHead(
            d_edge=d_edge,
            num_classes=self.num_residue_types,
        )
        self.torsion_head = ResidualStateHead(
            in_features=d_node,
            hidden_features=d_node // 2,
            out_features=self.num_torsion_outputs * 2,
        )
        self.rmsd_head = ResidualStateHead(
            in_features=d_node,
            hidden_features=d_node // 2,
            out_features=1,
        )
        if self.pred_node_exist:
            self.node_existence_head = ResidualStateHead(
                in_features=d_node,
                hidden_features=d_node // 2,
                out_features=1,
            )
        if self.pred_edge_exist:
            self.edge_existence_head = PairPredictionHead(d_edge=d_edge, out_features=3)
        if self.pred_pairing:
            self.pairing_head = PairPredictionHead(d_edge=d_edge, out_features=1)

        self.recycle_pair_state = nn.Sequential(
            nn.LayerNorm(d_edge),
            nn.Linear(d_edge, d_edge, bias=False),
        )
        self.recycle_aa = nn.Sequential(
            nn.LayerNorm(self.num_residue_types),
            nn.Linear(self.num_residue_types, d_node, bias=False),
        )
        self.recycle_edge_aa = nn.Sequential(
            nn.LayerNorm(self.num_residue_types * self.num_residue_types),
            nn.Linear(self.num_residue_types * self.num_residue_types, d_edge, bias=False),
        )
        self.recycle_confidence = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, d_node, bias=False),
        )
        self.recycle_edge_confidence = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, d_edge, bias=False),
        )

    def _build_confidence_features(self, prev_rmsd, edge_index):
        prev_rmsd = torch.clamp(prev_rmsd, min=0.0)
        prev_conf = torch.exp(-prev_rmsd)

        node_conf_feat = torch.cat([prev_rmsd, prev_conf], dim=-1)

        neighbor_rmsd = prev_rmsd[edge_index]
        neighbor_conf = prev_conf[edge_index]
        src_rmsd = prev_rmsd[:, None, :].expand_as(neighbor_rmsd)
        src_conf = prev_conf[:, None, :].expand_as(neighbor_conf)
        edge_conf_feat = torch.cat(
            [
                0.5 * (src_rmsd + neighbor_rmsd),
                torch.abs(src_rmsd - neighbor_rmsd),
                0.5 * (src_conf + neighbor_conf),
                torch.abs(src_conf - neighbor_conf),
            ],
            dim=-1,
        )
        return node_conf_feat, edge_conf_feat

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
        batch=None,
        run_iters=1,
        **kwargs,
    ):
        if batch is None:
            batch = torch.zeros(len(affines), dtype=torch.long, device=affines.device)

        if cryo_grids is None or cryo_global_origins is None or cryo_voxel_sizes is None:
            raise ValueError("Cryo grid inputs must be provided.")

        current_affines = affines
        max_k = min(self.k, len(affines) - 1)
        current_node_state = torch.zeros((len(affines), self.d_node), device=affines.device)
        current_pair_state = torch.zeros((len(affines), max_k, self.d_edge), device=affines.device)
        current_torsions = None

        init_aa_probs = torch.zeros(
            (len(affines), self.num_residue_types),
            device=affines.device,
        )
        init_edge_aa_probs = torch.zeros(
            (len(affines), max_k, self.num_residue_types * self.num_residue_types),
            device=affines.device,
        )
        init_rmsd = torch.zeros((len(affines), 1), device=affines.device)

        if prev_aa_probs is not None:
            if prev_aa_probs.shape != init_aa_probs.shape:
                raise ValueError(
                    f"prev_aa_probs must have shape {tuple(init_aa_probs.shape)}, "
                    f"got {tuple(prev_aa_probs.shape)}"
                )
            init_aa_probs = prev_aa_probs.to(device=affines.device, dtype=init_aa_probs.dtype)

        if prev_rmsd is not None:
            if prev_rmsd.shape == (len(affines),):
                prev_rmsd = prev_rmsd[..., None]
            if prev_rmsd.shape != init_rmsd.shape:
                raise ValueError(
                    f"prev_rmsd must have shape {tuple(init_rmsd.shape)}, "
                    f"got {tuple(prev_rmsd.shape)}"
                )
            init_rmsd = prev_rmsd.to(device=affines.device, dtype=init_rmsd.dtype)

        fixed_edge_index = None
        fixed_full_edge_index = None

        pred_aatype = None
        pred_edge_aa_logits = None
        pred_torsions = None
        pred_rmsd = None
        pred_node_existence = None
        pred_edge_existence = None
        pred_pairing = None
        affines_list = []
        torsion_list = []

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                feature_output = self.feature_initializer(
                    affines=current_affines,
                    prot_mask=prot_mask,
                    residual_node=current_node_state,
                    torsion_angles=current_torsions,
                    cryo_grids=cryo_grids,
                    cryo_global_origins=cryo_global_origins,
                    cryo_voxel_sizes=cryo_voxel_sizes,
                    edge_index=fixed_edge_index,
                    full_edge_index=fixed_full_edge_index,
                    batch=batch,
                )

                geometry = feature_output.geometry
                if fixed_edge_index is None:
                    fixed_edge_index = geometry.edge_index
                    fixed_full_edge_index = geometry.full_edge_index

                node = feature_output.node_state
                pair = feature_output.pair_state
                node_density = feature_output.node_density
                pair_density = feature_output.pair_density

                node_type, pair_type = self.mol_type_embedder(
                    prot_mask,
                    edge_index=geometry.edge_index,
                )
                node = node + node_type
                pair = pair + pair_type

                pair = pair + self.recycle_pair_state(current_pair_state)
                node = node + self.recycle_aa(init_aa_probs)
                pair = pair + self.recycle_edge_aa(init_edge_aa_probs)
                node_conf_feat, edge_conf_feat = self._build_confidence_features(
                    init_rmsd,
                    geometry.edge_index,
                )
                node = node + self.recycle_confidence(node_conf_feat)
                pair = pair + self.recycle_edge_confidence(edge_conf_feat)

                for block in self.encoder_blocks:
                    node, pair = block(
                        node=node,
                        pair=pair,
                        position_embedding=geometry.position_embedding,
                        edge_index=geometry.edge_index,
                        prot_mask=prot_mask.bool(),
                        batch=batch,
                        attention_batch_size=200,
                        prot_seq_emb=prot_seq_embed,
                        prot_seq_mask=prot_seq_embed_mask,
                        na_seq_emb=na_seq_embed,
                        na_seq_mask=na_seq_embed_mask,
                    )

                node_residual = node
                affines_list = []
                torsion_list = []
                refined_affines = current_affines
                for block in self.structure_blocks:
                    node, refined_affines = block(
                        node=node,
                        pair=pair,
                        affines=refined_affines,
                        position_embedding=geometry.position_embedding,
                        edge_index=geometry.edge_index,
                    )
                    affines_list.append(refined_affines)
                    torsion_list.append(
                        self.torsion_head(node_residual, node).reshape(
                            len(node),
                            self.num_torsion_outputs,
                            2,
                        )
                    )

                pred_aatype = self.residue_type_head(node_density, node)
                pred_edge_aa_logits = self.pair_residue_type_head(pair_density, pair)
                pred_torsions = torsion_list[-1]
                pred_rmsd = self.rmsd_head(node_residual, node)

                if self.pred_node_exist:
                    pred_node_existence = self.node_existence_head(node_residual, node)
                if self.pred_edge_exist:
                    pred_edge_existence = self.edge_existence_head(pair)
                if self.pred_pairing:
                    pred_pairing = self.pairing_head(pair)

                current_affines = refined_affines
                current_node_state = node
                current_pair_state = pair
                current_torsions = pred_torsions

                with torch.no_grad():
                    prot_probs = torch.softmax(pred_aatype[..., :20], dim=-1)
                    na_probs = torch.softmax(pred_aatype[..., 20:], dim=-1)
                    init_aa_probs = torch.zeros_like(pred_aatype)
                    prot_mask_bool = prot_mask.bool().unsqueeze(-1)
                    init_aa_probs[..., :20] = torch.where(
                        prot_mask_bool,
                        prot_probs,
                        torch.zeros_like(prot_probs),
                    )
                    init_aa_probs[..., 20:] = torch.where(
                        ~prot_mask_bool,
                        na_probs,
                        torch.zeros_like(na_probs),
                    )

                    init_edge_aa_probs = torch.softmax(
                        pred_edge_aa_logits.reshape(
                            pred_edge_aa_logits.shape[0],
                            pred_edge_aa_logits.shape[1],
                            -1,
                        ),
                        dim=-1,
                    )
                    init_rmsd = pred_rmsd

        return Output(
            prot_mask=prot_mask,
            pred_aatype=pred_aatype,
            pred_aa_logits=pred_aatype,
            pred_edge_aa_logits=pred_edge_aa_logits,
            pred_torsions=torsion_list,
            pred_affines=affines_list,
            pred_positions=[x[..., :3, -1] for x in affines_list],
            pred_rmsd=pred_rmsd,
            pred_node_existence=pred_node_existence if self.pred_node_exist else None,
            pred_edge_existence=pred_edge_existence if self.pred_edge_exist else None,
            edge_index=fixed_edge_index,
            full_edge_index=fixed_full_edge_index,
            pred_pairing=pred_pairing if self.pred_pairing else None,
        )
