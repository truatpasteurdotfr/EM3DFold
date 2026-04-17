import torch
import torch.nn as nn
from typing import List
import contextlib

from em3dfold.models.v1.cryo_init import CryoInit
from em3dfold.models.v1.trackxx import TrackBlock
from em3dfold.models.v1.sequence_attention import SequenceAttention

from em3dfold.models.mol import MolTypeEmbedder

class Output:
    def __init__(self, **kwargs):
        self.data = dict()
        for k, v in kwargs.items():
            self.data[k] = v

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def to(self, device_or_dtype):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device_or_dtype)
            elif isinstance(v, list):
                self.data[k] = [x.to(device_or_dtype) for x in v]
        return self

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, **kwargs):
        self.data.update(kwargs)

    def __repr__(self):
        return f"Output({self.data})"


class FusedAAPredictor(nn.Module):
    def __init__(
        self,
        d_node: int = 256,
        n_classes: int = 4,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_node, d_node)
        self.linear2 = nn.Linear(d_node, d_node)

        self.aa_head = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, n_classes),
        )

    def forward(self, node_density, node):
        x = self.linear1(node_density) + self.linear2(node)
        return self.aa_head(x)


class Model(nn.Module):
    def __init__(
        self,
        d_node=256,
        d_edge=256,
        d_head=48,
        d_seq=1280, 
        n_qk_point=4,
        n_v_point=8,
        n_head=8,
        n_block=8,
        k=32,
        p_drop=0.10,
        c_grid=2,
        pred_node_exist=False, 
        pred_edge_exist=True, 
        pred_pairing=False, 
        use_checkpoint=False,
        model_type="protein",
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.d_node = d_node
        self.d_edge = d_edge
        self.k = k
        self.model_type = model_type

        # Feature init
        self.init = CryoInit(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node,
            c_grid=c_grid, 
            k=k,
        )

        # Embed mol type
        self.embed_mol_type = MolTypeEmbedder(
            d_node=d_node,
            d_edge=d_edge,
            num_classes=2,
        )

        # Track blocks
        self.n_block = n_block
        self.blocks = nn.ModuleList()
        for i in range(self.n_block):
            self.blocks.append(TrackBlock(
                d_node=d_node,
                d_edge=d_edge,
                d_head=d_head,
                n_head=n_head,
                n_qk_point=n_qk_point,
                n_v_point=n_v_point,
                p_drop=p_drop,
                k=k,
            ))

        # Sequence attention
        self.seq_attn_blocks = nn.ModuleList()
        for i in range(self.n_block):
            self.seq_attn_blocks.append(SequenceAttention(
                d=d_node,
                d_seq=d_seq,
                d_head=d_head,
                n_head=n_head,
                checkpoint=use_checkpoint,
            ))

        # Fused AA predictor
        self.aa_predictor = FusedAAPredictor(
            d_node=d_node,
            n_classes=20 + 4,
        )

        # pLDDT
        self.rmsd_predictor = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, 1),
        )

        # Node existence pred
        self.pred_node_exist = pred_node_exist
        if self.pred_node_exist:
            self.node_existence_predictor = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, 1),
            )

        # Edge existence pred
        self.pred_edge_exist = pred_edge_exist
        if self.pred_edge_exist:
            self.edge_existence_predictor = nn.Sequential(
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, 3),
            )

        # NA pairing pred
        self.pred_pairing = pred_pairing
        if self.pred_pairing:
            self.pairing_predictor = nn.Sequential(
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, d_edge),
                nn.ReLU(),
                nn.Linear(d_edge, 1),
            )

        # Previous cycle node embedder
        self.embed_cycle_node = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_node, bias=False),
        )

        # Previous cycle edge embedder
        self.embed_cycle_edge = nn.Sequential(
            nn.LayerNorm(d_edge),
            nn.Linear(d_edge, d_edge, bias=False),
        )

    def forward(
        self,
        affines: torch.Tensor, # (n, 3, 4) rot and trans
        prot_mask: torch.Tensor, # (n, )
        cryo_grids: List[torch.Tensor],
        cryo_global_origins: List[torch.Tensor],
        cryo_voxel_sizes: List[torch.Tensor],
        prot_seq_embed: torch.Tensor = None,
        prot_seq_embed_mask: torch.Tensor = None,
        na_seq_embed: torch.Tensor = None,
        na_seq_embed_mask: torch.Tensor = None,
        batch=None,
        run_iters=1,
        **kwargs,
    ):

        if batch is None:
            batch = torch.zeros( len(affines) ).to(affines.device).long()

        init_affines = affines
        init_node = torch.zeros( (len(affines), self.d_node) ).to(affines.device)
        max_k = min(self.k, len(affines) - 1)
        init_edge = torch.zeros( (len(affines), max_k, self.d_edge) ).to(affines.device)

        if self.model_type == "protein":
            seq_embed = prot_seq_embed
            seq_embed_mask = prot_seq_embed_mask
            node_mask = prot_mask.bool()
        elif self.model_type == "na":
            seq_embed = na_seq_embed
            seq_embed_mask = na_seq_embed_mask
            node_mask = ~(prot_mask.bool())
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        if seq_embed is None:
            raise ValueError(f"{self.model_type} model requires its matching sequence embedding.")
        
        if seq_embed_mask is None:
            seq_embed_mask = torch.ones(
                seq_embed.shape[0],
                seq_embed.shape[1],
                device=seq_embed.device,
                dtype=torch.bool,
            )

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                # Init features from map
                node, edge, bde_out = self.init(
                    affines=init_affines,
                    cryo_grids=cryo_grids,
                    cryo_global_origins=cryo_global_origins,
                    cryo_voxel_sizes=cryo_voxel_sizes,
                    batch=batch,
                )
                # dummy edge mask
                node_mask = torch.ones_like(node[..., 0]).bool()
                edge_mask = torch.ones_like(edge[..., 0]).bool()

                node_dens = node

                # Add mol type embedding
                node_type, edge_type = self.embed_mol_type(prot_mask, edge_index=bde_out.edge_index)
                node = node + node_type
                edge = edge + edge_type

                # Add previous cycle node and edge embedding
                node_prev = self.embed_cycle_node(init_node)
                edge_prev = self.embed_cycle_edge(init_edge)
                node = node + node_prev
                edge = edge + edge_prev

                # Three-Track block
                affines_list = []
                torsion_list = []
                torsion_angles = None
                for i in range(self.n_block):
                    node, edge, affines, torsion_angles = self.blocks[i](
                        node,
                        edge,
                        affines,
                        torsion_angles, 
                        prot_mask=prot_mask,
                        pos_emb=bde_out.pos3d_emb,
                        edge_index=bde_out.edge_index,
                        use_checkpoint=self.use_checkpoint,
                    )

                    affines_list.append(affines)
                    torsion_list.append(torsion_angles)

                    # Sequence attention
                    node = self.seq_attn_blocks[i](
                        node,
                        seq_embed,
                        seq_embed_mask,
                        node_mask=node_mask,
                    )

                # AA prediction
                pred_aatype = self.aa_predictor(node_dens, node)

                # pLDDT prediction
                rmsd = self.rmsd_predictor(node)

                # Node existence prediction
                if self.pred_node_exist:
                    pred_node_existence = self.node_existence_predictor(node)

                # Edge existence prediction
                if self.pred_edge_exist:
                    pred_edge_existence = self.edge_existence_predictor(
                        edge,
                    )

                # For NAs, pairing prediction
                if self.pred_pairing:
                    pred_pairing = self.pairing_predictor(
                        edge,
                    )

                # For next iteration
                init_affines = affines_list[-1]
                init_node = node
                init_edge = edge


        return Output(
            prot_mask=prot_mask,
            pred_aatype=pred_aatype,
            pred_torsions=torsion_list,
            pred_affines=affines_list,
            pred_positions=[affines[..., :3, -1] for affines in affines_list],
            pred_rmsd=rmsd,
            pred_node_existence=pred_node_existence if self.pred_node_exist else None,
            pred_edge_existence=pred_edge_existence if self.pred_edge_exist else None,
            edge_index=bde_out.edge_index,
            full_edge_index=bde_out.full_edge_index,
            pred_pairing=pred_pairing if self.pred_pairing else None,
        )

import time
if __name__ == '__main__':
    model = Model(n_block=12, use_checkpoint=True).cuda()
    print(sum(p.numel() for p in model.parameters()))

    n = 200
    for i in range(10):
        t0 = time.time()
        batch = torch.zeros(n).long().cuda()

        out = model(
            affines=torch.randn(n, 3, 4).cuda(),
            prot_mask=torch.randint(0, 2, (n, )).cuda(),
            prot_seq_embed=torch.randn(1, 2000, 1280).cuda(),
            prot_seq_embed_mask=torch.ones(1, 2000).bool().cuda(),
            na_seq_embed=torch.randn(1, 2000, 1280).cuda(),
            na_seq_embed_mask=torch.ones(1, 2000).bool().cuda(),
            cryo_grids=[torch.zeros(1, 1, 120, 120, 120).cuda()],
            cryo_global_origins=[torch.zeros(3).cuda()],
            cryo_voxel_sizes=[torch.ones(3).cuda()],
            batch=batch,
            run_iters=2,
        )
        t1 = time.time()


        out["pred_affines"][-1].mean().backward()

        for k, v in out.items():
            if hasattr(v, "shape"):
                print(k)
                print(v.shape)

        print("{:.4f}".format(t1 - t0))

        time.sleep(1)

        #exit()

