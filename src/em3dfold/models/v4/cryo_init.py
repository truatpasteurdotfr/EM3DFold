from collections import namedtuple

import einops
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from em3dfold.models.v4.backbone_geometry import BackboneGeometryEmbedding
from em3dfold.utils.affine_utils import (
    sample_centered_cube_rot_matrix,
    sample_centered_rectangle_along_vector,
)
from em3dfold.utils.torch_utils import get_batches_to_idx


CryoFeatureInitOutput = namedtuple(
    "CryoFeatureInitOutput",
    [
        "node_state",
        "pair_state",
        "node_density",
        "pair_density",
        "geometry",
    ],
)


class SpatialMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=[-3, -2, -1])


class LocalDensityStem(nn.Module):
    def __init__(self, in_channels, out_channels, activation_class=nn.ReLU, checkpoint=True):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = activation_class()

    def _forward_impl(self, x):
        return self.activation(self.norm(self.conv(x)))

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(x)


class PyramidDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation_class=nn.ReLU, checkpoint=True):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.conv_in = nn.Conv3d(in_channels, out_channels * 3, kernel_size=1, bias=False)
        self.norm_in = nn.InstanceNorm3d(out_channels * 3, affine=True)
        self.conv_down = nn.Conv3d(
            out_channels * 3,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm_down = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = activation_class()

    def _forward_impl(self, x):
        x = self.activation(self.norm_in(self.conv_in(x)))
        x = self.activation(self.norm_down(self.conv_down(x)))
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._forward_impl(x)


class MultiScaleDensityEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation_class=nn.ReLU, checkpoint=True):
        super().__init__()
        self.pool = SpatialMean()
        self.stem = LocalDensityStem(
            in_channels=in_channels,
            out_channels=hidden_channels * 4,
            activation_class=activation_class,
            checkpoint=checkpoint,
        )
        self.down1 = PyramidDownsample(
            in_channels=hidden_channels * 4,
            out_channels=hidden_channels,
            activation_class=activation_class,
            checkpoint=checkpoint,
        )
        self.vision1 = nn.Conv3d(hidden_channels, out_channels, kernel_size=9, bias=False)
        self.down2 = PyramidDownsample(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 4,
            activation_class=activation_class,
            checkpoint=checkpoint,
        )
        self.vision2 = nn.Conv3d(hidden_channels * 4, out_channels, kernel_size=5, bias=False)
        self.down3 = PyramidDownsample(
            in_channels=hidden_channels * 4,
            out_channels=hidden_channels * 16,
            activation_class=activation_class,
            checkpoint=checkpoint,
        )
        self.vision3 = nn.Conv3d(hidden_channels * 16, out_channels, kernel_size=3, bias=False)
        self.output_norm = nn.LayerNorm(out_channels)

    def _pool_branch(self, branch, x):
        return self.pool(branch(x))

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x1 = self._pool_branch(self.vision1, x)
        x = self.down2(x)
        x2 = self._pool_branch(self.vision2, x)
        x = self.down3(x)
        x3 = self._pool_branch(self.vision3, x)
        return self.output_norm(x1 + x2 + x3)


class CryoFeatureInitializer(nn.Module):
    def __init__(
        self,
        d_node: int = 256,
        d_edge: int = 128,
        d_cryo_emb: int = 256,
        c_grid: int = 1,
        cube_size: int = 17,
        rectangle_length: int = 12,
        k: int = 32,
        activation_class: nn.Module = nn.ReLU,
        checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__()
        if d_cryo_emb % 4 != 0:
            raise ValueError(f"d_cryo_emb must be divisible by 4, got {d_cryo_emb}")

        self.k = k
        self.d_node = d_node
        self.d_edge = d_edge
        self.cube_size = cube_size
        self.rectangle_length = rectangle_length
        self.d_cryo_emb = d_cryo_emb
        self.use_checkpoint = checkpoint

        self.geometry_embedding = BackboneGeometryEmbedding(
            num_neighbors=k,
            position_encoding_dim=16,
            pair_geometry_dim=d_edge,
        )
        self.node_residual_norm = nn.LayerNorm(d_node)
        self.residue_type_bias = nn.Embedding(2, d_node)

        self.cube_encoder = MultiScaleDensityEncoder(
            in_channels=c_grid,
            hidden_channels=self.d_cryo_emb // 4,
            out_channels=self.d_node,
            activation_class=activation_class,
            checkpoint=checkpoint,
        )
        self.rectangle_encoder_pre = nn.Conv3d(
            in_channels=c_grid,
            out_channels=self.d_cryo_emb // 4,
            kernel_size=3,
            bias=False,
        )
        self.rectangle_encoder_post = nn.Sequential(
            nn.LayerNorm(self.d_cryo_emb // 4 * (self.rectangle_length - 2)),
            activation_class(),
            nn.Linear(self.d_cryo_emb // 4 * (self.rectangle_length - 2), self.d_edge, bias=False),
        )

    def _run_with_checkpoint(self, module, x):
        if self.use_checkpoint and x.requires_grad:
            return torch_checkpoint(
                module,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return module(x)

    def forward(
        self,
        affines,
        prot_mask,
        residual_node=None,
        torsion_angles=None,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        edge_index=None,
        full_edge_index=None,
        batch=None,
        **kwargs,
    ):
        if cryo_grids is None:
            raise ValueError("cryo_grids must be provided")

        if batch is None:
            batch = torch.zeros(len(affines), dtype=torch.long, device=affines.device)

        num_residues = len(affines)
        num_neighbors = min(num_residues - 1, self.k)
        batch_to_indices = get_batches_to_idx(batch)

        geometry = self.geometry_embedding(
            affines=affines,
            prot_mask=prot_mask,
            torsion_angles=torsion_angles,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
            batch=batch,
            k=num_neighbors,
        )

        with torch.no_grad():
            batch_cryo_grids = [
                cryo_grid.expand(len(indices), -1, -1, -1, -1)
                for cryo_grid, indices in zip(cryo_grids, batch_to_indices)
            ]
            cryo_points = [
                (geometry.positions[indices].reshape(-1, 3) - origin) / voxel_size
                for indices, origin, voxel_size in zip(
                    batch_to_indices,
                    cryo_global_origins,
                    cryo_voxel_sizes,
                )
            ]
            cryo_rotations = [
                affines[indices][..., :3, :3].reshape(-1, 3, 3)
                for indices in batch_to_indices
            ]
            cryo_cubes = sample_centered_cube_rot_matrix(
                batch_cryo_grids,
                cryo_rotations,
                cryo_points,
                cube_side=self.cube_size,
            )

        node_density = self.cube_encoder(cryo_cubes.requires_grad_())
        if residual_node is None:
            residual_node = torch.zeros_like(node_density)
        node_state = (
            node_density
            + self.residue_type_bias(prot_mask.long())
            + self.node_residual_norm(residual_node)
        )

        with torch.no_grad():
            batch_cryo_grids = [
                cryo_grid.expand(len(indices) * num_neighbors, -1, -1, -1, -1)
                for cryo_grid, indices in zip(cryo_grids, batch_to_indices)
            ]
            cryo_vectors = geometry.neighbor_vectors.detach()
            cryo_vectors = [cryo_vectors[indices].reshape(-1, 3) for indices in batch_to_indices]
            cryo_vector_origins = [
                (
                    geometry.positions[indices]
                    .unsqueeze(1)
                    .expand(len(indices), num_neighbors, 3)
                    .reshape(-1, 3)
                    - origin
                )
                / voxel_size
                for indices, origin, voxel_size in zip(
                    batch_to_indices,
                    cryo_global_origins,
                    cryo_voxel_sizes,
                )
            ]
            cryo_rectangles = sample_centered_rectangle_along_vector(
                batch_cryo_grids,
                cryo_vectors,
                cryo_vector_origins,
                rectangle_length=self.rectangle_length,
            )

        pair_density = self._run_with_checkpoint(
            self.rectangle_encoder_pre,
            cryo_rectangles.requires_grad_(),
        )
        pair_density = einops.rearrange(
            pair_density,
            "(b k) c z y x -> b k (c z y x)",
            k=num_neighbors,
            c=self.d_cryo_emb // 4,
            z=(self.rectangle_length - 2),
            y=1,
            x=1,
        )
        pair_density = self._run_with_checkpoint(self.rectangle_encoder_post, pair_density)
        pair_state = pair_density + geometry.pair_geometry

        return CryoFeatureInitOutput(
            node_state=node_state,
            pair_state=pair_state,
            node_density=node_density,
            pair_density=pair_density,
            geometry=geometry,
        )
