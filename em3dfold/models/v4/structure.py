import math

import torch
from torch import nn
from einops.layers.torch import Rearrange

from em3dfold.models.modules import qk_modulation
from em3dfold.utils.affine_utils import (
    affine_composition,
    affine_mul_vecs,
    get_affine,
    invert_affine,
    quaternion_to_matrix,
)


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        n_head: int = 8,
        d_head: int = 48,
        n_qk_point: int = 4,
        n_v_point: int = 8,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.n_qk_point = n_qk_point
        self.n_v_point = n_v_point
        self.scale = math.sqrt(self.d_head)

        self.point_weight = nn.Parameter(1.1 * torch.ones(1, 1, self.n_head))
        self.output_norm = nn.LayerNorm(d_node)

        self.query_scalar = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=False),
            Rearrange("n (h d) -> n h d", h=n_head, d=d_head),
        )
        self.key_scalar = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=False),
            Rearrange("n (h d) -> n h d", h=n_head, d=d_head),
        )
        self.value_scalar = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=False),
            Rearrange("n (h d) -> n h d", h=n_head, d=d_head),
        )
        self.query_point = nn.Sequential(
            nn.Linear(d_node, n_head * n_qk_point * 3, bias=False),
            Rearrange("n (h p d) -> n h p d", h=n_head, p=n_qk_point, d=3),
        )
        self.key_point = nn.Sequential(
            nn.Linear(d_node, n_head * n_qk_point * 3, bias=False),
            Rearrange("n (h p d) -> n h p d", h=n_head, p=n_qk_point, d=3),
        )
        self.value_point = nn.Sequential(
            nn.Linear(d_node, n_head * n_v_point * 3, bias=False),
            Rearrange("n (h p d) -> n h p d", h=n_head, p=n_v_point, d=3),
        )
        self.attention_bias = nn.Linear(d_edge, n_head, bias=False)
        self.output_proj = nn.Linear(
            n_head * d_head + n_head * d_edge + n_head * n_v_point * 4,
            d_node,
            bias=False,
        )

    def forward(self, node, pair, affines, position_embedding, edge_index):
        query_scalar = self.query_scalar(node)
        key_scalar = self.key_scalar(node)[edge_index]
        query_scalar, key_scalar = qk_modulation(
            query_scalar,
            key_scalar,
            position_embedding,
            edge_index,
        )
        value_scalar = self.value_scalar(node)[edge_index]

        query_point = self.query_point(node)
        key_point = self.key_point(node)[edge_index]
        value_point = self.value_point(node)

        scalar_scores = (
            torch.einsum("n h d, n k h d -> n k h", query_scalar, key_scalar) / self.scale
            + self.attention_bias(pair)
        )

        query_point_global = affine_mul_vecs(affines, query_point)
        key_point_global = affine_mul_vecs(affines, key_point)
        point_scores = -torch.sum(
            torch.square(query_point_global[:, None] - key_point_global),
            dim=-1,
        ).sum(dim=-1)

        point_scale = math.sqrt(2.0 / (9.0 * self.n_qk_point))
        scalar_scale = math.sqrt(1.0 / 3.0)
        attention_scores = scalar_scale * (
            scalar_scores + 0.1 * point_scale * self.point_weight * point_scores
        )
        attention_weights = torch.softmax(attention_scores, dim=1)

        scalar_output = torch.einsum("n k h, n k h d -> n h d", attention_weights, value_scalar)
        pair_output = torch.einsum("n k h, n k d -> n h d", attention_weights, pair)

        value_point_global = affine_mul_vecs(affines, value_point)[edge_index]
        point_output = torch.einsum("n k h, n k h p d -> n h p d", attention_weights, value_point_global)
        point_output = affine_mul_vecs(invert_affine(affines), point_output)
        point_output_norm = torch.norm(point_output, dim=-1, p=2)

        combined = torch.cat(
            [
                scalar_output.flatten(1),
                pair_output.flatten(1),
                point_output.flatten(1),
                point_output_norm.flatten(1),
            ],
            dim=-1,
        )
        return self.output_norm(self.output_proj(combined) + math.sqrt(2) * node)


class StructureTransition(nn.Module):
    def __init__(self, d_node: int, expansion: int = 2):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Linear(d_node, d_node * expansion),
            nn.ReLU(),
            nn.Linear(d_node * expansion, d_node * expansion),
            nn.ReLU(),
            nn.Linear(d_node * expansion, d_node, bias=False),
        )
        self.output_norm = nn.LayerNorm(d_node)

    def forward(self, node):
        return self.output_norm(self.transition(node) + math.sqrt(2) * node)


class BackboneFrameUpdate(nn.Module):
    def __init__(self, d_node: int):
        super().__init__()
        self.backbone_proj = nn.Linear(d_node, 6)
        self.rotation_bias = nn.Parameter(torch.tensor(1.5, dtype=torch.float))
        self.eps = 1e-6
        torch.nn.init.normal_(self.backbone_proj.weight, std=0.02)
        self.backbone_proj.bias.data.zero_()

    def forward(self, node, affines):
        update = self.backbone_proj(node)
        update = torch.cat(
            [
                torch.sqrt(torch.square(self.rotation_bias) + self.eps)
                * torch.ones(update.shape[:-1] + (1,), device=update.device, dtype=update.dtype),
                update,
            ],
            dim=-1,
        )
        rotation = quaternion_to_matrix(update[..., :4])
        translation = update[..., 4:]
        return affine_composition(affines, get_affine(rotation, translation))


class StructureRefinementBlock(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        n_head: int = 8,
        d_head: int = 48,
        n_qk_point: int = 4,
        n_v_point: int = 8,
    ):
        super().__init__()
        self.ipa = InvariantPointAttention(
            d_node=d_node,
            d_edge=d_edge,
            n_head=n_head,
            d_head=d_head,
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
        )
        self.transition = StructureTransition(d_node)
        self.backbone_update = BackboneFrameUpdate(d_node)

    def forward(self, node, pair, affines, position_embedding, edge_index):
        node = self.ipa(node, pair, affines, position_embedding, edge_index)
        node = self.transition(node)
        affines = self.backbone_update(node, affines)
        return node, affines
