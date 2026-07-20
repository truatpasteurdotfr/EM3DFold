import torch
import torch.nn as nn
import torch.nn.functional as F

class MolTypeEmbedder(nn.Module):
    def __init__(self, d_node=256, d_edge=128, num_classes=2):
        super().__init__()

        self.d_node = d_node
        self.d_edge = d_edge

        # node embedding
        self.node_proj = nn.Linear(2, self.d_node, bias=False)

        # pair embedding projections
        self.left_proj = nn.Linear(d_node, d_edge, bias=False)
        self.right_proj = nn.Linear(d_node, d_edge, bias=False)

        self.num_classes = num_classes

    def forward(self, x, edge_index=None):
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes).float()

        # embed node
        node_feat = self.node_proj(x)

        # embed edge
        left_feat = self.left_proj(node_feat)
        right_feat = self.right_proj(node_feat)

        if edge_index is not None:
            # knn
            n, k = edge_index.shape
            neigh_right_feat = right_feat[edge_index]
            src_left_feat = left_feat[:, None, :].expand(n, k, self.d_edge)
            pair_feat = src_left_feat + neigh_right_feat
        else:
            # all-to-all
            pair_feat = left_feat[:, None, :] + right_feat[None, :, :]

        return node_feat, pair_feat

