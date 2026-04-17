from torch import nn


class FusedResidueTypeHead(nn.Module):
    def __init__(self, d_node: int, num_classes: int):
        super().__init__()
        self.density_proj = nn.Linear(d_node, d_node)
        self.state_proj = nn.Linear(d_node, d_node)
        self.head = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, num_classes),
        )

    def forward(self, node_density, node_state):
        fused = self.density_proj(node_density) + self.state_proj(node_state)
        return self.head(fused)


class FusedPairResidueTypeHead(nn.Module):
    def __init__(self, d_edge: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.density_proj = nn.Linear(d_edge, d_edge)
        self.state_proj = nn.Linear(d_edge, d_edge)
        self.head = nn.Sequential(
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, num_classes * num_classes),
        )

    def forward(self, pair_density, pair_state):
        fused = self.density_proj(pair_density) + self.state_proj(pair_state)
        logits = self.head(fused)
        return logits.view(*logits.shape[:-1], self.num_classes, self.num_classes)


class ResidualStateHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.residual_proj = nn.Linear(in_features, hidden_features)
        self.state_proj = nn.Linear(in_features, hidden_features)
        self.block1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, residual_state, current_state):
        hidden = self.residual_proj(residual_state) + self.state_proj(current_state)
        hidden = hidden + self.block1(hidden)
        hidden = hidden + self.block2(hidden)
        return self.output_proj(hidden)


class PairPredictionHead(nn.Module):
    def __init__(self, d_edge: int, out_features: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, out_features),
        )

    def forward(self, pair_state):
        return self.head(pair_state)
