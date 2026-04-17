import torch

def knn_graph(x, k, batch=None, loop=False, flow='source_to_target'):
    assert flow in ['source_to_target', 'target_to_source']

    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    if x.dim() == 1:
        x = x.view(-1, 1)

    n = x.size(0)
    if not loop:
        k_effective = min(k, n - 1)
        if k_effective < k:
            print(f"Warning: k reduced from {k} to {k_effective} because loop=False and n={n}")
    else:
        k_effective = min(k, n)
        if k_effective < k:
            print(f"Warning: k reduced from {k} to {k_effective} because n={n}")

    if k_effective <= 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    if batch is not None:
        batch_max = batch.max().item() + 1
        batch_scale = 2 * x.size(1) * batch_max
        x_offset = x + batch_scale * batch.view(-1, 1).to(x.dtype)
    else:
        x_offset = x

    x_squared = (x_offset * x_offset).sum(dim=1, keepdim=True)
    dist_matrix = x_squared + x_squared.t() - 2 * torch.mm(x_offset, x_offset.t())

    dist_matrix = torch.clamp(dist_matrix, min=0)

    if not loop:
        dist_matrix.fill_diagonal_(float('inf'))

    try:
        _, indices = torch.topk(dist_matrix, k=k_effective, dim=1, largest=False)
    except RuntimeError as e:
        print(f"Warning: topk failed, using alternative method: {e}")
        return _safe_knn_graph(x, k, batch, loop, flow)

    row = torch.arange(n, device=x.device).view(-1, 1).repeat(1, k_effective)
    col = indices

    edge_index = torch.stack([row.view(-1), col.view(-1)], dim=0)

    if flow == 'source_to_target':
        edge_index = torch.stack([col.view(-1), row.view(-1)], dim=0)

    return edge_index

def _safe_knn_graph(x, k, batch=None, loop=False, flow='source_to_source'):
    n = x.size(0)

    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=x.device)

    if not loop:
        k_effective = min(k, n - 1)
    else:
        k_effective = min(k, n)

    if k_effective <= 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    edges = []

    for i in range(n):
        current_batch = batch[i]

        same_batch_mask = batch == current_batch
        same_batch_indices = torch.where(same_batch_mask)[0]

        if len(same_batch_indices) <= (1 if not loop else 0):
            continue

        distances = torch.cdist(x[i:i+1], x[same_batch_indices]).squeeze(0)

        if not loop:
            self_mask = same_batch_indices != i
            distances = distances[self_mask]
            candidate_indices = same_batch_indices[self_mask]
        else:
            candidate_indices = same_batch_indices

        actual_k = min(k_effective, len(candidate_indices))
        if actual_k <= 0:
            continue

        _, topk_indices = torch.topk(distances, k=actual_k, largest=False)
        neighbors = candidate_indices[topk_indices]

        for j in range(actual_k):
            if flow == 'source_to_target':
                edges.append([i, neighbors[j].item()])
            else:
                edges.append([neighbors[j].item(), i])

    if not edges:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    return torch.tensor(edges, dtype=torch.long, device=x.device).t()

def test_knn_graph():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = 2
    edge_index = knn_graph(x, k, loop=False)
    print(f"{edge_index.shape}")
    print(f"{edge_index}")

    x_small = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    k_large = 5
    edge_index_small = knn_graph(x_small, k_large, loop=False)
    print(f"{edge_index_small.shape}")
    print(f"{edge_index_small}")

    batch = torch.tensor([0, 0, 1, 1])
    edge_index_batch = knn_graph(x, k, batch=batch, loop=False)
    print(f"{edge_index_batch.shape}")
    print(f"{edge_index_batch}")

if __name__ == "__main__":
    test_knn_graph()

    positions = torch.randn(100, 3)
    edge_index = knn_graph(positions, 100, loop=False, flow='source_to_target')

    print(edge_index[0].shape)
    print(edge_index[1].shape)
    print(edge_index.shape)

