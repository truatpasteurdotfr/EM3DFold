Implement a function for 3D Gaussian-weighted fusion of overlapping patch predictions into a full volume.

## Goal
Given multiple overlapping 3D patches (predictions), reconstruct a full 3D volume using Gaussian-weighted averaging to smoothly resolve overlaps.

## Inputs
- patches: List or tensor of shape (K, B, B, B)
    Each element ŷ_k is a predicted 3D patch.
- positions: List or tensor of shape (K, 3)
    Each p_k is the starting voxel coordinate (z, y, x) of the patch in the full volume.
- volume_shape: Tuple (D, H, W)
    Shape of the output full volume (before padding).
- sigma: Float
    Standard deviation for Gaussian kernel.
- epsilon: Small float (e.g., 1e-6) to avoid division by zero.

## Requirements

1. Construct a 3D Gaussian kernel W_ker of size (B, B, B):
    - Center c = ((B-1)/2, (B-1)/2, (B-1)/2)
    - w[i,j,k] = exp(-|| (i,j,k) - c ||^2 / (2*sigma^2))
    - Normalize weights to range [1, 3] (center highest)

2. Initialize:
    - V: zero tensor of shape volume_shape (float32)
    - S: zero tensor of same shape

3. For each patch k:
    - Extract region R_k in V starting at p_k with size (B, B, B)
    - Accumulate:
        V[R_k] += patches[k] * W_ker
        S[R_k] += W_ker

4. Normalize:
    M = V / maximum(S, epsilon)

5. Return M

## Edge Cases
- Handle patches that go out of bounds (crop patch and kernel accordingly)
- Ensure no division by zero using epsilon
- Support both NumPy and PyTorch (prefer PyTorch, but keep it easily adaptable)

## Output
- Return a 3D tensor of shape (D, H, W)

## Additional Notes
- Use vectorized operations where possible
- Avoid Python loops over voxels (loop over patches is OK)
- Make the implementation efficient and clean
- Include a small test example

## Optional
- Add a function to visualize a central slice
- Support batch processing
