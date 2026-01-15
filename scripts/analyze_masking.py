import numpy as np
import torch
from scipy.spatial.distance import cdist

def simulate_inclusion(coords, k, num_trials=10000, strategy='uniform'):
    L = len(coords)
    inclusion_counts = np.zeros(L)
    
    # Precompute neighbor matrix (top-k neighbors for each residue)
    dists = cdist(coords, coords)
    nn_matrix = np.zeros((L, L))
    for i in range(L):
        indices = np.argsort(dists[i])[:k]
        nn_matrix[i, indices] = 1
    
    # c_i is the "coverage" (how many other residues have i in their top-k)
    c = nn_matrix.sum(axis=0)
    
    if strategy == 'uniform':
        weights = np.ones(L) / L
    elif strategy == 'balanced':
        # Simple heuristic: 1/coverage
        weights = 1.0 / np.clip(c, 1, None)
        weights /= weights.sum()
    elif strategy == 'ipf':
        # Iterative Proportional Fitting
        # We want P = M.T @ w = target
        target = k / L
        w = np.ones(L) / L
        for _ in range(200):
            current_prob = w @ nn_matrix # P_j = sum_i w_i M_ij
            correction = target / (current_prob + 1e-10)
            # Update w_i: increase if its neighbors are under-covered
            w = w * (nn_matrix @ correction) / k
            w /= w.sum()
        weights = w
    elif strategy == 'spatial':
        # Pick a random point in box, find nearest residue
        min_p = coords.min(axis=0)
        max_p = coords.max(axis=0)
        weights = np.zeros(L) # This is hard to represent as fixed weights
        for _ in range(num_trials):
            pt = np.random.uniform(min_p, max_p)
            d = np.linalg.norm(coords - pt, axis=1)
            seed = np.argmin(d)
            indices = np.where(nn_matrix[seed])[0]
            inclusion_counts[indices] += 1
        return inclusion_counts / num_trials

    for _ in range(num_trials):
        seed = np.random.choice(L, p=weights)
        indices = np.where(nn_matrix[seed])[0]
        inclusion_counts[indices] += 1
        
    return inclusion_counts / num_trials

# Dummy protein (a line of points)
L = 50
coords = np.zeros((L, 3))
coords[:, 0] = np.arange(L)
k = 10

print("Strategy: Uniform")
probs = simulate_inclusion(coords, k, strategy='uniform')
print(f"Mean: {probs.mean():.4f}, Std: {probs.std():.4f}, Min: {probs.min():.4f}, Max: {probs.max():.4f}")

print("\nStrategy: Balanced (Heuristic)")
probs = simulate_inclusion(coords, k, strategy='balanced')
print(f"Mean: {probs.mean():.4f}, Std: {probs.std():.4f}, Min: {probs.min():.4f}, Max: {probs.max():.4f}")

print("\nStrategy: IPF (Optimal)")
probs = simulate_inclusion(coords, k, strategy='ipf')
print(f"Mean: {probs.mean():.4f}, Std: {probs.std():.4f}, Min: {probs.min():.4f}, Max: {probs.max():.4f}")
