"""
K-subspace clustering of SAE decoder vectors.

Each SAE feature has a corresponding decoder column — a direction in residual-stream
space. We cluster those directions into N_CLUSTERS groups using k-subspace clustering:

  1. Warm-start with k-means on the (L2-normalized) decoder vectors.
  2. For each cluster, fit an orthonormal subspace via SVD.
  3. Re-assign every feature to the subspace it projects onto most strongly.
  4. Repeat until convergence (or for a fixed number of iterations).

Clusters with fewer than MIN_CLUSTER_SIZE features are discarded.

Output saved to results/{model_name}/clusters.pt as a list of dicts:
  {
    cluster_id       : int,
    feature_indices  : LongTensor  (cluster_size,),
    decoder_vectors  : FloatTensor (cluster_size, d_model),
    subspace_basis   : FloatTensor (subspace_dim, d_model),   # orthonormal rows
  }
"""

import os
import sys
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import (
    N_CLUSTERS,
    MIN_CLUSTER_SIZE,
    RESULTS_DIR,
)

# Maximum subspace dimension per cluster (kept small for geometry computations)
SUBSPACE_DIM = 10
# K-subspace iteration count
MAX_ITER = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_save_path(model_name: str, results_dir: str) -> str:
    path = os.path.join(results_dir, model_name)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "clusters.pt")


def _load_decoder_matrix(model_cfg: dict, results_dir: str) -> torch.Tensor:
    """
    Load the SAE decoder weight matrix from disk.

    SAELens stores the decoder as W_dec with shape (d_sae, d_model), i.e.
    one row per feature. We transpose to get (d_model, d_sae) and then
    return the columns so that decoder_vectors[i] is the direction for feature i.

    We save the decoder alongside activations; if it isn't there yet we load
    the SAE fresh from SAELens (requires internet / cached weights).
    """
    decoder_path = os.path.join(results_dir, model_cfg["name"], "decoder.pt")

    if os.path.exists(decoder_path):
        dec = torch.load(decoder_path, weights_only=True)
        print(f"[clustering] Loaded decoder from {decoder_path}, shape: {dec.shape}")
        return dec.float()

    print(f"[clustering] Decoder not cached — loading SAE from SAELens ...")
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(
        release=model_cfg["sae_release"],
        sae_id=model_cfg["sae_id"],
    )
    # W_dec: (d_sae, d_model) → we want (d_sae, d_model) with rows = features
    dec = sae.W_dec.detach().float()   # (n_features, d_model)
    torch.save(dec, decoder_path)
    print(f"[clustering] Saved decoder to {decoder_path}, shape: {dec.shape}")
    return dec


def _fit_subspace_basis(vectors: np.ndarray, max_dim: int = SUBSPACE_DIM) -> np.ndarray:
    """
    Compute an orthonormal basis for the subspace spanned by `vectors` via SVD.

    Returns rows of shape (min(max_dim, rank), d_model).
    """
    if vectors.shape[0] == 0:
        return np.zeros((1, vectors.shape[1]), dtype=np.float32)
    # Center the vectors (optional but improves subspace quality)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # Thin SVD: U (n, k), S (k,), Vt (k, d_model)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        _, _, Vt = np.linalg.svd(centered + 1e-8 * np.random.randn(*centered.shape), full_matrices=False)
    k = min(max_dim, Vt.shape[0], vectors.shape[0])
    return Vt[:k].astype(np.float32)   # (k, d_model)


def _projection_distances(
    vectors: np.ndarray,               # (n_features, d_model)
    bases: list[np.ndarray],           # list of (k_i, d_model) orthonormal bases
) -> np.ndarray:
    """
    For each vector, compute its squared distance to each subspace.

    distance(v, S) = ||v||^2 - ||P_S v||^2
    where P_S v = B^T (B v^T) for orthonormal rows B.

    Returns (n_features, n_clusters).
    """
    n = vectors.shape[0]
    k = len(bases)
    dist = np.zeros((n, k), dtype=np.float32)
    v_norm_sq = (vectors ** 2).sum(axis=1)  # (n,)

    for j, B in enumerate(bases):
        # B: (dim_j, d_model)
        proj = vectors @ B.T    # (n, dim_j)
        proj_norm_sq = (proj ** 2).sum(axis=1)  # (n,)
        dist[:, j] = v_norm_sq - proj_norm_sq

    return dist


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def cluster_decoder_vectors(
    model_cfg: dict,
    results_dir: str = RESULTS_DIR,
    n_clusters: int = N_CLUSTERS,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    force_recluster: bool = False,
) -> list[dict]:
    """
    Cluster SAE decoder vectors into k subspaces.

    Returns a list of cluster dicts (only clusters ≥ min_cluster_size).
    """
    save_path = _get_save_path(model_cfg["name"], results_dir)
    if not force_recluster and os.path.exists(save_path):
        print(f"[clustering] Loading clusters from {save_path}")
        clusters = torch.load(save_path, weights_only=False)
        print(f"[clustering] Loaded {len(clusters)} clusters")
        return clusters

    decoder = _load_decoder_matrix(model_cfg, results_dir)  # (n_features, d_model)
    print(f"[clustering] Decoder shape: {decoder.shape}")

    n_features, d_model = decoder.shape
    dec_np = decoder.numpy()

    # L2-normalize for angle-based clustering
    norms = np.linalg.norm(dec_np, axis=1, keepdims=True).clip(min=1e-8)
    dec_normalized = dec_np / norms    # (n_features, d_model)

    # ---- Warm-start: k-means on normalized vectors ----
    print(f"[clustering] K-means warm-start with k={n_clusters} ...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42, max_iter=100)
    labels = kmeans.fit_predict(dec_normalized)
    print(f"[clustering] K-means done. Label counts: "
          f"min={np.bincount(labels).min()}, max={np.bincount(labels).max()}")

    # ---- K-subspace refinement ----
    for iteration in tqdm(range(MAX_ITER), desc="K-subspace refinement"):
        # Build subspace bases from current assignment
        bases = []
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() < 2:
                # Degenerate cluster: use random unit vector
                b = np.random.randn(1, d_model).astype(np.float32)
                b /= np.linalg.norm(b, axis=1, keepdims=True)
                bases.append(b)
            else:
                bases.append(_fit_subspace_basis(dec_normalized[mask]))

        # Re-assign to closest subspace
        dist = _projection_distances(dec_normalized, bases)   # (n_features, n_clusters)
        new_labels = dist.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            print(f"[clustering] Converged at iteration {iteration+1}")
            break
        labels = new_labels

    # ---- Build cluster dicts ----
    clusters = []
    kept = 0
    for c in tqdm(range(n_clusters), desc="Building cluster dicts"):
        indices = np.where(labels == c)[0]
        if len(indices) < min_cluster_size:
            continue

        vecs = dec_np[indices].astype(np.float32)          # (size, d_model)
        basis = _fit_subspace_basis(dec_normalized[indices]) # (k, d_model)

        clusters.append({
            "cluster_id": c,
            "feature_indices": torch.tensor(indices, dtype=torch.long),
            "decoder_vectors": torch.tensor(vecs, dtype=torch.float32),
            "subspace_basis": torch.tensor(basis, dtype=torch.float32),
        })
        kept += 1

    print(f"[clustering] Kept {kept} clusters (discarded "
          f"{n_clusters - kept} below min_size={min_cluster_size})")

    for cl in clusters[:3]:
        print(f"  cluster {cl['cluster_id']}: "
              f"n_features={len(cl['feature_indices'])}, "
              f"subspace_dim={cl['subspace_basis'].shape[0]}, "
              f"d_model={cl['subspace_basis'].shape[1]}")

    torch.save(clusters, save_path)
    print(f"[clustering] Saved clusters to {save_path}")
    return clusters


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from config import get_active_models

    models = get_active_models()
    if not models:
        print("No active models.")
        sys.exit(1)

    m = models[0]
    print(f"=== Smoke test: {m['name']} ===")

    # Build a fake decoder and save it so we don't need to download the model
    d_model = 256
    n_features = 500
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, m["name"])
        os.makedirs(model_dir, exist_ok=True)

        fake_decoder = torch.randn(n_features, d_model)
        torch.save(fake_decoder, os.path.join(model_dir, "decoder.pt"))

        clusters = cluster_decoder_vectors(
            m,
            results_dir=tmpdir,
            n_clusters=10,
            min_cluster_size=5,
        )
        print(f"Got {len(clusters)} clusters")
        for cl in clusters[:3]:
            print(f"  cluster {cl['cluster_id']}: "
                  f"n_features={len(cl['feature_indices'])}, "
                  f"basis={cl['subspace_basis'].shape}")

    print("[clustering] Smoke test passed.")
