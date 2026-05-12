"""
Geometry classifier — the core novel contribution.

For each SAE feature cluster we measure how "quantum-like" the geometry of
the cluster's decoder directions is, using von Neumann entropy of the
normalised decoder covariance matrix as a density-matrix proxy.

Background (Riechers et al. 2025)
----------------------------------
A classical probability distribution over K outcomes is a convex combination
of K pure states — its density matrix has rank 1 (one dominant eigenvalue,
all others zero). A mixed quantum state has a density matrix with many
non-negligible eigenvalues — its von Neumann entropy is higher.

When we treat the covariance of a cluster's decoder directions as a density
matrix ρ (normalised to unit trace), we can ask:

    S(ρ) = -Σ λ_i log(λ_i)    (von Neumann entropy)

A cluster whose decoder directions all point in roughly the same direction
(classical, low-rank) has low entropy. A cluster with many independent
directions (quantum-like, high-rank) has high entropy close to log(rank).

Normalising by log(rank) gives a score in [0, 1]:
    0 → pure classical state (rank-1 decoder covariance)
    1 → maximally mixed quantum state (uniform eigenspectrum)

This is fast (one eigendecomposition per cluster), robust (no iterative
optimisation), and grounded in quantum information theory.
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from config import RESULTS_DIR


# ---------------------------------------------------------------------------
# Core quantum-ness measure: von Neumann entropy of decoder covariance
# ---------------------------------------------------------------------------

def _von_neumann_entropy_score(decoder_vectors: np.ndarray) -> tuple[float, float]:
    """
    Compute the normalised von Neumann entropy of the decoder covariance matrix.

    Parameters
    ----------
    decoder_vectors : (n_features, d_model)  float32

    Returns
    -------
    quantum_ness_score : float in [0, 1]
        von_neumann_entropy(ρ) / log(rank(ρ))
        where ρ = (D^T D / n_features) normalised to unit trace.
    min_eigenvalue : float
        Smallest eigenvalue of ρ (should be ≥ 0 for a valid density matrix;
        kept for backward-compatibility with downstream code).
    """
    n, d = decoder_vectors.shape

    # Step 1 — build the covariance matrix of the decoder directions
    # C = D^T D / n  →  shape (d_model, d_model)
    # We don't centre: decoder directions are not activations, and centring
    # would remove the mean direction which carries signal.
    C = (decoder_vectors.T @ decoder_vectors) / n   # (d_model, d_model)

    # Step 2 — normalise to unit trace (density-matrix convention)
    tr = np.trace(C)
    if tr < 1e-10:
        return 0.0, 0.0
    rho = C / tr   # (d_model, d_model), trace = 1

    # Step 3 — eigendecomposition (eigvalsh exploits symmetry, returns sorted reals)
    eigvals = np.linalg.eigvalsh(rho)   # (d_model,) ascending order

    min_eigenvalue = float(eigvals[0])

    # Step 4 — von Neumann entropy over eigenvalues above the numerical noise floor
    # λ_i < 1e-6 contribute ~0 to -λ log λ and arise from floating-point noise
    positive = eigvals[eigvals > 1e-6]
    rank = len(positive)

    if rank <= 1:
        # Rank-1 = pure classical state, entropy = 0
        return 0.0, min_eigenvalue

    entropy = -float(np.sum(positive * np.log(positive)))

    # Step 5 — normalise by log(rank) so score is in [0, 1]
    # log(rank) is the maximum entropy achievable for this rank
    quantum_ness_score = entropy / np.log(rank)

    # Clamp to [0, 1] to absorb any floating-point overshoot
    quantum_ness_score = float(np.clip(quantum_ness_score, 0.0, 1.0))

    return quantum_ness_score, min_eigenvalue


# ---------------------------------------------------------------------------
# Per-cluster geometry classification
# ---------------------------------------------------------------------------

def classify_cluster_geometry(
    cluster: dict,
    feature_activations: torch.Tensor,  # kept for API compatibility, not used
    n_archetypes: int = 4,              # kept for API compatibility, not used
    n_iters: int = 500,                 # kept for API compatibility, not used
) -> dict:
    """
    Classify the geometry of a single cluster via von Neumann entropy.

    The cluster dict must contain:
        cluster_id      : int
        feature_indices : LongTensor (cluster_size,)
        decoder_vectors : FloatTensor (cluster_size, d_model)
        subspace_basis  : FloatTensor (subspace_dim, d_model)

    Returns a dict with:
        cluster_id         : int
        n_features         : int
        quantum_ness_score : float in [0, 1]
        min_eigenvalue     : float
        classification     : "classical" | "ambiguous" | "quantum"
    """
    dec_vecs = cluster["decoder_vectors"].numpy().astype(np.float32)
    # (cluster_size, d_model)

    if dec_vecs.shape[0] < 2:
        return _degenerate_result(cluster)

    quantum_ness_score, min_eigenvalue = _von_neumann_entropy_score(dec_vecs)

    # ---- Classification ----
    # classical : low entropy → one dominant decoder direction → classical pure state
    # quantum   : high entropy → many comparable directions → quantum mixed state
    if quantum_ness_score < 0.3:
        classification = "classical"
    elif quantum_ness_score >= 0.6:
        classification = "quantum"
    else:
        classification = "ambiguous"

    return {
        "cluster_id": int(cluster["cluster_id"]),
        "n_features": int(dec_vecs.shape[0]),
        "quantum_ness_score": quantum_ness_score,
        "min_eigenvalue": min_eigenvalue,
        "classification": classification,
    }


def _degenerate_result(cluster: dict) -> dict:
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "n_features": int(len(cluster["feature_indices"])),
        "quantum_ness_score": 0.0,
        "min_eigenvalue": 0.0,
        "classification": "ambiguous",
    }


# ---------------------------------------------------------------------------
# Run all clusters
# ---------------------------------------------------------------------------

def classify_all_clusters(
    model_cfg: dict,
    clusters: list[dict],
    feature_activations: torch.Tensor,  # (n_prompts, n_features)
    results_dir: str = RESULTS_DIR,
    force_reclassify: bool = False,
) -> list[dict]:
    """
    Classify geometry for every cluster and save results.
    """
    save_path = os.path.join(results_dir, model_cfg["name"], "geometry.pt")
    if not force_reclassify and os.path.exists(save_path):
        print(f"[geometry] Loading geometry results from {save_path}")
        geo = torch.load(save_path, weights_only=False)
        print(f"[geometry] Loaded {len(geo)} cluster geometry results")
        return geo

    print(f"[geometry] Classifying {len(clusters)} clusters ...")
    print(f"[geometry] Method: von Neumann entropy of decoder covariance")

    results = []
    class_counts = {"classical": 0, "quantum": 0, "ambiguous": 0}

    for cluster in tqdm(clusters, desc="Classifying clusters"):
        geo = classify_cluster_geometry(cluster, feature_activations)
        results.append(geo)
        class_counts[geo["classification"]] += 1

    print(f"[geometry] Classification summary: {class_counts}")
    q_scores = [r["quantum_ness_score"] for r in results]
    print(f"[geometry] Quantum-ness scores: "
          f"mean={np.mean(q_scores):.3f}, "
          f"min={np.min(q_scores):.3f}, "
          f"max={np.max(q_scores):.3f}")

    torch.save(results, save_path)
    print(f"[geometry] Saved geometry results to {save_path}")
    return results


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from config import get_active_models

    print("=== Smoke test: geometry_classifier (von Neumann entropy) ===")
    rng = np.random.default_rng(0)
    n_prompts, n_features, d_model = 200, 500, 64

    feature_activations = torch.zeros(n_prompts, n_features)  # not used by new method

    def _make_cluster(cid, n_feat, rank, d=d_model):
        """Make a cluster whose decoder has a given approximate rank."""
        # Build a (d, d) orthonormal basis then scale the first `rank` columns
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
        s = np.ones(d, dtype=np.float32) * 1e-4
        s[:rank] = 1.0                          # `rank` strong directions
        # Project n_feat random unit vectors onto the scaled basis
        dec = rng.standard_normal((n_feat, d)).astype(np.float32) @ (Q * s[None, :])
        dec /= np.linalg.norm(dec, axis=1, keepdims=True) + 1e-8
        _, _, Vt = np.linalg.svd(dec, full_matrices=False)
        return {
            "cluster_id": cid,
            "feature_indices": torch.arange(n_feat),
            "decoder_vectors": torch.tensor(dec),
            "subspace_basis": torch.tensor(Vt[:8]),
        }

    # Low rank → classical; high rank → quantum
    for rank, label in [(1, "rank-1 (expect classical)"),
                        (4, "rank-4 (expect ambiguous)"),
                        (32, "rank-32 (expect quantum)")]:
        cl = _make_cluster(rank, 60, rank)
        r = classify_cluster_geometry(cl, feature_activations)
        print(f"  {label:35s}  "
              f"score={r['quantum_ness_score']:.4f}  "
              f"class={r['classification']}")
        assert 0.0 <= r["quantum_ness_score"] <= 1.0, "Score out of [0, 1]"

    # Check scores are spread across the range (not all identical).
    # Mix rank-1 clusters (score ≈ 0) with high-rank clusters (score ≈ 1)
    # to guarantee meaningful variance.
    scores = []
    for cid in range(10):
        rank = 1 if cid < 3 else rng.integers(4, 30)  # 3 classical, 7 quantum-ish
        cl = _make_cluster(cid, 40, int(rank))
        r = classify_cluster_geometry(cl, feature_activations)
        scores.append(r["quantum_ness_score"])
    print(f"\n  10-cluster spread: min={min(scores):.3f}  max={max(scores):.3f}  "
          f"std={float(np.std(scores)):.3f}")
    assert max(scores) - min(scores) > 0.3, \
        f"Score range too narrow ({max(scores)-min(scores):.3f}) — check implementation"

    # Test full pipeline
    models = get_active_models()
    if models:
        m = models[0]
        cl = _make_cluster(0, 40, 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, m["name"]), exist_ok=True)
            geo = classify_all_clusters(m, [cl], feature_activations, results_dir=tmpdir)
            print(f"\n  classify_all_clusters: {len(geo)} result(s)")

    print("[geometry_classifier] Smoke test passed.")
