"""
Geometry classifier — the core novel contribution.

For each feature cluster we ask: does the empirical distribution of activations
in this cluster live inside a classical simplex (classical probability simplex
= convex hull of a finite set of archetypes), or does it require going *outside*
that simplex (quantum / post-quantum geometry)?

Background (Riechers et al. 2025)
----------------------------------
A classical probability distribution over K outcomes is a convex combination of
K pure states — it lives in the (K-1)-simplex. Quantum states generalise this:
a density matrix ρ is PSD with trace 1, but when you project onto an
over-complete frame the resulting "quasi-probabilities" (Wigner / Husimi) can be
negative. A model whose internal belief state has this property will produce
activations that, when projected onto the feature subspace, cannot be explained
as convex combinations of the archetypes. Points outside the simplex ↔ quantum
geometry.

Our test
---------
1.  Fit archetypal analysis (AA-Net variant) to the cluster's *decoder vectors*
    to obtain K simplex vertices (archetypes) in residual-stream space.

2.  For each observed feature-activation-weighted representation:
      - Project the activation-weighted residual-stream vector onto the cluster
        subspace (low-dim embedding).
      - Compute barycentric coordinates w.r.t. the fitted simplex vertices.
      - Classical signature: all coordinates ≥ 0.
      - Quantum signature: some coordinates < 0 (point is outside the simplex).

3.  Quantum-ness score = fraction of activations with ≥ 1 negative barycentric
    coordinate.

4.  Additionally fit the empirical "density matrix":
      - Construct the covariance matrix of projected activations.
      - Normalise to unit trace.
      - Minimum eigenvalue < -0.05 is additional evidence of quantum geometry
        (a PSD density matrix cannot have negative eigenvalues; deviation here
        signals non-classical geometry in the cluster).

5.  Classify:
      - classical : quantum-ness < 0.10 AND min_eigenvalue > -0.05
      - quantum   : quantum-ness > 0.30 OR  min_eigenvalue < -0.10
      - ambiguous : everything else
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from config import (
    AANET_ITERS,
    AANET_N_ARCHETYPES,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Step 1 — Archetypal Analysis (AA-Net variant)
# ---------------------------------------------------------------------------

def _simplex_project_rows(M: np.ndarray) -> np.ndarray:
    """
    Project each *row* of M onto the probability simplex:
      min ||m - x||  s.t. x >= 0, sum(x) = 1

    Uses the classic O(n log n) algorithm (Duchi et al. 2008).
    """
    n, d = M.shape
    out = np.empty_like(M)
    for i in range(n):
        u = np.sort(M[i])[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, d + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        out[i] = np.maximum(M[i] - theta, 0.0)
    return out


def fit_archetypes(
    X: np.ndarray,
    n_archetypes: int = AANET_N_ARCHETYPES,
    n_iters: int = AANET_ITERS,
    lr: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit archetypal analysis to X (n_points, d) using projected gradient descent.

    Model:
        A = X @ B          (archetypes as convex combinations of data points)
        X_hat = S @ A      (data reconstructed as convex combinations of archetypes)

    where B (n_points, K) and S (n_points, K) both have rows on the simplex.

    Returns:
        archetypes : (K, d)          — the K archetype vectors
        S          : (n_points, K)   — reconstruction weights
        B          : (n_points, K)   — archetype mixture weights
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = n_archetypes

    # Initialise B uniformly on the simplex
    B = rng.dirichlet(np.ones(n), size=K).T   # (n, K)  — each column is an archetype mix
    B = B / B.sum(axis=0, keepdims=True)       # normalise columns

    # Initialise S uniformly
    S = rng.dirichlet(np.ones(K), size=n)      # (n, K)

    X_norm_sq = (X ** 2).sum()

    for it in range(n_iters):
        # Dimensions: X (n,d), B (n,K) columns-sum-to-1, S (n,K) rows-sum-to-1
        #   A = B^T X  →  (K, d)   archetype vectors as rows
        #   X_hat = S A  →  (n, d)

        # ---- Update S (fix B, update S via projected GD) ----
        A = B.T @ X                             # (K, d)
        X_hat = S @ A                           # (n, d)
        R = X_hat - X                           # residual (n, d)
        grad_S = R @ A.T                        # (n, K)
        S = S - lr * grad_S
        S = _simplex_project_rows(S)            # project rows onto probability simplex

        # ---- Update B (fix S, update B via projected GD) ----
        # d(loss)/d(B) = X @ (R^T @ S)  shape (n, K)
        # derivation: loss = ||S B^T X - X||^2; d/dB via chain rule through C=B^T
        A = B.T @ X
        X_hat = S @ A
        R = X_hat - X                           # (n, d)
        # R^T @ S: (d,n)(n,K) = (d,K);  X @ (...): (n,d)(d,K) = (n,K)
        grad_B = X @ (R.T @ S)                  # (n, K)
        B = B - lr * grad_B
        # Project each column of B onto the probability simplex (each col sums to 1)
        B = _simplex_project_rows(B.T).T        # (n, K)

    A_final = B.T @ X    # (K, d) — final archetypes
    return A_final, S, B


# ---------------------------------------------------------------------------
# Step 2 — Barycentric coordinates
# ---------------------------------------------------------------------------

def _barycentric_coords(
    points: np.ndarray,     # (n_points, d)
    vertices: np.ndarray,   # (K, d) — simplex vertices
) -> np.ndarray:
    """
    Compute barycentric coordinates of `points` w.r.t. a simplex defined by
    `vertices` using least-squares with the constraint that coordinates sum to 1.

    For a K-simplex in d-dimensional space (K ≤ d+1):
        p = sum_k lambda_k * v_k,   sum_k lambda_k = 1

    We solve the (K-1) × n system after subtracting v_0:
        p - v_0 = sum_{k=1}^{K-1} lambda_k * (v_k - v_0)

    Returns: (n_points, K) — lambda_k values (can be negative).
    """
    K, d = vertices.shape
    n = points.shape[0]

    if K == 1:
        # Degenerate simplex — every point maps to weight 1
        return np.ones((n, 1), dtype=np.float32)

    # Shift: work in the affine subspace relative to vertex 0
    v0 = vertices[0]                           # (d,)
    T = (vertices[1:] - v0).T                 # (d, K-1)

    # Solve T @ lambdas = (points - v0).T  in least-squares sense
    pts_shifted = (points - v0).T              # (d, n)
    # lstsq expects (m, n) where m = equations, n = variables
    lambdas_rest, _, _, _ = np.linalg.lstsq(T, pts_shifted, rcond=None)
    # lambdas_rest: (K-1, n)

    lambda0 = 1.0 - lambdas_rest.sum(axis=0)  # (n,)
    coords = np.vstack([lambda0, lambdas_rest]).T   # (n, K)
    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — Density-matrix eigenvalue test
# ---------------------------------------------------------------------------

def _density_matrix_min_eigenvalue(projected: np.ndarray) -> float:
    """
    Construct the empirical covariance of `projected` (n_points, dim),
    normalise to unit trace (treating it as a density matrix ρ), and return
    the minimum eigenvalue.

    For a valid classical density matrix, all eigenvalues ≥ 0.
    Negative eigenvalues signal non-classical (quantum-like) geometry.

    Note: a sample covariance is always PSD by construction, but when we
    *normalise* by something other than the standard Bessel-corrected denominator
    (e.g. by total activation magnitude rather than number of samples) or when
    we work in an *over-complete* projected basis, rounding / numerical artefacts
    can produce small negative eigenvalues. The threshold (-0.05) filters genuine
    signal from numerical noise.
    """
    n, dim = projected.shape
    if n < 2:
        return 0.0

    centered = projected - projected.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(n - 1, 1)   # (dim, dim)

    # Normalise to unit trace (density-matrix convention)
    tr = np.trace(cov)
    if abs(tr) < 1e-10:
        return 0.0
    rho = cov / tr

    eigvals = np.linalg.eigvalsh(rho)
    return float(eigvals.min())


# ---------------------------------------------------------------------------
# Step 4 — Per-cluster geometry classification
# ---------------------------------------------------------------------------

def classify_cluster_geometry(
    cluster: dict,
    feature_activations: torch.Tensor,   # (n_prompts, n_features) — from sae_extractor
    n_archetypes: int = AANET_N_ARCHETYPES,
    n_iters: int = AANET_ITERS,
) -> dict:
    """
    Classify the geometry of a single cluster.

    The cluster dict has:
        feature_indices  : (cluster_size,)
        decoder_vectors  : (cluster_size, d_model)
        subspace_basis   : (subspace_dim, d_model)

    feature_activations shape: (n_prompts, n_features_total)

    Returns a dict with:
        cluster_id         : int
        n_features         : int
        archetypes         : (K, subspace_dim)
        quantum_ness_score : float  — fraction of activations outside simplex
        min_eigenvalue     : float  — minimum eigenvalue of normalised covariance
        classification     : str    — "classical" | "quantum" | "ambiguous"
    """
    feat_idx = cluster["feature_indices"].numpy()       # (cluster_size,)
    basis = cluster["subspace_basis"].numpy()           # (subspace_dim, d_model)
    dec_vecs = cluster["decoder_vectors"].numpy()       # (cluster_size, d_model)

    # ---- Project decoder vectors into subspace ----
    # basis rows are orthonormal ⟹ projection = dec_vecs @ basis.T
    dec_projected = dec_vecs @ basis.T                  # (cluster_size, subspace_dim)

    # ---- Fit archetypes on projected decoder vectors ----
    # We find the archetypal structure of the *feature directions* in this cluster.
    K = min(n_archetypes, dec_projected.shape[0] - 1)
    if K < 2:
        return _degenerate_result(cluster)

    archetypes, _, _ = fit_archetypes(
        dec_projected.astype(np.float32),
        n_archetypes=K,
        n_iters=n_iters,
    )
    # archetypes: (K, subspace_dim)

    # ---- Extract per-prompt activation-weighted representations ----
    # For each prompt, take the SAE activations for features in this cluster
    # and project the activation-weighted sum of decoder vectors into the subspace.
    #
    # Interpretation: this gives us the "cluster's contribution to the residual
    # stream for this prompt" expressed in the cluster's own subspace coordinates.
    feat_acts_cluster = feature_activations[:, feat_idx].numpy().astype(np.float32)
    # (n_prompts, cluster_size)

    # Weighted sum of decoder vectors per prompt
    # shape: (n_prompts, d_model)
    weighted_dec = feat_acts_cluster @ dec_vecs         # (n_prompts, d_model)

    # Project into subspace: (n_prompts, subspace_dim)
    projected_acts = weighted_dec @ basis.T             # (n_prompts, subspace_dim)

    # Filter to prompts that actually activated this cluster
    activation_magnitudes = np.linalg.norm(projected_acts, axis=1)
    active_mask = activation_magnitudes > 1e-6
    if active_mask.sum() < 5:
        return _degenerate_result(cluster)

    projected_active = projected_acts[active_mask]      # (n_active, subspace_dim)

    # ---- Classical vs quantum test via barycentric coordinates ----
    #
    # Classical probability theory: any mixed state is a convex combination of
    # pure states (archetypes). Barycentric coords must all be ≥ 0, and a point
    # is fully inside the simplex iff every coord is non-negative.
    #
    # Quantum theory allows states outside the classical simplex — the geometry
    # of quantum state space has curved boundaries (e.g. Bloch sphere), not flat
    # simplex faces. Points outside the simplex have at least one negative coord.
    #
    # We compute a *continuous* quantum-ness score (rather than a binary flag) by
    # measuring how far each activation sits outside the simplex:
    #
    #   violation_i = sum of |negative coords| for activation i
    #                 ----------------------------------------
    #                 sum of |all coords| + 1e-8
    #
    # This is 0 when the point is perfectly inside the simplex (all coords ≥ 0),
    # and approaches 1 when the point is almost entirely outside (large negative
    # coords dominate the sum). Averaging over all active prompts gives a score
    # in [0, 1]: 0 = perfectly classical, 1 = maximally quantum.
    coords = _barycentric_coords(projected_active, archetypes)
    # coords: (n_active, K) — can be negative for out-of-simplex points

    neg_mass = np.abs(np.minimum(coords, 0)).sum(axis=1)   # (n_active,) sum of |negative coords|
    total_mass = np.abs(coords).sum(axis=1)                # (n_active,) sum of |all coords|
    per_activation_violation = neg_mass / (total_mass + 1e-8)  # (n_active,) in [0, 1]
    quantum_ness_score = float(per_activation_violation.mean())

    # ---- Density-matrix eigenvalue test ----
    #
    # Normalise the empirical covariance to unit trace and check its minimum
    # eigenvalue. A classical covariance is always PSD (all eigenvalues ≥ 0).
    # Significant negative eigenvalues suggest a curved (quantum) manifold.
    min_eig = _density_matrix_min_eigenvalue(projected_active)

    # ---- Classification decision ----
    if quantum_ness_score < 0.15:
        classification = "classical"
    elif quantum_ness_score >= 0.40:
        classification = "quantum"
    else:
        classification = "ambiguous"

    return {
        "cluster_id": int(cluster["cluster_id"]),
        "n_features": len(feat_idx),
        "archetypes": archetypes,                # (K, subspace_dim)
        "quantum_ness_score": quantum_ness_score,
        "min_eigenvalue": min_eig,
        "classification": classification,
    }


def _degenerate_result(cluster: dict) -> dict:
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "n_features": len(cluster["feature_indices"]),
        "archetypes": None,
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
    feature_activations: torch.Tensor,   # (n_prompts, n_features)
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
    print(f"[geometry] feature_activations shape: {feature_activations.shape}")

    results = []
    class_counts = {"classical": 0, "quantum": 0, "ambiguous": 0}

    for cluster in tqdm(clusters, desc="Classifying clusters"):
        geo = classify_cluster_geometry(
            cluster,
            feature_activations,
        )
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

    print("=== Smoke test: geometry_classifier ===")
    rng = np.random.default_rng(0)

    # Fake data: 200 prompts, 500 features, d_model=64
    n_prompts, n_features, d_model = 200, 500, 64
    feature_activations = torch.from_numpy(
        rng.exponential(size=(n_prompts, n_features)).astype(np.float32)
    )

    # Fake cluster
    cluster_size = 40
    feat_idx = np.arange(cluster_size)
    dec_vecs = rng.standard_normal((cluster_size, d_model)).astype(np.float32)
    dec_vecs /= np.linalg.norm(dec_vecs, axis=1, keepdims=True)

    # Compute subspace basis
    _, _, Vt = np.linalg.svd(dec_vecs, full_matrices=False)
    basis = Vt[:8]

    cluster = {
        "cluster_id": 0,
        "feature_indices": torch.tensor(feat_idx),
        "decoder_vectors": torch.tensor(dec_vecs),
        "subspace_basis": torch.tensor(basis),
    }

    result = classify_cluster_geometry(cluster, feature_activations, n_iters=50)
    print(f"  cluster_id:        {result['cluster_id']}")
    print(f"  n_features:        {result['n_features']}")
    print(f"  quantum_ness_score:{result['quantum_ness_score']:.4f}")
    print(f"  min_eigenvalue:    {result['min_eigenvalue']:.4f}")
    print(f"  classification:    {result['classification']}")

    # Test with multiple clusters
    models = get_active_models()
    if models:
        m = models[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, m["name"]), exist_ok=True)
            clusters = [cluster]
            geo = classify_all_clusters(m, clusters, feature_activations, results_dir=tmpdir)
            print(f"\nAll-cluster results: {len(geo)} entries")

    print("[geometry_classifier] Smoke test passed.")
