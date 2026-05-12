"""
Geometry classifier — the core novel contribution.

We test whether a cluster's activation trajectory across prompts requires
going outside the classical probability simplex to be explained.

Theoretical grounding (Riechers et al. 2025)
---------------------------------------------
SAEs assume every hidden state decomposes as a non-negative sparse linear
combination of decoder directions: x ≈ Σ_i s_i d_i, with s_i ≥ 0 (the ReLU
non-linearity enforces this in practice). This is precisely the classical-mixture
assumption: each prompt's state is a convex combination of a finite set of
"pure" directions (the archetypes).

Quantum geometry means the actual state space of the cluster is not a simplex
but a curved convex body (like a Bloch sphere). Points on such a body cannot
always be written as convex combinations of a small set of extremal points — the
unconstrained least-squares mixing weights (S_free) must go negative to fit the
data. The gap between constrained-simplex FVU (classical) and unconstrained FVU
(quantum) directly measures this deficit.

Algorithm per cluster
---------------------
1.  Project activation trajectory into the cluster's low-dimensional subspace.
2.  Fit K=4 simplex archetypes to the trajectory via farthest-point init +
    coordinate descent (constrained NNLS per prompt, unconstrained lstsq for A).
3.  Solve the SAME system without non-negativity constraints (S_free).
4.  quantum_ness_score = (classical_fvu - quantum_fvu) / (classical_fvu + 1e-8)
    — how much extra variance the simplex constraint forces us to leave unexplained.
5.  negative_weight_fraction — fraction of prompts where S_free has any weight < -0.05
    (secondary diagnostic: direct evidence of out-of-simplex activations).
"""

import os
import numpy as np
import torch
from scipy.optimize import nnls
from tqdm import tqdm

from config import RESULTS_DIR

# Fixed number of archetypes — enough to capture a simplex in low-dim subspace
K_ARCHETYPES = 4
# Coordinate-descent iterations for simplex fitting
AA_ITERS = 100

# Module-level counter used to limit debug prints to the first 5 clusters
_DEBUG_FVU_PRINTS = 0


# ---------------------------------------------------------------------------
# Farthest-point initialisation for archetypes
# ---------------------------------------------------------------------------

def _farthest_point_init(X: np.ndarray, K: int) -> np.ndarray:
    """
    Greedy farthest-point initialisation: pick K rows from X that are
    maximally spread out.  More stable than random initialisation for small K.

    Returns archetype matrix A of shape (K, d).
    """
    n, d = X.shape
    mean = X.mean(axis=0)
    # First archetype: point furthest from the mean
    dists = np.sum((X - mean) ** 2, axis=1)   # (n,)
    chosen = [int(np.argmax(dists))]

    for _ in range(K - 1):
        # Distance from each point to its nearest chosen archetype
        min_d = np.full(n, np.inf)
        for idx in chosen:
            d_to = np.sum((X - X[idx]) ** 2, axis=1)
            min_d = np.minimum(min_d, d_to)
        chosen.append(int(np.argmax(min_d)))

    return X[chosen].astype(np.float32)   # (K, d)


# ---------------------------------------------------------------------------
# Simplex fitting via coordinate descent
# ---------------------------------------------------------------------------

def _fit_simplex(
    projected: np.ndarray,    # (n_prompts, subspace_dim)  float32
    K: int = K_ARCHETYPES,
    n_iters: int = AA_ITERS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit K archetypal simplex vertices to the activation trajectory.

    Alternates between:
      - Fixing A, solving for each row of S via NNLS then normalising to Σ=1
        (constrained simplex membership: s_i ≥ 0, Σ s_i = 1)
      - Fixing S, updating A via unconstrained least-squares

    Returns
    -------
    A : (K, subspace_dim)  — archetype (simplex vertex) matrix
    S : (n_prompts, K)     — constrained mixing weights, rows sum to 1
    """
    n, dim = projected.shape
    A = _farthest_point_init(projected, K)              # (K, dim)
    S = np.ones((n, K), dtype=np.float32) / K          # uniform init

    for _ in range(n_iters):
        # ---- Fix A, update S (constrained: non-negative + sum-to-1) ----
        # nnls solves min ||A^T s - p||  s.t. s >= 0  for each prompt
        AT = A.T.astype(np.float64)   # nnls wants float64
        for i in range(n):
            w, _ = nnls(AT, projected[i].astype(np.float64))
            total = w.sum()
            S[i] = (w / total).astype(np.float32) if total > 1e-8 \
                   else np.ones(K, dtype=np.float32) / K

        # ---- Fix S, update A (unconstrained least-squares) ----
        # min ||S A - projected||^2  →  A = pinv(S) @ projected
        A_new, _, _, _ = np.linalg.lstsq(S, projected, rcond=None)
        A = A_new.astype(np.float32)                    # (K, dim)

    return A, S


# ---------------------------------------------------------------------------
# Per-cluster geometry classification
# ---------------------------------------------------------------------------

def classify_cluster_geometry(
    cluster: dict,
    feature_activations: torch.Tensor,  # (n_prompts, n_features_total)
    n_archetypes: int = K_ARCHETYPES,   # kept for API compat
    n_iters: int = AA_ITERS,            # kept for API compat
) -> dict:
    """
    Classify the geometry of one cluster by comparing constrained-simplex
    (classical) vs unconstrained (quantum) reconstruction of its activation
    trajectory across prompts.

    Returns a dict with keys:
        cluster_id, n_features,
        quantum_ness_score, negative_weight_fraction,
        classical_fvu, quantum_fvu,
        classification
    """
    feat_idx  = cluster["feature_indices"].numpy()         # (n_feat,)
    dec_vecs  = cluster["decoder_vectors"].numpy().astype(np.float32)   # (n_feat, d_model)
    basis     = cluster["subspace_basis"].numpy().astype(np.float32)    # (sub_dim, d_model)
    sub_dim   = basis.shape[0]

    # Require enough subspace dimensions to fit K archetypes
    if sub_dim < K_ARCHETYPES:
        return _degenerate_result(cluster, reason="subspace_dim < K_ARCHETYPES")

    # ------------------------------------------------------------------
    # Step 1 — activation trajectory in cluster subspace
    # ------------------------------------------------------------------
    # cluster_acts : (n_prompts, n_feat)
    cluster_acts = feature_activations[:, feat_idx].numpy().astype(np.float32)

    # activation-weighted decoder sum per prompt: (n_prompts, d_model)
    act_weighted = cluster_acts @ dec_vecs

    # project onto subspace: (n_prompts, sub_dim)
    projected = act_weighted @ basis.T                     # (n_prompts, sub_dim)

    # Use ALL prompts — no active-mask filtering. Even prompts where the
    # cluster's features didn't activate contribute (as the origin) to the
    # trajectory geometry and inform the simplex fit.
    proj_all = projected.astype(np.float32)               # (n_prompts, sub_dim)

    # ------------------------------------------------------------------
    # Step 2 — fit simplex archetypes (constrained reconstruction)
    # ------------------------------------------------------------------
    A, S = _fit_simplex(proj_all, K=K_ARCHETYPES, n_iters=AA_ITERS)
    # A : (K, sub_dim),  S : (n_prompts, K)  rows sum to 1, all >= 0
    classical_recon = S @ A                                # (n_prompts, sub_dim)

    # ------------------------------------------------------------------
    # Step 3 — unconstrained reconstruction with the same archetypes
    # ------------------------------------------------------------------
    # We want to solve  A^T s_i = proj_i  per prompt (no constraints).
    # Stack: solve in normal-equation form with ridge regularisation to keep
    # things numerically stable when archetypes are nearly collinear.
    #
    #   min_S  || S A - projected ||^2 + λ || S ||^2
    #   →  (A A^T + λ I) S^T = A projected^T
    K = A.shape[0]
    try:
        ATA_reg = (A @ A.T).astype(np.float64) + 1e-4 * np.eye(K, dtype=np.float64)
        ATP = (A @ proj_all.T).astype(np.float64)         # (K, n_prompts)
        S_free = np.linalg.solve(ATA_reg, ATP).T.astype(np.float32)  # (n_prompts, K)
    except np.linalg.LinAlgError as e:
        return _degenerate_result(cluster, reason=f"linalg_error: {e}")

    quantum_recon = S_free @ A                             # (n_prompts, sub_dim)

    # ------------------------------------------------------------------
    # Step 4 — FVU gap = quantum-ness score
    # ------------------------------------------------------------------
    centered   = proj_all - proj_all.mean(axis=0)
    total_var  = float((centered ** 2).sum())

    classical_residual = proj_all - classical_recon
    classical_fvu      = float((classical_residual ** 2).sum()) / (total_var + 1e-8)

    quantum_residual = proj_all - quantum_recon
    quantum_fvu      = float((quantum_residual ** 2).sum()) / (total_var + 1e-8)

    # Debug: show raw FVU numbers for the first few clusters so we can
    # diagnose what magnitude of geometry we're dealing with.
    global _DEBUG_FVU_PRINTS
    if _DEBUG_FVU_PRINTS < 5:
        print(
            f"  [debug] cluster {int(cluster['cluster_id']):3d}  "
            f"total_var={total_var:.6f}  "
            f"classical_fvu={classical_fvu:.6f}  "
            f"quantum_fvu={quantum_fvu:.6f}"
        )
        _DEBUG_FVU_PRINTS += 1

    # Clamp to [0, 1]: the gap cannot be negative (unconstrained ≤ constrained error)
    quantum_ness_score = float(np.clip(
        (classical_fvu - quantum_fvu) / (classical_fvu + 1e-8), 0.0, 1.0
    ))

    # ------------------------------------------------------------------
    # Step 5 — barycentric negative-weight diagnostic
    # ------------------------------------------------------------------
    # Fraction of prompts where the unconstrained solution requires going
    # outside the simplex (any mixing weight < -0.05)
    negative_weight_fraction = float((S_free < -0.05).any(axis=1).mean())

    # ------------------------------------------------------------------
    # Step 6 — classification
    # ------------------------------------------------------------------
    if quantum_ness_score < 0.10:
        classification = "classical"
    elif quantum_ness_score >= 0.30:
        classification = "quantum"
    else:
        classification = "ambiguous"

    print(
        f"  cluster {cluster['cluster_id']:3d}: "
        f"n_features={len(feat_idx):4d}, "
        f"qness={quantum_ness_score:.4f}, "
        f"neg_wt={negative_weight_fraction:.3f}, "
        f"class={classification}, "
        f"cl_fvu={classical_fvu:.4f}, "
        f"q_fvu={quantum_fvu:.4f}"
    )

    return {
        "cluster_id":              int(cluster["cluster_id"]),
        "n_features":              int(len(feat_idx)),
        "quantum_ness_score":      quantum_ness_score,
        "negative_weight_fraction": negative_weight_fraction,
        "classical_fvu":           classical_fvu,
        "quantum_fvu":             quantum_fvu,
        "classification":          classification,
    }


def _degenerate_result(
    cluster: dict,
    reason: str = "degenerate",
    classical_fvu: float = 0.0,
) -> dict:
    cid = int(cluster["cluster_id"])
    n   = int(len(cluster["feature_indices"]))
    print(f"  cluster {cid:3d}: SKIPPED ({reason})")
    return {
        "cluster_id":               cid,
        "n_features":               n,
        "quantum_ness_score":       0.0,
        "negative_weight_fraction": 0.0,
        "classical_fvu":            classical_fvu,
        "quantum_fvu":              0.0,
        "classification":           "ambiguous",
    }


# ---------------------------------------------------------------------------
# Run all clusters
# ---------------------------------------------------------------------------

def classify_all_clusters(
    model_cfg: dict,
    clusters: list[dict],
    feature_activations: torch.Tensor,   # (n_prompts, n_features_total)
    results_dir: str = RESULTS_DIR,
    force_reclassify: bool = False,
) -> list[dict]:
    """Classify geometry for every cluster and save results."""
    save_path = os.path.join(results_dir, model_cfg["name"], "geometry.pt")
    if not force_reclassify and os.path.exists(save_path):
        print(f"[geometry] Loading geometry results from {save_path}")
        geo = torch.load(save_path, weights_only=False)
        print(f"[geometry] Loaded {len(geo)} cluster geometry results")
        return geo

    print(f"[geometry] Classifying {len(clusters)} clusters ...")
    print(f"[geometry] feature_activations shape: {feature_activations.shape}")
    print(f"[geometry] Method: activation-trajectory simplex gap (K={K_ARCHETYPES})")

    # Reset debug-print counter so we see fresh FVU values for the first 5 clusters
    global _DEBUG_FVU_PRINTS
    _DEBUG_FVU_PRINTS = 0

    results = []
    class_counts = {"classical": 0, "quantum": 0, "ambiguous": 0}

    for cluster in tqdm(clusters, desc="Classifying clusters"):
        geo = classify_cluster_geometry(cluster, feature_activations)
        results.append(geo)
        class_counts[geo["classification"]] += 1

    print(f"\n[geometry] Classification summary: {class_counts}")
    q_scores = [r["quantum_ness_score"] for r in results]
    neg_fracs = [r["negative_weight_fraction"] for r in results]
    print(f"[geometry] Quantum-ness scores:         "
          f"mean={np.mean(q_scores):.3f}, "
          f"min={np.min(q_scores):.3f}, "
          f"max={np.max(q_scores):.3f}")
    print(f"[geometry] Neg-weight fractions:        "
          f"mean={np.mean(neg_fracs):.3f}, "
          f"min={np.min(neg_fracs):.3f}, "
          f"max={np.max(neg_fracs):.3f}")

    torch.save(results, save_path)
    print(f"[geometry] Saved geometry results to {save_path}")
    return results


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from config import get_active_models

    print("=== Smoke test: geometry_classifier (activation-trajectory simplex gap) ===\n")
    rng = np.random.default_rng(42)

    n_prompts, n_features, d_model, sub_dim = 80, 200, 64, 10

    def _make_cluster(cid, n_feat=30):
        dec = rng.standard_normal((n_feat, d_model)).astype(np.float32)
        dec /= np.linalg.norm(dec, axis=1, keepdims=True) + 1e-8
        _, _, Vt = np.linalg.svd(dec, full_matrices=False)
        return {
            "cluster_id": cid,
            "feature_indices": torch.arange(n_feat),
            "decoder_vectors": torch.tensor(dec),
            "subspace_basis":  torch.tensor(Vt[:sub_dim].astype(np.float32)),
        }

    # ---- Test 1: simplex-like activations (classical) ----
    # Make activations that are convex combinations of K vertices
    vertices = rng.standard_normal((K_ARCHETYPES, n_features)).astype(np.float32)
    vertices = np.abs(vertices)                           # non-negative directions
    weights  = rng.dirichlet(np.ones(K_ARCHETYPES), size=n_prompts).astype(np.float32)
    fa_classical = torch.from_numpy(weights @ vertices)  # (n_prompts, n_features)

    cl1 = _make_cluster(0)
    r1  = classify_cluster_geometry(cl1, fa_classical)
    print(f"\nClassical input: qness={r1['quantum_ness_score']:.4f}  "
          f"class={r1['classification']}")

    # ---- Test 2: mixed activations with negative-weight structure ----
    fa_mixed = torch.from_numpy(
        rng.standard_normal((n_prompts, n_features)).astype(np.float32)
    )
    cl2 = _make_cluster(1)
    r2  = classify_cluster_geometry(cl2, fa_mixed)
    print(f"Mixed input:     qness={r2['quantum_ness_score']:.4f}  "
          f"class={r2['classification']}")

    # ---- Sanity: all scores in [0, 1] ----
    for r in (r1, r2):
        assert 0.0 <= r["quantum_ness_score"] <= 1.0, \
            f"Score out of range: {r['quantum_ness_score']}"
        assert 0.0 <= r["negative_weight_fraction"] <= 1.0

    # ---- Full pipeline ----
    models = get_active_models()
    if models:
        m = models[0]
        fa = torch.from_numpy(
            rng.exponential(size=(n_prompts, n_features)).astype(np.float32)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, m["name"]), exist_ok=True)
            clusters = [_make_cluster(i) for i in range(4)]
            geo = classify_all_clusters(
                m, clusters, fa, results_dir=tmpdir, force_reclassify=True
            )
            print(f"\nclassify_all_clusters: {len(geo)} results")

    print("\n[geometry_classifier] Smoke test passed.")
