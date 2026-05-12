"""
Reconstruction analysis.

For each cluster and each prompt we measure how much of the SAE's unexplained
variance (residual) falls inside that cluster's subspace. We then ask:

    Does the quantum-ness of a cluster's geometry predict how much residual
    variance it harbours?

Method
------
1.  Load per-prompt residuals: (n_prompts, d_model)  [from sae_extractor]
2.  For each cluster:
    - Project every residual vector onto the cluster's subspace basis.
    - Cluster-level FVU contribution = mean over prompts of:
        ||P_cluster(residual)||^2 / total_residual_variance
3.  Correlate (Spearman r) quantum-ness scores with cluster FVU contributions.
4.  Print and save results.
"""

import os
import sys
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from config import RESULTS_DIR


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_cluster_fvu_contributions(
    residuals: torch.Tensor,   # (n_prompts, d_model)
    clusters: list[dict],
) -> np.ndarray:
    """
    For each cluster, compute the mean fraction of residual variance explained
    by projecting residuals onto the cluster's subspace.

    Returns an array of shape (n_clusters,) — one FVU-contribution per cluster.
    """
    res_np = residuals.float().numpy()                           # (n_prompts, d_model)
    n_prompts = res_np.shape[0]

    # Total per-prompt residual squared norm (denominator for normalisation)
    total_var_per_prompt = (res_np ** 2).sum(axis=1)             # (n_prompts,)
    total_var_per_prompt = np.maximum(total_var_per_prompt, 1e-8)

    fvu_contributions = np.zeros(len(clusters), dtype=np.float32)

    for i, cluster in enumerate(tqdm(clusters, desc="Computing cluster FVU contributions")):
        basis = cluster["subspace_basis"].numpy()                # (subspace_dim, d_model)

        # Project residuals onto the subspace: P_S(r) = B^T (B r^T)
        # proj shape: (n_prompts, subspace_dim)
        proj = res_np @ basis.T

        # Squared norm of projection per prompt
        proj_norm_sq = (proj ** 2).sum(axis=1)                   # (n_prompts,)

        # Fraction of each prompt's residual that lives in this cluster's subspace
        frac = proj_norm_sq / total_var_per_prompt               # (n_prompts,)

        # Average across prompts
        fvu_contributions[i] = float(frac.mean())

    return fvu_contributions


def run_reconstruction_analysis(
    model_cfg: dict,
    clusters: list[dict],
    geometry_results: list[dict],
    activations: dict,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Full reconstruction analysis for one model.

    activations: output dict from sae_extractor, must contain 'residual'.

    Returns a dict with:
        cluster_ids          : list[int]
        quantum_ness_scores  : np.ndarray  (n_clusters,)
        fvu_contributions    : np.ndarray  (n_clusters,)
        classifications      : list[str]
        spearman_r           : float
        spearman_p           : float
        n_clusters           : int
    """
    save_path = os.path.join(results_dir, model_cfg["name"], "reconstruction_analysis.pt")

    residuals = activations["residual"]    # (n_prompts, d_model)
    print(f"[recon_analysis] residuals shape: {residuals.shape}")
    print(f"[recon_analysis] n_clusters: {len(clusters)}")

    # Build lookup from cluster_id → geometry result
    geo_by_id = {g["cluster_id"]: g for g in geometry_results}

    # Ensure cluster list and geometry list are aligned
    aligned_clusters = []
    aligned_geo = []
    for cl in clusters:
        cid = int(cl["cluster_id"])
        if cid in geo_by_id:
            aligned_clusters.append(cl)
            aligned_geo.append(geo_by_id[cid])

    print(f"[recon_analysis] Aligned {len(aligned_clusters)} clusters with geometry results")

    fvu_contributions = compute_cluster_fvu_contributions(residuals, aligned_clusters)

    quantum_ness_scores = np.array(
        [g["quantum_ness_score"] for g in aligned_geo], dtype=np.float32
    )
    # negative_weight_fraction: Levinson barycentric prediction test
    neg_weight_fracs = np.array(
        [g.get("negative_weight_fraction", 0.0) for g in aligned_geo], dtype=np.float32
    )
    classifications = [g["classification"] for g in aligned_geo]
    cluster_ids = [g["cluster_id"] for g in aligned_geo]

    print(f"\n[recon_analysis] Quantum-ness scores:       "
          f"mean={quantum_ness_scores.mean():.4f}, std={quantum_ness_scores.std():.4f}")
    print(f"[recon_analysis] Neg-weight fractions:      "
          f"mean={neg_weight_fracs.mean():.4f}, std={neg_weight_fracs.std():.4f}")
    print(f"[recon_analysis] FVU contributions:         "
          f"mean={fvu_contributions.mean():.4f}, std={fvu_contributions.std():.4f}")

    def _spearman(x, y, label):
        if len(x) < 3:
            print(f"[recon_analysis] Too few clusters for {label}.")
            return float("nan"), float("nan")
        r, p = spearmanr(x, y)
        return float(r), float(p)

    spearman_r, spearman_p = _spearman(quantum_ness_scores, fvu_contributions,
                                        "quantum_ness vs FVU")
    neg_r,      neg_p      = _spearman(neg_weight_fracs,    fvu_contributions,
                                        "neg_weight_frac vs FVU")

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  PRIMARY   — quantum_ness_score vs cluster FVU contribution")
    print(f"  Spearman r = {spearman_r:+.4f}  (p = {spearman_p:.4e})"
          f"  n = {len(aligned_clusters)}")
    if not np.isnan(spearman_r):
        if spearman_r > 0.4 and spearman_p < 0.05:
            print("  *** HYPOTHESIS SUPPORTED: quantum geometry predicts FVU ***")
        elif spearman_p >= 0.05:
            print("  Not statistically significant (p >= 0.05)")
        else:
            print(f"  Weak or negative correlation")
    print(f"\n  SECONDARY — negative_weight_fraction vs cluster FVU contribution")
    print(f"  (Levinson barycentric prediction test adapted to this setting)")
    print(f"  Spearman r = {neg_r:+.4f}  (p = {neg_p:.4e})"
          f"  n = {len(aligned_clusters)}")
    if not np.isnan(neg_r):
        if neg_r > 0.4 and neg_p < 0.05:
            print("  *** Out-of-simplex activations predict residual FVU ***")
        elif neg_p >= 0.05:
            print("  Not statistically significant (p >= 0.05)")
    print(f"{sep}\n")

    results = {
        "cluster_ids":               cluster_ids,
        "quantum_ness_scores":       quantum_ness_scores,
        "negative_weight_fractions": neg_weight_fracs,
        "fvu_contributions":         fvu_contributions,
        "classifications":           classifications,
        "spearman_r":                spearman_r,
        "spearman_p":                spearman_p,
        "neg_weight_spearman_r":     neg_r,
        "neg_weight_spearman_p":     neg_p,
        "n_clusters":                len(aligned_clusters),
    }
    torch.save(results, save_path)
    print(f"[recon_analysis] Saved results to {save_path}")
    return results


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from config import get_active_models

    print("=== Smoke test: reconstruction_analysis ===")
    rng = np.random.default_rng(42)

    n_prompts, d_model, n_clusters_fake = 100, 64, 15

    # Fake residuals
    residuals = torch.from_numpy(
        rng.standard_normal((n_prompts, d_model)).astype(np.float32)
    )
    activations_fake = {
        "residual": residuals,
        "feature_activations": torch.zeros(n_prompts, 500),
        "sae_reconstruction": torch.zeros(n_prompts, d_model),
        "fvu_per_prompt": torch.zeros(n_prompts),
    }

    # Fake clusters
    clusters = []
    geo_results = []
    for i in range(n_clusters_fake):
        dec = rng.standard_normal((20, d_model)).astype(np.float32)
        dec /= np.linalg.norm(dec, axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(dec, full_matrices=False)
        basis = Vt[:5]
        clusters.append({
            "cluster_id": i,
            "feature_indices": torch.arange(20),
            "decoder_vectors": torch.tensor(dec),
            "subspace_basis": torch.tensor(basis),
        })
        qs = float(rng.uniform(0, 1))
        geo_results.append({
            "cluster_id": i,
            "quantum_ness_score": qs,
            "negative_weight_fraction": float(rng.uniform(0, 1)),
            "classical_fvu": float(rng.uniform(0.01, 0.5)),
            "quantum_fvu": float(rng.uniform(0, 0.3)),
            "classification": "quantum" if qs > 0.3 else "classical",
        })

    models = get_active_models()
    if not models:
        print("No active models.")
        sys.exit(1)

    m = models[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, m["name"]), exist_ok=True)
        results = run_reconstruction_analysis(
            m, clusters, geo_results, activations_fake, results_dir=tmpdir
        )
        print(f"  Spearman r (qness)    ={results['spearman_r']:.4f}, p={results['spearman_p']:.4e}")
        print(f"  Spearman r (neg_wt)   ={results['neg_weight_spearman_r']:.4f}, p={results['neg_weight_spearman_p']:.4e}")
        print(f"  n_clusters={results['n_clusters']}")

    print("[reconstruction_analysis] Smoke test passed.")
