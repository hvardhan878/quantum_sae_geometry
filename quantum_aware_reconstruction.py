"""
quantum_aware_reconstruction.py

Evaluates the Quantum-Aware SAE Reconstruction algorithm using saved results
from the layer sweep. No model or SAE reloading required.

ALGORITHM
---------
For each cluster:
  - Project residual-stream hidden states onto the cluster's subspace.
  - Fit K archetypes (AA simplex) on TRAIN projections only.
  - Evaluate two reconstruction conditions on TEST projections only:

    Condition A — Standard/classical:
      Constrained NNLS weights against train archetypes → classical FVU.

    Condition B — Quantum-aware:
      Unconstrained ridge-regularised weights against the SAME train
      archetypes → quantum FVU.

  - fvu_gap = (fvu_classical − fvu_quantum) / fvu_classical

PRIMARY RESULT: quantum clusters should have a larger fvu_gap than
classical ones (Mann-Whitney U, one-sided), and fvu_gap should correlate
positively with quantum_ness_score (Spearman r).

SECONDARY RESULT: aggregate FVU reduction from switching quantum clusters
to unconstrained reconstruction.

Usage
-----
    python quantum_aware_reconstruction.py --model_dir ./results/gemma-2-2b-it-layer-0
    python quantum_aware_reconstruction.py \\
        --model_dir ./results/gemma-2-9b-it-layer-0 \\
        --quantum_threshold 0.30 --n_archetypes 4 --train_frac 0.75 --top_n 10
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu, spearmanr

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Quantum-Aware SAE Reconstruction evaluation"
)
parser.add_argument(
    "--model_dir", type=str, required=True,
    help="Path to a single model results directory "
         "(e.g. ./results/gemma-2-2b-it-layer-0). "
         "Must contain activations.pt, geometry.pt, clusters.pt.",
)
parser.add_argument(
    "--train_frac", type=float, default=0.75,
    help="Fraction of prompts used to fit archetypes (default 0.75).",
)
parser.add_argument(
    "--quantum_threshold", type=float, default=0.30,
    help="quantum_ness_score >= this → cluster treated as quantum (default 0.30).",
)
parser.add_argument(
    "--n_archetypes", type=int, default=4,
    help="Number of AA archetypes K (default 4).",
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for train/test split and simplex init (default 42).",
)
parser.add_argument(
    "--top_n", type=int, default=10,
    help="Number of top-quantum / bottom-classical clusters to print in detail "
         "(default 10).",
)
args = parser.parse_args()

np.random.seed(args.seed)

base = args.model_dir
model_name = os.path.basename(os.path.normpath(base))

if not os.path.isdir(base):
    print(f"[error] model directory does not exist: {base}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------------

print(f"\nLoading data from {base} ...")
activations  = torch.load(os.path.join(base, "activations.pt"), weights_only=False)
geometry_raw = torch.load(os.path.join(base, "geometry.pt"),    weights_only=False)
clusters_raw = torch.load(os.path.join(base, "clusters.pt"),    weights_only=False)


def _to_id_dict(items):
    """Convert a list-of-dicts (each with cluster_id) to a dict keyed by int id."""
    if isinstance(items, dict):
        return items
    return {int(d["cluster_id"]): d for d in items}


geometry = _to_id_dict(geometry_raw)
clusters = _to_id_dict(clusters_raw)

last_hidden = activations["last_hidden"].float().numpy()   # (n_prompts, d_model)
n_prompts, d_model = last_hidden.shape
print(f"last_hidden shape: {last_hidden.shape}")
print(f"n_clusters: {len(geometry)}")

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

all_idx   = np.arange(n_prompts)
n_train   = int(n_prompts * args.train_frac)
train_idx = np.random.choice(all_idx, size=n_train, replace=False)
test_idx  = np.setdiff1d(all_idx, train_idx)
n_test    = len(test_idx)

print(f"Train prompts: {n_train}  |  Test prompts: {n_test}")

# ---------------------------------------------------------------------------
# Simplex fitting helpers  (identical logic to intervention_test.py)
# ---------------------------------------------------------------------------

def _farthest_point_init(projected: np.ndarray, K: int) -> np.ndarray:
    """Pick K farthest-point seeds from projected rows. Returns (K, d)."""
    n, d = projected.shape
    A = np.zeros((K, d))
    A[0] = projected[np.random.randint(n)]
    for k in range(1, K):
        dists = np.min(
            np.stack([np.sum((projected - A[j]) ** 2, axis=1) for j in range(k)]),
            axis=0,
        )
        A[k] = projected[np.argmax(dists)]
    return A


def _fit_simplex(projected: np.ndarray, K: int = 4, n_iters: int = 100):
    """
    Constrained Archetypal Analysis.

    Fits K archetypes to `projected` (n, d).
    Returns:
        S : (n, K)   — constrained mixing weights (non-negative, sum-to-1)
        A : (K, d)   — archetype vectors
    """
    n, d = projected.shape
    A = _farthest_point_init(projected, K)
    S = np.ones((n, K)) / K

    for _ in range(n_iters):
        # Update S — constrained NNLS per prompt, then normalise to sum-to-1
        for i in range(n):
            s, _ = nnls(A.T, projected[i])
            s_sum = s.sum()
            S[i] = s / s_sum if s_sum > 1e-8 else np.ones(K) / K
        # Update A — unconstrained least squares
        AtA = S.T @ S + 1e-6 * np.eye(K)
        AtP = S.T @ projected
        A   = np.linalg.solve(AtA, AtP)

    return S, A


def _constrained_weights(A: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """
    For each row of `projected`, solve constrained NNLS against archetypes A.
    Returns S_con (n, K), normalised to sum-to-1.
    """
    n = projected.shape[0]
    K = A.shape[0]
    S_con = np.zeros((n, K))
    for i in range(n):
        s, _ = nnls(A.T, projected[i])
        s_sum = s.sum()
        S_con[i] = s / s_sum if s_sum > 1e-8 else np.ones(K) / K
    return S_con


def _unconstrained_weights(A: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """
    Unconstrained ridge-regularised weights for `projected` given archetypes A.
    Solves: (A A^T + 1e-4 I) S^T = A projected^T
    Returns S_free (n, K).
    """
    K = A.shape[0]
    ATA = A @ A.T + 1e-4 * np.eye(K)
    ATP = A @ projected.T                  # (K, n)
    S_free = np.linalg.solve(ATA, ATP).T   # (n, K)
    return S_free


def _fvu(projected: np.ndarray, recon: np.ndarray) -> float:
    """FVU = ||projected - recon||^2 / ||projected - mean||^2."""
    centered  = projected - projected.mean(axis=0)
    total_var = (centered ** 2).sum()
    if total_var < 1e-8:
        return 0.0
    return float(((projected - recon) ** 2).sum() / total_var)


# ---------------------------------------------------------------------------
# Per-cluster held-out evaluation
# ---------------------------------------------------------------------------

K         = args.n_archetypes
threshold = args.quantum_threshold

cluster_ids = sorted(set(geometry.keys()) & set(clusters.keys()))
n_skipped   = 0

results = []   # list of dicts: cluster_id, qness, fvu_classical, fvu_quantum, fvu_gap

print(f"\nEvaluating {len(cluster_ids)} clusters "
      f"(K={K}, threshold={threshold}, "
      f"train={n_train}, test={n_test}) ...")

for cid in cluster_ids:
    qness  = float(geometry[cid]["quantum_ness_score"])
    basis  = clusters[cid]["subspace_basis"].float().numpy()  # (sub_dim, d_model)
    sub_dim = basis.shape[0]

    # Project all prompts onto the subspace
    proj_train = (last_hidden[train_idx] @ basis.T).astype(np.float32)  # (n_train, sub_dim)
    proj_test  = (last_hidden[test_idx]  @ basis.T).astype(np.float32)  # (n_test, sub_dim)

    # Guard: need at least K test prompts to compute a meaningful FVU
    if n_test < K:
        print(f"  [skip] cluster {cid:3d}: only {n_test} test prompts < K={K}")
        n_skipped += 1
        continue

    # Check train variance is non-degenerate
    train_centered  = proj_train - proj_train.mean(axis=0)
    train_total_var = (train_centered ** 2).sum()
    if train_total_var < 1e-6:
        print(f"  [skip] cluster {cid:3d}: near-zero train variance")
        n_skipped += 1
        continue

    try:
        # Fit archetypes on TRAIN projections only
        _, A = _fit_simplex(proj_train, K=K, n_iters=100)

        # Condition A — constrained weights on TEST
        S_con  = _constrained_weights(A, proj_test)
        recon_a = S_con @ A
        fvu_a  = _fvu(proj_test, recon_a)

        # Condition B — unconstrained weights on TEST (same archetypes)
        S_free = _unconstrained_weights(A, proj_test)
        recon_b = S_free @ A
        fvu_b  = _fvu(proj_test, recon_b)

    except np.linalg.LinAlgError as exc:
        print(f"  [skip] cluster {cid:3d}: LinAlgError — {exc}")
        n_skipped += 1
        continue

    # fvu_gap: how much does unconstrained beat constrained?
    fvu_gap = float(np.clip((fvu_a - fvu_b) / (fvu_a + 1e-8), 0.0, 1.0))

    results.append({
        "cluster_id":    cid,
        "qness":         qness,
        "fvu_classical": fvu_a,
        "fvu_quantum":   fvu_b,
        "fvu_gap":       fvu_gap,
        "is_quantum":    qness >= threshold,
    })

if n_skipped:
    print(f"  ({n_skipped} clusters skipped — see messages above)")

if not results:
    print("[error] No clusters produced valid results. Exiting.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Organise results
# ---------------------------------------------------------------------------

all_qness   = np.array([r["qness"]         for r in results])
all_fvu_cl  = np.array([r["fvu_classical"]  for r in results])
all_fvu_qu  = np.array([r["fvu_quantum"]    for r in results])
all_gaps    = np.array([r["fvu_gap"]        for r in results])
is_quantum  = np.array([r["is_quantum"]     for r in results])

quantum_gaps   = all_gaps[is_quantum]
classical_gaps = all_gaps[~is_quantum]
n_q = int(is_quantum.sum())
n_c = int((~is_quantum).sum())

# Per-cluster detail for top-N quantum and bottom-N classical
ranked     = np.argsort(all_qness)[::-1]
top_n_act  = min(args.top_n, len(results) // 2 if len(results) >= 2 else 1)

print(f"\nTop {top_n_act} quantum clusters (held-out FVU):")
for i in ranked[:top_n_act]:
    r = results[i]
    print(f"  cluster {r['cluster_id']:3d}: "
          f"qness={r['qness']:.3f}  "
          f"fvu_cl={r['fvu_classical']:.4f}  "
          f"fvu_qu={r['fvu_quantum']:.4f}  "
          f"gap={r['fvu_gap']:.4f}")

print(f"\nBottom {top_n_act} classical clusters (held-out FVU):")
for i in ranked[-top_n_act:]:
    r = results[i]
    print(f"  cluster {r['cluster_id']:3d}: "
          f"qness={r['qness']:.3f}  "
          f"fvu_cl={r['fvu_classical']:.4f}  "
          f"fvu_qu={r['fvu_quantum']:.4f}  "
          f"gap={r['fvu_gap']:.4f}")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

# Mann-Whitney U — one-sided: quantum gaps > classical gaps
if n_q > 0 and n_c > 0:
    mw_stat, mw_p = mannwhitneyu(quantum_gaps, classical_gaps, alternative="greater")
else:
    mw_stat, mw_p = float("nan"), float("nan")
    print("[warning] One group is empty — Mann-Whitney U not computable.")

# Spearman r across all clusters
if len(results) >= 3:
    r_val, r_p = spearmanr(all_qness, all_gaps)
else:
    r_val, r_p = float("nan"), float("nan")

# Aggregate FVU reduction from quantum-aware reconstruction
# Baseline: sum of classical FVUs across all clusters
baseline_total = all_fvu_cl.sum()
# Hybrid:   keep classical FVU for classical clusters, use quantum FVU for quantum ones
hybrid_fvu     = np.where(is_quantum, all_fvu_qu, all_fvu_cl)
hybrid_total   = hybrid_fvu.sum()
if baseline_total > 1e-8:
    pct_reduction = 100.0 * (baseline_total - hybrid_total) / baseline_total
else:
    pct_reduction = 0.0

q_mean_gap  = float(quantum_gaps.mean())   if n_q > 0 else float("nan")
c_mean_gap  = float(classical_gaps.mean()) if n_c > 0 else float("nan")
sig_label   = "*** SIGNIFICANT" if (not np.isnan(mw_p) and mw_p < 0.05) else "not significant"

print(f"""
==============================================================
  QUANTUM-AWARE RECONSTRUCTION — {model_name}
  Train prompts: {n_train}  |  Test prompts: {n_test}
  Quantum clusters  (n={n_q}, threshold={threshold}): mean FVU gap = {q_mean_gap:.4f}
  Classical clusters (n={n_c}): mean FVU gap = {c_mean_gap:.4f}
  Mann-Whitney U={mw_stat:.1f} p={mw_p:.4f} [{sig_label}]
  Spearman r={r_val:.3f} p={r_p:.4f}
  Aggregate FVU reduction from quantum-aware reconstruction: {pct_reduction:.1f}%
==============================================================
""")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel — boxplot + jitter
ax = axes[0]
plot_data   = [quantum_gaps, classical_gaps] if n_q > 0 and n_c > 0 else [all_gaps]
plot_labels = ["Quantum", "Classical"]       if n_q > 0 and n_c > 0 else ["All"]
ax.boxplot(plot_data)
ax.set_xticks(range(1, len(plot_labels) + 1))
ax.set_xticklabels(plot_labels)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Held-out FVU Gap\n(classical − unconstrained) / classical")
ax.set_title(f"{model_name}\nFVU Gap by Cluster Type (held-out test set)")
for j, (gaps, col) in enumerate(zip(plot_data, ["red", "blue"]), start=1):
    for g in gaps:
        ax.scatter(j + np.random.uniform(-0.06, 0.06), g,
                   color=col, alpha=0.6, s=40, zorder=3)
p_text  = f"p={mw_p:.4f}" if not np.isnan(mw_p) else "p=n/a"
p_color = "green" if (not np.isnan(mw_p) and mw_p < 0.05) else "red"
ax.text(0.05, 0.95, p_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="top", color=p_color)

# Right panel — scatter: qness vs gap
ax = axes[1]
colors = np.where(is_quantum, "red", "blue")
for c in ["red", "blue"]:
    mask  = colors == c
    label = "Quantum" if c == "red" else "Classical"
    ax.scatter(all_qness[mask], all_gaps[mask],
               c=c, alpha=0.7, s=60, label=label, zorder=3)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(threshold, color="gray", linestyle=":", alpha=0.5,
           label=f"threshold={threshold}")
ax.set_xlabel("Quantum-ness Score")
ax.set_ylabel("Held-out FVU Gap")
r_str = f"Spearman r={r_val:.3f} p={r_p:.4f}" if not np.isnan(r_val) else ""
ax.set_title(f"{model_name}\nQuantum-ness vs Held-out FVU Gap\n{r_str}")
ax.legend(fontsize=9, loc="best")

plt.tight_layout()
out = os.path.join(base, "quantum_aware_reconstruction.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out}")
