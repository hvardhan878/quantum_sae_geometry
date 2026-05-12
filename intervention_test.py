"""
intervention_test.py — quick sanity check for the intervention experiment.

Run after the layer sweep completes:

    # default: Gemma-2-2B
    python intervention_test.py --layer 0
    python intervention_test.py --layer 12 --top_n 10

    # 9B run
    python intervention_test.py --model gemma-2-9b-it --layer 0
    python intervention_test.py --model gemma-2-9b-it --layer 41

    # different results dir
    python intervention_test.py --results_dir /storage/results --model gemma-2-9b-it --layer 0

    # fully explicit (overrides --model/--layer/--results_dir composition)
    python intervention_test.py --model_dir /storage/results/gemma-2-9b-it-layer-0

What this does
--------------
1. Loads activations.pt, geometry.pt, and clusters.pt for one layer.
2. Picks the top-N highest- and bottom-N lowest-quantum-ness clusters.
3. Recomputes the constrained vs unconstrained simplex FVU for each one
   directly from the saved subspace bases and residual-stream hidden states.
4. Runs a one-sided Mann-Whitney U test asking whether the top-quantum
   clusters have a *larger* unconstrained-vs-constrained FVU gap than the
   bottom-quantum clusters.
5. Saves a two-panel diagnostic plot to <model_dir>/intervention_test.png.
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

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=0,
                    help="layer number (composes with --model into "
                         "<results_dir>/<model>-layer-<layer>)")
parser.add_argument("--model", type=str, default="gemma-2-2b-it",
                    help="model family prefix, e.g. 'gemma-2-2b-it', "
                         "'gemma-2-9b-it', 'llama-3.1-8b'")
parser.add_argument("--results_dir", type=str, default="./results",
                    help="parent results directory containing per-model folders")
parser.add_argument("--model_dir", type=str, default=None,
                    help="optional: full path to a single model directory; "
                         "if given, overrides --model/--layer/--results_dir "
                         "composition")
parser.add_argument("--top_n", type=int, default=10)
parser.add_argument("--seed", type=int, default=0,
                    help="seed for the random simplex initialisation")
args = parser.parse_args()

np.random.seed(args.seed)

# Resolve the per-layer results directory.
if args.model_dir is not None:
    base = args.model_dir
    model_name = os.path.basename(os.path.normpath(base))
else:
    model_name = f"{args.model}-layer-{args.layer}"
    base = os.path.join(args.results_dir, model_name)

if not os.path.isdir(base):
    print(f"[error] model directory does not exist: {base}", file=sys.stderr)
    sys.exit(1)

print(f"\nLoading data from {base} ...")
activations  = torch.load(os.path.join(base, "activations.pt"), weights_only=False)
geometry_raw = torch.load(os.path.join(base, "geometry.pt"),     weights_only=False)
clusters_raw = torch.load(os.path.join(base, "clusters.pt"),     weights_only=False)

# geometry.pt and clusters.pt are saved as LISTS of dicts (each carrying
# its own cluster_id). Build dicts keyed by cluster_id for convenient lookup.
def _to_id_dict(items):
    if isinstance(items, dict):
        return items
    return {int(d["cluster_id"]): d for d in items}

geometry = _to_id_dict(geometry_raw)
clusters = _to_id_dict(clusters_raw)

last_hidden = activations["last_hidden"].float().numpy()   # (n_prompts, d_model)
print(f"last_hidden shape: {last_hidden.shape}")
print(f"n_clusters in geometry: {len(geometry)}")

# ── Extract quantum-ness scores ───────────────────────────────────────────────
cluster_ids     = sorted(geometry.keys())
qness_scores    = np.array([geometry[c]["quantum_ness_score"] for c in cluster_ids])
cl_fvu_scores   = np.array([geometry[c].get("classical_fvu", 0.0) for c in cluster_ids])
q_fvu_scores    = np.array([geometry[c].get("quantum_fvu",   0.0) for c in cluster_ids])
classifications = [geometry[c]["classification"] for c in cluster_ids]

print(f"\nQuantum-ness scores: mean={qness_scores.mean():.3f} "
      f"min={qness_scores.min():.3f} max={qness_scores.max():.3f}")

# Cap top_n so the two groups don't overlap when there are few clusters.
top_n = min(args.top_n, len(cluster_ids) // 2)
if top_n < args.top_n:
    print(f"[warning] only {len(cluster_ids)} clusters available — "
          f"using top_n={top_n} so groups don't overlap")

# ── Split into top-N quantum and bottom-N classical ───────────────────────────
ranked = np.argsort(qness_scores)[::-1]
top_quantum_idx      = ranked[:top_n]
bottom_classical_idx = ranked[-top_n:]

print(f"\nTop {top_n} quantum clusters:")
for i in top_quantum_idx:
    cid = cluster_ids[i]
    print(f"  cluster {cid:3d}: qness={qness_scores[i]:.3f}  "
          f"cl_fvu={cl_fvu_scores[i]:.4f}  q_fvu={q_fvu_scores[i]:.4f}  "
          f"gap={cl_fvu_scores[i]-q_fvu_scores[i]:.4f}  "
          f"class={classifications[i]}")

print(f"\nBottom {top_n} classical clusters:")
for i in bottom_classical_idx:
    cid = cluster_ids[i]
    print(f"  cluster {cid:3d}: qness={qness_scores[i]:.3f}  "
          f"cl_fvu={cl_fvu_scores[i]:.4f}  q_fvu={q_fvu_scores[i]:.4f}  "
          f"gap={cl_fvu_scores[i]-q_fvu_scores[i]:.4f}  "
          f"class={classifications[i]}")


# ── For each cluster recompute constrained vs unconstrained error ─────────────
def fit_simplex(projected, K=4, n_iters=100):
    """Constrained archetypal analysis. Returns S (n, K), A (K, d)."""
    n, d = projected.shape
    # farthest-point init
    A = np.zeros((K, d))
    A[0] = projected[np.random.randint(n)]
    for k in range(1, K):
        dists = np.min(
            np.stack([np.sum((projected - A[j])**2, axis=1) for j in range(k)]),
            axis=0,
        )
        A[k] = projected[np.argmax(dists)]

    S = np.ones((n, K)) / K
    for _ in range(n_iters):
        # update S — constrained (non-negative, sum to 1)
        for i in range(n):
            s, _ = nnls(A.T, projected[i])
            s_sum = s.sum()
            S[i] = s / s_sum if s_sum > 1e-8 else np.ones(K) / K
        # update A — unconstrained
        AtA = S.T @ S + 1e-6 * np.eye(K)
        AtP = S.T @ projected
        A = np.linalg.solve(AtA, AtP)
    return S, A


def compute_errors(projected, K=4):
    """Returns classical_fvu, quantum_fvu, fvu_gap for a cluster."""
    centered  = projected - projected.mean(axis=0)
    total_var = (centered ** 2).sum()
    if total_var < 1e-6:
        return 0.0, 0.0, 0.0

    S_con, A = fit_simplex(projected, K=K)

    # constrained reconstruction
    recon_con = S_con @ A
    resid_con = projected - recon_con
    cl_fvu    = (resid_con ** 2).sum() / (total_var + 1e-8)

    # unconstrained reconstruction — same archetypes, free (ridge) weights
    ATA = A @ A.T + 1e-4 * np.eye(K)
    ATP = A @ projected.T
    S_free = np.linalg.solve(ATA, ATP).T
    recon_free = S_free @ A
    resid_free = projected - recon_free
    q_fvu      = (resid_free ** 2).sum() / (total_var + 1e-8)

    gap = float(np.clip((cl_fvu - q_fvu) / (cl_fvu + 1e-8), 0, 1))
    return float(cl_fvu), float(q_fvu), gap


def get_projected(cid):
    cluster = clusters[cid]
    basis = cluster["subspace_basis"].float().numpy()   # (subspace_dim, d_model)
    return last_hidden @ basis.T                         # (n_prompts, subspace_dim)


print(f"\nRecomputing errors for top/bottom {top_n} clusters ...")

quantum_gaps   = []
classical_gaps = []

print("\nQuantum clusters:")
for i in top_quantum_idx:
    cid = cluster_ids[i]
    projected = get_projected(cid)
    cl, qu, gap = compute_errors(projected)
    quantum_gaps.append(gap)
    print(f"  cluster {cid:3d}: cl_fvu={cl:.4f}  q_fvu={qu:.4f}  "
          f"gap={gap:.4f}  qness={qness_scores[i]:.3f}")

print("\nClassical/low-quantum clusters:")
for i in bottom_classical_idx:
    cid = cluster_ids[i]
    projected = get_projected(cid)
    cl, qu, gap = compute_errors(projected)
    classical_gaps.append(gap)
    print(f"  cluster {cid:3d}: cl_fvu={cl:.4f}  q_fvu={qu:.4f}  "
          f"gap={gap:.4f}  qness={qness_scores[i]:.3f}")

# ── Statistical test ──────────────────────────────────────────────────────────
quantum_gaps   = np.array(quantum_gaps)
classical_gaps = np.array(classical_gaps)

stat, p = mannwhitneyu(quantum_gaps, classical_gaps, alternative="greater")
r_val, r_p = spearmanr(
    np.concatenate([qness_scores[top_quantum_idx],
                    qness_scores[bottom_classical_idx]]),
    np.concatenate([quantum_gaps, classical_gaps]),
)

print(f"""
==============================================================
  INTERVENTION SANITY CHECK — {model_name}
==============================================================
  Quantum clusters   (n={top_n}): mean gap = {quantum_gaps.mean():.4f}
  Classical clusters (n={top_n}): mean gap = {classical_gaps.mean():.4f}

  Mann-Whitney U (quantum > classical):
    U={stat:.1f}  p={p:.4f}  {'*** SIGNIFICANT' if p < 0.05 else 'not significant'}

  Spearman r (qness vs gap, combined):
    r={r_val:.3f}  p={r_p:.4f}

  Interpretation:
    gap > 0 means unconstrained beats constrained — quantum geometry is real
    significant Mann-Whitney means quantum clusters benefit MORE from
    unconstrained decomposition than classical clusters
==============================================================
""")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: gap comparison
ax = axes[0]
ax.boxplot([quantum_gaps, classical_gaps])
ax.set_xticks([1, 2])
ax.set_xticklabels(["Quantum", "Classical"])
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("FVU Gap (classical − unconstrained) / classical")
ax.set_title(f"{model_name} — FVU Gap by Cluster Type")
for g in quantum_gaps:
    ax.scatter(1 + np.random.uniform(-0.05, 0.05), g, color="red",  alpha=0.6, s=40)
for g in classical_gaps:
    ax.scatter(2 + np.random.uniform(-0.05, 0.05), g, color="blue", alpha=0.6, s=40)
ax.text(0.05, 0.95, f"p={p:.4f}", transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        color="green" if p < 0.05 else "red")

# Right: qness vs gap scatter
ax = axes[1]
all_qness = np.concatenate([qness_scores[top_quantum_idx],
                            qness_scores[bottom_classical_idx]])
all_gaps  = np.concatenate([quantum_gaps, classical_gaps])
colors    = ["red"] * top_n + ["blue"] * top_n
ax.scatter(all_qness, all_gaps, c=colors, alpha=0.7, s=60)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Quantum-ness Score")
ax.set_ylabel("FVU Gap")
ax.set_title(f"{model_name} — Quantum-ness vs FVU Gap\n"
             f"Spearman r={r_val:.3f} p={r_p:.4f}")

plt.tight_layout()
out = os.path.join(base, "intervention_test.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved to {out}")
