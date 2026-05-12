"""
Visualisation for the quantum SAE geometry experiment.

Produces three plots per model:
  1. Scatter: quantum-ness score vs. mean FVU contribution (coloured by classification).
  2. Scatter: negative-weight fraction vs. mean FVU contribution (continuous colormap).
  3. Bar chart: count of classical / quantum / ambiguous clusters.

All saved under results/{model_name}/.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from config import RESULTS_DIR

CLASS_COLORS = {
    "classical": "#2C7BB6",
    "quantum":   "#D7191C",
    "ambiguous": "#808080",
}


def _load_results(model_name: str, results_dir: str) -> tuple[list | None, dict | None]:
    geo_path   = os.path.join(results_dir, model_name, "geometry.pt")
    recon_path = os.path.join(results_dir, model_name, "reconstruction_analysis.pt")

    geo, recon = None, None
    if os.path.exists(geo_path):
        geo = torch.load(geo_path, weights_only=False)
    else:
        print(f"[visualize] geometry.pt not found at {geo_path}")

    if os.path.exists(recon_path):
        recon = torch.load(recon_path, weights_only=False)
    else:
        print(f"[visualize] reconstruction_analysis.pt not found at {recon_path}")

    return geo, recon


def _annotate_spearman(ax, r, p, n, x=0.98, y=0.97):
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    ax.text(
        x, y,
        f"Spearman r = {r:.3f}\np = {p_str}\nn = {n}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def plot_quantum_vs_fvu(
    model_cfg: dict,
    results_dir: str = RESULTS_DIR,
) -> str | None:
    """
    Primary scatter: quantum-ness score vs FVU contribution, coloured by classification.
    """
    _, recon = _load_results(model_cfg["name"], results_dir)
    if recon is None:
        return None

    qs     = np.array(recon["quantum_ness_scores"], dtype=float)
    fvu    = np.array(recon["fvu_contributions"],   dtype=float)
    labels = recon["classifications"]
    r, p   = recon["spearman_r"], recon["spearman_p"]
    colors = [CLASS_COLORS.get(lbl, "#808080") for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(qs, fvu, c=colors, alpha=0.75, edgecolors="white", linewidths=0.4, s=60)

    patches = [
        mpatches.Patch(color=CLASS_COLORS["classical"], label="Classical"),
        mpatches.Patch(color=CLASS_COLORS["quantum"],   label="Quantum"),
        mpatches.Patch(color=CLASS_COLORS["ambiguous"], label="Ambiguous"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    _annotate_spearman(ax, r, p, len(qs))

    ax.set_xlabel(
        "Quantum-ness Score\n(classical vs unconstrained simplex FVU gap)",
        fontsize=10,
    )
    ax.set_ylabel(
        "Mean Cluster FVU Contribution\n(fraction of SAE residual variance in subspace)",
        fontsize=10,
    )
    ax.set_title(
        f"{model_cfg['name']} — Quantum Geometry vs SAE Reconstruction Error",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    save_path = os.path.join(results_dir, model_cfg["name"], "quantum_vs_fvu.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved scatter (quantum-ness vs FVU) to {save_path}")
    return save_path


def plot_neg_weight_vs_fvu(
    model_cfg: dict,
    results_dir: str = RESULTS_DIR,
) -> str | None:
    """
    Secondary scatter: negative-weight fraction vs FVU contribution.

    Colour encodes negative_weight_fraction as a continuous heatmap so we can
    see the Levinson barycentric prediction test result visually.
    """
    _, recon = _load_results(model_cfg["name"], results_dir)
    if recon is None:
        return None

    neg   = np.array(recon.get("negative_weight_fractions",
                               recon.get("quantum_ness_scores")),  # graceful fallback
                     dtype=float)
    fvu   = np.array(recon["fvu_contributions"], dtype=float)
    r     = recon.get("neg_weight_spearman_r", float("nan"))
    p     = recon.get("neg_weight_spearman_p", float("nan"))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(neg, fvu, c=neg, cmap="plasma",
                    vmin=0, vmax=1,
                    alpha=0.80, edgecolors="white", linewidths=0.3, s=65)
    plt.colorbar(sc, ax=ax, label="Negative-weight fraction")

    if not (np.isnan(r) or np.isnan(p)):
        _annotate_spearman(ax, r, p, len(neg))

    ax.set_xlabel(
        "Negative-weight Fraction\n"
        "(fraction of prompts where unconstrained simplex weights go negative)",
        fontsize=10,
    )
    ax.set_ylabel(
        "Mean Cluster FVU Contribution\n(fraction of SAE residual variance in subspace)",
        fontsize=10,
    )
    ax.set_title(
        f"{model_cfg['name']} — Barycentric Negativity vs SAE Reconstruction Error",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    save_path = os.path.join(
        results_dir, model_cfg["name"], "neg_weight_vs_fvu.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved scatter (neg-weight vs FVU) to {save_path}")
    return save_path


def plot_cluster_classifications(
    model_cfg: dict,
    results_dir: str = RESULTS_DIR,
) -> str | None:
    """Bar chart: count of classical / quantum / ambiguous clusters."""
    geo, _ = _load_results(model_cfg["name"], results_dir)
    if geo is None:
        return None

    counts = {"classical": 0, "quantum": 0, "ambiguous": 0}
    for g in geo:
        counts[g["classification"]] = counts.get(g["classification"], 0) + 1

    labels     = list(counts.keys())
    values     = [counts[k] for k in labels]
    bar_colors = [CLASS_COLORS[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("Geometry Classification", fontsize=10)
    ax.set_ylabel("Number of Clusters", fontsize=10)
    ax.set_title(
        f"{model_cfg['name']} — Cluster Geometry Classifications", fontsize=11
    )
    ax.set_ylim(0, max(values) * 1.2 + 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    save_path = os.path.join(
        results_dir, model_cfg["name"], "cluster_classifications.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved bar chart (classifications) to {save_path}")
    return save_path


def visualize_model(model_cfg: dict, results_dir: str = RESULTS_DIR) -> None:
    """Generate all plots for a single model."""
    print(f"[visualize] Generating plots for {model_cfg['name']} ...")
    plot_quantum_vs_fvu(model_cfg, results_dir)
    plot_neg_weight_vs_fvu(model_cfg, results_dir)
    plot_cluster_classifications(model_cfg, results_dir)


# ---------------------------------------------------------------------------
# Layer-sweep plot
# ---------------------------------------------------------------------------

def _fisher_z_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """
    95% confidence interval for a Spearman correlation via the Fisher z-transform.

        z   = atanh(r) = 0.5 * ln((1+r)/(1-r))
        se  = 1 / sqrt(n - 3)
        ci  = tanh(z ± 1.96 * se)

    Returns (lower, upper). If n <= 3 or r is NaN/|r| >= 1, returns (r, r).
    """
    if n is None or n <= 3 or r is None:
        return (float("nan"), float("nan"))
    if np.isnan(r) or abs(r) >= 1.0:
        return (r, r)
    z   = np.arctanh(r)
    se  = 1.0 / np.sqrt(n - 3)
    z_crit = 1.96  # ~95%
    lo = float(np.tanh(z - z_crit * se))
    hi = float(np.tanh(z + z_crit * se))
    return (lo, hi)


def plot_layer_sweep(
    family: str,
    sweeps: list[dict],
    results_dir: str = RESULTS_DIR,
) -> str | None:
    """
    Cross-model layer sweep: Spearman r (quantum-ness vs FVU) as a function of
    the layer at which the SAE was trained.

    Parameters
    ----------
    family : str
        Short label for the model family (e.g. 'gemma-2-2b-it').
        Used in the title and the saved filename.
    sweeps : list[dict]
        Per-model summaries from `run_experiment.run_model`. Each must contain
        at least `target_layer`, `spearman_r`, `spearman_p`, `n_clusters`.
    """
    if not sweeps:
        return None

    sweeps = sorted(sweeps, key=lambda s: s["target_layer"])
    layers = np.array([s["target_layer"] for s in sweeps], dtype=float)
    rs     = np.array([s["spearman_r"]   for s in sweeps], dtype=float)
    ns     = [int(s["n_clusters"])      for s in sweeps]

    # 95% Fisher z confidence intervals.
    cis  = [_fisher_z_ci(float(r), n) for r, n in zip(rs, ns)]
    err_lo = np.array([r - lo if not np.isnan(lo) else 0.0
                       for r, (lo, _) in zip(rs, cis)])
    err_hi = np.array([hi - r if not np.isnan(hi) else 0.0
                       for r, (_, hi) in zip(rs, cis)])
    yerr = np.vstack([err_lo, err_hi])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        layers, rs, yerr=yerr,
        fmt="o-", color="#2C7BB6", ecolor="#2C7BB6",
        elinewidth=1.3, capsize=4, markersize=7, linewidth=1.5,
        label="Spearman r (95% Fisher z CI)",
    )

    # Reference lines
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(0.4, color="#D7191C", linestyle="--", linewidth=0.9, alpha=0.7,
               label="r = 0.4 (hypothesis-support threshold)")

    # Annotate each point with its p-value
    for x, y, s in zip(layers, rs, sweeps):
        p = s["spearman_p"]
        if not np.isnan(p):
            ax.annotate(f"p={p:.1e}\nn_q={s.get('n_quantum', '?')}/{s['n_clusters']}",
                        xy=(x, y), xytext=(6, -16), textcoords="offset points",
                        fontsize=7.5, alpha=0.75)

    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(l)) for l in layers])
    ax.set_xlabel("Target Layer", fontsize=10)
    ax.set_ylabel("Spearman r (quantum-ness vs FVU contribution)", fontsize=10)
    ax.set_title(
        f"Quantum-ness vs SAE Reconstruction Error by Layer — {family}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    # Pad y so the CI bars are visible
    finite_lo = np.array([lo for lo, _ in cis if not np.isnan(lo)] or [0.0])
    finite_hi = np.array([hi for _, hi in cis if not np.isnan(hi)] or [0.0])
    y_min = min(-0.15, float(finite_lo.min()) - 0.05)
    y_max = max(0.55,  float(finite_hi.max()) + 0.05)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"layer_sweep_{family}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved layer-sweep plot to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from config import get_active_models

    print("=== Smoke test: visualize ===")
    rng = np.random.default_rng(0)
    models = get_active_models()
    if not models:
        print("No active models.")
        sys.exit(1)

    m = models[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, m["name"])
        os.makedirs(model_dir, exist_ok=True)

        n_c  = 20
        qs   = rng.uniform(0, 1, size=n_c).astype(np.float32)
        neg  = rng.uniform(0, 1, size=n_c).astype(np.float32)
        fvu  = qs * 0.3 + rng.normal(0, 0.05, n_c).astype(np.float32)
        fvu  = np.clip(fvu, 0, 1).astype(np.float32)
        cls  = [
            "quantum"   if q >= 0.30 else
            ("classical" if q < 0.10 else "ambiguous")
            for q in qs
        ]

        geo_results = [
            {
                "cluster_id":               i,
                "quantum_ness_score":        float(qs[i]),
                "negative_weight_fraction":  float(neg[i]),
                "classical_fvu":             float(rng.uniform(0.01, 0.5)),
                "quantum_fvu":               float(rng.uniform(0, 0.3)),
                "classification":            cls[i],
            }
            for i in range(n_c)
        ]
        torch.save(geo_results, os.path.join(model_dir, "geometry.pt"))

        recon_results = {
            "cluster_ids":               list(range(n_c)),
            "quantum_ness_scores":        qs,
            "negative_weight_fractions":  neg,
            "fvu_contributions":          fvu,
            "classifications":            cls,
            "spearman_r":                 0.52,
            "spearman_p":                 0.018,
            "neg_weight_spearman_r":      0.41,
            "neg_weight_spearman_p":      0.035,
            "n_clusters":                 n_c,
        }
        torch.save(recon_results, os.path.join(model_dir, "reconstruction_analysis.pt"))

        visualize_model(m, results_dir=tmpdir)
        print(f"  Files in {model_dir}:")
        for f in sorted(os.listdir(model_dir)):
            print(f"    {f}")

        # Smoke test the layer-sweep plot
        fake_sweeps = [
            {"target_layer":  0, "spearman_r":  0.05, "spearman_p": 0.50,
             "n_clusters": 50, "n_quantum":  4},
            {"target_layer":  6, "spearman_r":  0.22, "spearman_p": 0.10,
             "n_clusters": 50, "n_quantum": 12},
            {"target_layer": 12, "spearman_r":  0.55, "spearman_p": 0.0003,
             "n_clusters": 50, "n_quantum": 28},
            {"target_layer": 18, "spearman_r":  0.41, "spearman_p": 0.003,
             "n_clusters": 50, "n_quantum": 22},
            {"target_layer": 24, "spearman_r":  0.18, "spearman_p": 0.18,
             "n_clusters": 50, "n_quantum":  9},
        ]
        sweep_path = plot_layer_sweep("gemma-2-2b-it", fake_sweeps, results_dir=tmpdir)
        if sweep_path and os.path.exists(sweep_path):
            print(f"  Layer-sweep plot ok: {sweep_path}")

    print("[visualize] Smoke test passed.")
