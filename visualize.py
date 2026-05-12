"""
Visualisation for the quantum SAE geometry experiment.

Produces two plots per model:
  1. Scatter: quantum-ness score vs. mean FVU contribution (coloured by class).
  2. Bar chart: count of classical / quantum / ambiguous clusters.

Both saved under results/{model_name}/.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import RESULTS_DIR

# Colour mapping for classification labels
CLASS_COLORS = {
    "classical": "#2C7BB6",   # blue
    "quantum":   "#D7191C",   # red
    "ambiguous": "#808080",   # grey
}


def _load_results(model_name: str, results_dir: str) -> tuple[dict | None, dict | None]:
    geo_path = os.path.join(results_dir, model_name, "geometry.pt")
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


def plot_quantum_vs_fvu(
    model_cfg: dict,
    results_dir: str = RESULTS_DIR,
) -> str | None:
    """
    Scatter plot of quantum-ness score vs FVU contribution per cluster.
    Returns the save path, or None if data is missing.
    """
    _, recon = _load_results(model_cfg["name"], results_dir)
    if recon is None:
        print(f"[visualize] Cannot plot — missing reconstruction_analysis.pt")
        return None

    qs = np.array(recon["quantum_ness_scores"], dtype=float)
    fvu = np.array(recon["fvu_contributions"], dtype=float)
    labels = recon["classifications"]
    r = recon["spearman_r"]
    p = recon["spearman_p"]

    colors = [CLASS_COLORS.get(lbl, "#808080") for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(qs, fvu, c=colors, alpha=0.75, edgecolors="white", linewidths=0.4, s=60)

    # Legend patches
    patches = [
        mpatches.Patch(color=CLASS_COLORS["classical"], label="Classical"),
        mpatches.Patch(color=CLASS_COLORS["quantum"],   label="Quantum"),
        mpatches.Patch(color=CLASS_COLORS["ambiguous"], label="Ambiguous"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    # Spearman annotation
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    ax.text(
        0.98, 0.97,
        f"Spearman r = {r:.3f}\np = {p_str}\nn = {len(qs)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Quantum-ness Score\n(von Neumann entropy of decoder covariance / log rank)", fontsize=10)
    ax.set_ylabel("Mean Cluster FVU Contribution\n(fraction of residual variance in subspace)", fontsize=10)
    ax.set_title(f"{model_cfg['name']} — Quantum Geometry vs SAE Reconstruction Error", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    save_path = os.path.join(results_dir, model_cfg["name"], "quantum_vs_fvu.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved scatter plot to {save_path}")
    return save_path


def plot_cluster_classifications(
    model_cfg: dict,
    results_dir: str = RESULTS_DIR,
) -> str | None:
    """
    Bar chart showing count of classical / quantum / ambiguous clusters.
    Returns the save path, or None if data is missing.
    """
    geo, _ = _load_results(model_cfg["name"], results_dir)
    if geo is None:
        return None

    counts = {"classical": 0, "quantum": 0, "ambiguous": 0}
    for g in geo:
        counts[g["classification"]] = counts.get(g["classification"], 0) + 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
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
    ax.set_title(f"{model_cfg['name']} — Cluster Geometry Classifications", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2 + 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    save_path = os.path.join(results_dir, model_cfg["name"], "cluster_classifications.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved bar chart to {save_path}")
    return save_path


def visualize_model(model_cfg: dict, results_dir: str = RESULTS_DIR) -> None:
    """Generate all plots for a single model."""
    print(f"[visualize] Generating plots for {model_cfg['name']} ...")
    plot_quantum_vs_fvu(model_cfg, results_dir)
    plot_cluster_classifications(model_cfg, results_dir)


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

        n_c = 20
        qs = rng.uniform(0, 1, size=n_c).astype(np.float32)
        fvu = qs * 0.3 + rng.normal(0, 0.05, n_c).astype(np.float32)
        fvu = np.clip(fvu, 0, 1)
        cls = [
            "quantum" if q > 0.3 else ("classical" if q < 0.1 else "ambiguous")
            for q in qs
        ]

        # Fake geometry results
        geo_results = [
            {"cluster_id": i, "quantum_ness_score": float(qs[i]),
             "min_eigenvalue": float(rng.uniform(-0.2, 0.1)),
             "classification": cls[i]}
            for i in range(n_c)
        ]
        torch.save(geo_results, os.path.join(model_dir, "geometry.pt"))

        # Fake reconstruction analysis results
        recon_results = {
            "cluster_ids": list(range(n_c)),
            "quantum_ness_scores": qs,
            "fvu_contributions": fvu,
            "classifications": cls,
            "spearman_r": 0.52,
            "spearman_p": 0.018,
            "n_clusters": n_c,
        }
        torch.save(recon_results, os.path.join(model_dir, "reconstruction_analysis.pt"))

        visualize_model(m, results_dir=tmpdir)
        print(f"  Files in {model_dir}:")
        for f in os.listdir(model_dir):
            print(f"    {f}")

    print("[visualize] Smoke test passed.")
