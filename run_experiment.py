"""
Main experiment runner.

For each active model in config:
  1. Load and tokenize PopQA prompts.
  2. Extract SAE feature activations, reconstructions, and residuals.
  3. Cluster SAE decoder vectors into subspaces.
  4. Classify the geometry of each cluster (classical / quantum / ambiguous).
  5. Correlate quantum-ness with cluster-level FVU contribution.
  6. Visualise and save results.

Usage:
    python run_experiment.py
    python run_experiment.py --n_prompts 50          # quick test
"""

import argparse
import os
import sys
import time
import traceback

import torch

from config import (
    get_active_models,
    RESULTS_DIR,
    N_PROMPTS,
    N_CLUSTERS,
    MIN_CLUSTER_SIZE,
)
from data_loader import build_dataloader
from sae_extractor import extract_activations, _final_path
from clustering import cluster_decoder_vectors
from geometry_classifier import classify_all_clusters
from reconstruction_analysis import run_reconstruction_analysis
from visualize import visualize_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def _step(step_num: int, name: str) -> None:
    print(f"\n[step {step_num}] {name}")
    print("-" * 50)


def run_model(model_cfg: dict, args: argparse.Namespace) -> dict | None:
    """
    Run the full pipeline for a single model. Returns summary dict or None on failure.
    """
    model_name = model_cfg["name"]
    results_dir = args.results_dir
    n_prompts = args.n_prompts
    n_clusters = args.n_clusters

    _banner(f"Model: {model_name}")
    print(f"  hf_name      : {model_cfg['hf_name']}")
    print(f"  sae_release  : {model_cfg['sae_release']}")
    print(f"  sae_id       : {model_cfg['sae_id']}")
    print(f"  target_layer : {model_cfg['target_layer']}")
    print(f"  n_prompts    : {n_prompts}")
    print(f"  n_clusters   : {n_clusters}")
    print(f"  results_dir  : {results_dir}")

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Data loading
    # ------------------------------------------------------------------
    _step(1, "Data loading (PopQA)")
    try:
        loader = build_dataloader(
            model_name=model_name,
            hf_name=model_cfg["hf_name"],
            n_prompts=n_prompts,
            max_length=128,
            batch_size=16,
            results_dir=results_dir,
        )
        n_batches = len(loader)
        print(f"  DataLoader: {n_batches} batches")
    except Exception:
        print(f"[ERROR] Data loading failed for {model_name}:")
        traceback.print_exc()
        return None

    # ------------------------------------------------------------------
    # Step 2 — SAE activation extraction
    # ------------------------------------------------------------------
    _step(2, "SAE activation extraction")
    try:
        activations_path = _final_path(model_name, results_dir)
        if os.path.exists(activations_path):
            print(f"  Checkpoint found — loading from {activations_path}")
            activations = torch.load(activations_path, weights_only=False)
        else:
            activations = extract_activations(model_cfg, loader, results_dir=results_dir)

        for k, v in activations.items():
            print(f"  {k}: {v.shape}")

        # Save the SAE decoder while the SAE is in memory
        # (clustering.py will check for this before reloading the SAE)
        decoder_path = os.path.join(results_dir, model_name, "decoder.pt")
        if not os.path.exists(decoder_path):
            print("  Saving decoder matrix for clustering ...")
            _save_decoder(model_cfg, results_dir)

    except Exception:
        print(f"[ERROR] SAE extraction failed for {model_name}:")
        traceback.print_exc()
        return None

    # ------------------------------------------------------------------
    # Step 3 — Clustering
    # ------------------------------------------------------------------
    _step(3, "K-subspace clustering of SAE decoder vectors")
    try:
        clusters = cluster_decoder_vectors(
            model_cfg,
            results_dir=results_dir,
            n_clusters=n_clusters,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )
        print(f"  Clusters retained: {len(clusters)}")
        total_features = sum(len(cl["feature_indices"]) for cl in clusters)
        print(f"  Total features covered: {total_features}")
    except Exception:
        print(f"[ERROR] Clustering failed for {model_name}:")
        traceback.print_exc()
        return None

    # ------------------------------------------------------------------
    # Step 4 — Geometry classification
    # ------------------------------------------------------------------
    _step(4, "Geometry classification (classical / quantum / ambiguous)")
    try:
        geo_results = classify_all_clusters(
            model_cfg,
            clusters,
            activations["feature_activations"],
            results_dir=results_dir,
        )
        from collections import Counter
        class_counts = Counter(g["classification"] for g in geo_results)
        print(f"  Classification counts: {dict(class_counts)}")
    except Exception:
        print(f"[ERROR] Geometry classification failed for {model_name}:")
        traceback.print_exc()
        return None

    # ------------------------------------------------------------------
    # Step 5 — Reconstruction analysis
    # ------------------------------------------------------------------
    _step(5, "Reconstruction analysis (quantum-ness ↔ FVU)")
    try:
        recon_results = run_reconstruction_analysis(
            model_cfg,
            clusters,
            geo_results,
            activations,
            results_dir=results_dir,
        )
        r = recon_results["spearman_r"]
        p = recon_results["spearman_p"]
        print(f"  Spearman r = {r:.4f},  p = {p:.4e},  n = {recon_results['n_clusters']}")
    except Exception:
        print(f"[ERROR] Reconstruction analysis failed for {model_name}:")
        traceback.print_exc()
        return None

    # ------------------------------------------------------------------
    # Step 6 — Visualisation
    # ------------------------------------------------------------------
    _step(6, "Visualisation")
    try:
        visualize_model(model_cfg, results_dir=results_dir)
    except Exception:
        print(f"[WARNING] Visualisation failed for {model_name} (non-fatal):")
        traceback.print_exc()

    elapsed = time.time() - t_start
    _banner(f"Finished {model_name} in {elapsed:.1f}s")

    return {
        "model_name": model_name,
        "n_clusters": recon_results["n_clusters"],
        "spearman_r": recon_results["spearman_r"],
        "spearman_p": recon_results["spearman_p"],
        "class_counts": dict(class_counts),
    }


def _save_decoder(model_cfg: dict, results_dir: str) -> None:
    """
    Load the SAE and save its decoder weight matrix to disk so that
    clustering.py can use it without re-loading the full SAE.
    """
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(
        release=model_cfg["sae_release"],
        sae_id=model_cfg["sae_id"],
    )
    dec = sae.W_dec.detach().float()  # (n_features, d_model)
    decoder_path = os.path.join(results_dir, model_cfg["name"], "decoder.pt")
    torch.save(dec, decoder_path)
    print(f"  Decoder saved: {dec.shape} → {decoder_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum SAE Geometry Experiment")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS,
                        help="Override N_PROMPTS from config")
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS,
                        help="Override N_CLUSTERS from config")
    parser.add_argument("--force_reload", action="store_true",
                        help="Ignore all checkpoints and rerun from scratch")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    active_models = get_active_models()
    if not active_models:
        print("No active models in config. Set active=True for at least one model.")
        sys.exit(1)

    print(f"Active models: {[m['name'] for m in active_models]}")

    all_summaries = []
    for model_cfg in active_models:
        summary = run_model(model_cfg, args)
        if summary is not None:
            all_summaries.append(summary)

    _banner("ALL MODELS COMPLETE")
    for s in all_summaries:
        print(
            f"  {s['model_name']:30s}  "
            f"r={s['spearman_r']:+.3f}  p={s['spearman_p']:.3e}  "
            f"n={s['n_clusters']}  {s['class_counts']}"
        )


if __name__ == "__main__":
    main()
