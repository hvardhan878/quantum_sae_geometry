"""
Main experiment runner.

For each active model in config:
  1. Load and tokenize PopQA prompts.
  2. Extract SAE feature activations, reconstructions, and residuals.
  3. Cluster SAE decoder vectors into subspaces.
  4. Classify the geometry of each cluster (classical / quantum / ambiguous).
  5. Correlate quantum-ness with cluster-level FVU contribution.
  6. Visualise and save results.

When multiple consecutive entries share an hf_name (e.g. a layer sweep over
gemma-2-2b-it at layers 0/6/12/18/24), the HF model is loaded only once and
reused across all of them. After all models complete, a layer-sweep summary
table and plot are produced per hf_name family.

Usage:
    python run_experiment.py
    python run_experiment.py --n_prompts 50          # quick test
"""

import argparse
import os
import sys
import time
import traceback
from collections import Counter

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
from visualize import visualize_model, plot_layer_sweep


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


def _load_hf_model(hf_name: str, dtype_str: str):
    """Load an HF causal-LM in the configured dtype."""
    from transformers import AutoModelForCausalLM
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    print(f"[run_experiment] Loading HF model {hf_name} (dtype={dtype_str}) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"[run_experiment] Loaded {hf_name} in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def run_model(
    model_cfg: dict,
    args: argparse.Namespace,
    cached_model=None,
) -> dict | None:
    """Run the full pipeline for one model. Returns summary dict or None on failure."""
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
    print(f"  cached_model : {'YES (reused)' if cached_model is not None else 'no'}")

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
        print(f"  DataLoader: {len(loader)} batches")
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
            activations = extract_activations(
                model_cfg, loader, results_dir=results_dir, model=cached_model,
            )

        for k, v in activations.items():
            print(f"  {k}: {v.shape}")

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
        total_features = sum(len(cl["feature_indices"]) for cl in clusters)
        print(f"  Clusters retained: {len(clusters)}  ({total_features} features total)")
    except Exception:
        print(f"[ERROR] Clustering failed for {model_name}:")
        traceback.print_exc()
        return None

    # ------------------------------------------------------------------
    # Step 4 — Geometry classification
    # ------------------------------------------------------------------
    _step(4, "Geometry classification")
    try:
        geo_results = classify_all_clusters(
            model_cfg, clusters, activations, results_dir=results_dir,
        )
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
            model_cfg, clusters, geo_results, activations, results_dir=results_dir,
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

    n_quantum = class_counts.get("quantum", 0)
    return {
        "model_name":     model_name,
        "hf_name":        model_cfg["hf_name"],
        "target_layer":   int(model_cfg["target_layer"]),
        "n_clusters":     recon_results["n_clusters"],
        "n_quantum":      n_quantum,
        "spearman_r":     recon_results["spearman_r"],
        "spearman_p":     recon_results["spearman_p"],
        "neg_weight_spearman_r": recon_results.get("neg_weight_spearman_r", float("nan")),
        "neg_weight_spearman_p": recon_results.get("neg_weight_spearman_p", float("nan")),
        "class_counts":   dict(class_counts),
    }


def _save_decoder(model_cfg: dict, results_dir: str) -> None:
    from sae_lens import SAE
    from sae_extractor import _resolve_sae_id
    sae_id = _resolve_sae_id(
        model_cfg["sae_release"], model_cfg["sae_id"], model_cfg["target_layer"]
    )
    sae, _, _ = SAE.from_pretrained(release=model_cfg["sae_release"], sae_id=sae_id)
    dec = sae.W_dec.detach().float()
    decoder_path = os.path.join(results_dir, model_cfg["name"], "decoder.pt")
    torch.save(dec, decoder_path)
    print(f"  Decoder saved: {dec.shape} → {decoder_path}")


# ---------------------------------------------------------------------------
# Layer-sweep summary
# ---------------------------------------------------------------------------

def _short_family_name(hf_name: str) -> str:
    """Turn 'google/gemma-2-2b-it' → 'gemma-2-2b-it'."""
    return hf_name.split("/")[-1]


def _print_layer_sweep_table(family: str, sweeps: list[dict]) -> None:
    sweeps = sorted(sweeps, key=lambda s: s["target_layer"])
    print("\n" + "=" * 70)
    print(f"  LAYER SWEEP SUMMARY — {family}")
    print("=" * 70)
    print(f"  {'Layer':<6}|  {'Spearman r':<12}|  {'p-value':<10}|  n_quantum / n_total")
    print(f"  {'-'*6}|  {'-'*12}|  {'-'*10}|  ---------------------")
    for s in sweeps:
        r = s["spearman_r"]
        p = s["spearman_p"]
        r_str = f"{r:+.3f}"
        p_str = f"{p:.3e}"
        print(f"  {s['target_layer']:<6}|  {r_str:<12}|  {p_str:<10}|  "
              f"{s['n_quantum']:>3} / {s['n_clusters']:<3}")

    # Peak layer = highest |r| (use r itself for positive-correlation signal)
    valid = [s for s in sweeps if s["spearman_r"] == s["spearman_r"]]  # filter NaN
    if valid:
        peak = max(valid, key=lambda s: s["spearman_r"])
        print("=" * 70)
        print(f"  Peak layer: {peak['target_layer']} (r = {peak['spearman_r']:+.3f})")
    print("=" * 70)


def _group_by_family(all_summaries: list[dict]) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = {}
    for s in all_summaries:
        family = _short_family_name(s["hf_name"])
        by_family.setdefault(family, []).append(s)
    return by_family


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

    print(f"Active models ({len(active_models)}):")
    for m in active_models:
        print(f"  {m['name']:30s}  hf_name={m['hf_name']}  layer={m['target_layer']}")

    all_summaries: list[dict] = []
    cached_model = None
    cached_hf_name = None

    for model_cfg in active_models:
        # ---- Reuse the HF model when hf_name matches the previous one ----
        # We only need to actually call _load_hf_model when activations.pt
        # does NOT already exist for this model (otherwise extraction is skipped
        # and the model isn't touched). Defer the load until we know we need it.
        needs_extraction = not os.path.exists(_final_path(model_cfg["name"], args.results_dir))
        if needs_extraction:
            if cached_hf_name != model_cfg["hf_name"]:
                # Drop the previous cached model first to free memory
                if cached_model is not None:
                    print(f"[run_experiment] Releasing cached model "
                          f"{cached_hf_name} (now switching to {model_cfg['hf_name']})")
                    del cached_model
                    cached_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                cached_model = _load_hf_model(model_cfg["hf_name"], model_cfg["dtype"])
                cached_hf_name = model_cfg["hf_name"]
            model_to_pass = cached_model
        else:
            model_to_pass = None  # extraction will be skipped anyway

        summary = run_model(model_cfg, args, cached_model=model_to_pass)
        if summary is not None:
            all_summaries.append(summary)

    # ---- Free cached model before downstream summary work ----
    if cached_model is not None:
        del cached_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Final summary across all completed runs
    # ------------------------------------------------------------------
    _banner("ALL MODELS COMPLETE")
    for s in all_summaries:
        print(
            f"  {s['model_name']:30s}  "
            f"r={s['spearman_r']:+.3f}  p={s['spearman_p']:.3e}  "
            f"n={s['n_clusters']}  {s['class_counts']}"
        )

    # ------------------------------------------------------------------
    # Layer-sweep tables + plots (per hf_name family with >=2 layers)
    # ------------------------------------------------------------------
    by_family = _group_by_family(all_summaries)
    for family, sweeps in by_family.items():
        if len(sweeps) < 2:
            continue
        _print_layer_sweep_table(family, sweeps)
        try:
            plot_path = plot_layer_sweep(family, sweeps, args.results_dir)
            if plot_path:
                print(f"  Layer-sweep plot saved to {plot_path}")
        except Exception:
            print(f"[WARNING] Layer-sweep plot failed for {family}:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
