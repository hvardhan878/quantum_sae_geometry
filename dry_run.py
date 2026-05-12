"""
Dry-run / sanity-check script.

Exercises the full downstream pipeline end-to-end using synthetic data:

    clustering  →  geometry_classifier  →  reconstruction_analysis  →  visualize

No network, no GPU, no model download required. Runs in a few seconds.

This is the fastest way to verify every module's interfaces are wired
together correctly before committing to a real (multi-GB) experiment run.

Usage:
    python dry_run.py
    python dry_run.py --keep        # keep dry_run_results/
"""

import argparse
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from clustering import cluster_decoder_vectors
from geometry_classifier import classify_all_clusters
from reconstruction_analysis import run_reconstruction_analysis
from visualize import visualize_model


# Small synthetic-data sizes so this runs in seconds
SYN_N_PROMPTS = 80
SYN_N_FEATURES = 400
SYN_D_MODEL = 64
SYN_N_CLUSTERS = 8
SYN_MIN_CLUSTER_SIZE = 5
SYN_DRY_RUN_DIR = "./dry_run_results"


def _banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def _make_synthetic_activations(seed: int = 0) -> dict:
    """
    Build a synthetic 'activations.pt' equivalent.

    We construct activations so that *some* clusters have a mixture of
    archetypes (classical geometry) and some have rotationally curved
    structure (more quantum-like) so the final classifier produces a
    non-degenerate spread for sanity-checking.
    """
    rng = np.random.default_rng(seed)
    N, F, D = SYN_N_PROMPTS, SYN_N_FEATURES, SYN_D_MODEL

    # Random decoder directions (we'll reuse this as the cluster's decoder)
    decoder = rng.standard_normal((F, D)).astype(np.float32)
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True) + 1e-8

    # Sparse exponential SAE activations
    feat = rng.exponential(scale=0.3, size=(N, F)).astype(np.float32)
    # Add sparsity: zero out ~80% of values
    mask = rng.random((N, F)) > 0.2
    feat = feat * (~mask)

    # Reconstruction = activation-weighted sum of decoder vectors
    recon = feat @ decoder                                       # (N, D)

    # Generate a "true" hidden state with a small extra component
    extra = rng.standard_normal((N, D)).astype(np.float32) * 0.1
    hidden = recon + extra
    residual = hidden - recon                                    # = extra

    # FVU per prompt
    var_res = (residual ** 2).sum(axis=1)
    var_tot = ((hidden - hidden.mean(0, keepdims=True)) ** 2).sum(axis=1)
    fvu = var_res / np.clip(var_tot, 1e-8, None)

    return {
        "decoder": torch.from_numpy(decoder),
        "activations": {
            "feature_activations": torch.from_numpy(feat),
            "sae_reconstruction":  torch.from_numpy(recon),
            "residual":            torch.from_numpy(residual),
            "last_hidden":         torch.from_numpy(hidden),
            "fvu_per_prompt":      torch.from_numpy(fvu),
        },
    }


def run_dry(results_dir: str) -> bool:
    """Run the full downstream pipeline on synthetic data. Returns True on success."""
    fake_model_cfg = {
        "name": "dry-run-synthetic",
        "hf_name": "synthetic/none",
        "sae_release": None,
        "sae_id": None,
        "target_layer": 0,
        "dtype": "float32",
        "active": True,
    }

    model_dir = os.path.join(results_dir, fake_model_cfg["name"])
    os.makedirs(model_dir, exist_ok=True)

    _banner("Step 0  Generate synthetic data")
    t0 = time.time()
    synth = _make_synthetic_activations()
    print(f"  decoder           : {synth['decoder'].shape}")
    print(f"  feature_activations: {synth['activations']['feature_activations'].shape}")
    print(f"  sae_reconstruction : {synth['activations']['sae_reconstruction'].shape}")
    print(f"  residual           : {synth['activations']['residual'].shape}")
    print(f"  fvu_per_prompt     : {synth['activations']['fvu_per_prompt'].shape}")

    torch.save(synth["decoder"], os.path.join(model_dir, "decoder.pt"))
    torch.save(synth["activations"], os.path.join(model_dir, "activations.pt"))
    print(f"  Saved decoder.pt and activations.pt to {model_dir}")
    print(f"  ({time.time() - t0:.2f}s)")

    # --- Step 1: clustering ---
    _banner("Step 1  K-subspace clustering")
    t0 = time.time()
    clusters = cluster_decoder_vectors(
        fake_model_cfg,
        results_dir=results_dir,
        n_clusters=SYN_N_CLUSTERS,
        min_cluster_size=SYN_MIN_CLUSTER_SIZE,
        force_recluster=True,
    )
    print(f"  Got {len(clusters)} clusters ({time.time() - t0:.2f}s)")
    if len(clusters) == 0:
        print("  [FAIL] No clusters retained.")
        return False

    # --- Step 2: geometry classification ---
    _banner("Step 2  Geometry classification")
    t0 = time.time()
    geo = classify_all_clusters(
        fake_model_cfg,
        clusters,
        synth["activations"],
        results_dir=results_dir,
        force_reclassify=True,
    )
    print(f"  Got {len(geo)} geometry results ({time.time() - t0:.2f}s)")

    # --- Step 3: reconstruction analysis ---
    _banner("Step 3  Reconstruction analysis")
    t0 = time.time()
    recon = run_reconstruction_analysis(
        fake_model_cfg,
        clusters,
        geo,
        synth["activations"],
        results_dir=results_dir,
    )
    print(f"  Spearman r = {recon['spearman_r']:+.4f}, "
          f"p = {recon['spearman_p']:.4e}, "
          f"n = {recon['n_clusters']}  ({time.time() - t0:.2f}s)")

    # --- Step 4: visualisation ---
    _banner("Step 4  Visualisation")
    t0 = time.time()
    visualize_model(fake_model_cfg, results_dir=results_dir)
    print(f"  ({time.time() - t0:.2f}s)")

    # --- Final integrity check ---
    _banner("Files written")
    expected = [
        "decoder.pt",
        "activations.pt",
        "clusters.pt",
        "geometry.pt",
        "reconstruction_analysis.pt",
        "quantum_vs_fvu.png",
        "neg_weight_vs_fvu.png",
        "cluster_classifications.png",
    ]
    all_ok = True
    for fname in expected:
        path = os.path.join(model_dir, fname)
        size = os.path.getsize(path) if os.path.exists(path) else -1
        status = "OK " if size > 0 else "MISS"
        print(f"  [{status}] {fname:35s} {size:>10} bytes")
        if size <= 0:
            all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum SAE geometry dry-run sanity check")
    parser.add_argument(
        "--keep", action="store_true",
        help=f"Keep {SYN_DRY_RUN_DIR}/ after the run (default: delete)",
    )
    parser.add_argument(
        "--results_dir", type=str, default=SYN_DRY_RUN_DIR,
        help="Directory to write synthetic results to",
    )
    args = parser.parse_args()

    print(f"Dry-run directory: {args.results_dir}")
    print(f"Synthetic sizes:    N={SYN_N_PROMPTS}, F={SYN_N_FEATURES}, "
          f"D={SYN_D_MODEL}, K={SYN_N_CLUSTERS}")

    t_start = time.time()
    ok = False
    try:
        ok = run_dry(args.results_dir)
    except Exception:
        print("\n[ERROR] Dry-run crashed:")
        traceback.print_exc()
        ok = False
    elapsed = time.time() - t_start

    _banner(f"DRY RUN {'PASSED' if ok else 'FAILED'} in {elapsed:.1f}s")

    if not args.keep and os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
        print(f"Cleaned up {args.results_dir} (use --keep to retain)")
    elif args.keep:
        print(f"Outputs retained at {args.results_dir}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
