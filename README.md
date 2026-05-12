# Quantum vs Classical Belief Geometry in SAE Feature Clusters

A mechanistic-interpretability experiment testing whether the **quantum-ness of
geometry** in SAE feature clusters predicts SAE reconstruction error.

## Motivation

Riechers et al. (2025) show that transformers trained on next-token prediction
develop **quantum or post-quantum belief geometries** in their residual streams.
Sparse Autoencoders (SAEs), on the other hand, assume a **classical sparse-linear
decomposition** — every activation is a convex combination of feature directions.

If a feature cluster encodes quantum geometry rather than classical simplex
geometry, an SAE *cannot* fully recover it. The hypothesis tested here is:

> **The quantum-ness of a feature cluster's geometry predicts how much
> reconstruction error the SAE makes inside that cluster's subspace.**

A positive Spearman correlation between cluster-level quantum-ness scores and
cluster-level FVU (fraction of variance unexplained) contributions would support
the hypothesis.

## Pipeline

```
PopQA prompts
    │
    ▼
[data_loader]          Tokenize → DataLoader
    │
    ▼
[sae_extractor]        Forward LLM + SAE → activations, residuals, FVU
    │
    ▼
[clustering]           K-subspace clustering of SAE decoder vectors
    │
    ▼
[geometry_classifier]  AANet archetypes + barycentric coords + density matrix
    │                  → classical / quantum / ambiguous label per cluster
    ▼
[reconstruction_       Project residuals onto each cluster's subspace
 analysis]             → mean FVU contribution per cluster
                       → Spearman correlation with quantum-ness scores
    │
    ▼
[visualize]            Scatter plot + bar chart
```

## File map

| File | Purpose |
| --- | --- |
| `config.py` | Model list, dataset, hyperparameters, `get_active_models()`. |
| `data_loader.py` | Loads PopQA, tokenizes, caches to disk, returns `DataLoader`. |
| `sae_extractor.py` | Runs LLM forward pass via hooks, SAE forward pass, computes FVU. Resumable checkpoints. |
| `clustering.py` | K-subspace clustering on SAE decoder vectors (k-means warm-start + SVD subspace fits + reassignment). |
| `geometry_classifier.py` | **Novel core.** Fits AANet archetypes, computes barycentric coords, density-matrix eigenvalues; classifies each cluster. |
| `reconstruction_analysis.py` | Per-cluster FVU contribution + Spearman correlation with quantum-ness. |
| `visualize.py` | Scatter plot (quantum-ness vs FVU) and bar chart of classifications. |
| `run_experiment.py` | End-to-end runner over `get_active_models()`. |
| `dry_run.py` | Synthetic-data end-to-end sanity check (no network/GPU, ~3 sec). |

## Installation

```bash
pip install -r requirements.txt
```

For HuggingFace models you need an access token:

```bash
huggingface-cli login
# Gemma access requires accepting the license at huggingface.co/google/gemma-2-2b-it
```

## Sanity checks (run these first)

### Level 1 — per-module smoke tests (each runs on fake data in seconds)

```bash
python config.py
python clustering.py
python geometry_classifier.py
python reconstruction_analysis.py
python visualize.py
```

Each module has a `__main__` block that exercises its public API on synthetic
data. No network, no GPU.

### Level 2 — full pipeline dry run (~3 sec, no network/GPU)

```bash
python dry_run.py            # cleans up after itself
python dry_run.py --keep     # inspect artifacts in ./dry_run_results/
```

This generates synthetic activations and runs
`clustering → geometry_classifier → reconstruction_analysis → visualize`
end-to-end. Exits with code 0 on success, 1 on any failure.

### Level 3 — tiny real run (downloads model + SAE)

```bash
python run_experiment.py --n_prompts 10 --n_clusters 8
```

### Level 4 — full experiment

```bash
python run_experiment.py
```

## Configuration

Edit `config.py` to toggle which models run. Only models with `active=True` are
included; default is just `gemma-2-2b-it`.

```python
MODELS = [
    {"name": "gemma-2-2b-it",  ..., "active": True},
    {"name": "gemma-2-9b-it",  ..., "active": False},
    {"name": "llama-3.1-8b",   ..., "active": False},
    {"name": "mistral-7b",     ..., "active": False},
]
```

Other key knobs:

```python
N_PROMPTS = 400              # number of PopQA prompts
N_CLUSTERS = 50              # k for k-subspace clustering
MIN_CLUSTER_SIZE = 10        # discard clusters smaller than this
AANET_N_ARCHETYPES = 4       # vertices of each cluster's simplex
AANET_ITERS = 500            # archetypal-analysis gradient iterations
CHECKPOINT_EVERY = 10        # batches between SAE-extraction checkpoints
```

## Outputs

Each model writes to `results/{model_name}/`:

```
tokenized_dataset.pt        # cached tokenizer output
activations.pt              # feature_activations, sae_reconstruction, residual, fvu_per_prompt
decoder.pt                  # SAE W_dec, cached for clustering
clusters.pt                 # list[dict] of cluster assignments
geometry.pt                 # list[dict] per-cluster classifications
reconstruction_analysis.pt  # Spearman r, p-value, per-cluster fvu_contributions
quantum_vs_fvu.png          # scatter plot
cluster_classifications.png # bar chart
```

## Resumability

`sae_extractor.py` saves an `activations_ckpt.pt` every `CHECKPOINT_EVERY`
batches. If the run crashes during extraction it picks up from the last
completed batch. All other steps skip themselves entirely if their final `.pt`
output already exists.

## The novel part: how the quantum-ness test works

For each feature cluster, `geometry_classifier.py`:

1. **Fits archetypes (AA-Net variant).** Find K vertices of a simplex such that
   every decoder direction in the cluster is approximately a convex combination
   of them. Uses projected gradient descent on two simplex-constrained weight
   matrices.

2. **Computes barycentric coordinates** of each prompt's activation-weighted
   residual-stream projection w.r.t. the fitted simplex.
   - All coordinates ≥ 0 ⟹ the point is *inside* the simplex (classical).
   - Any coordinate < 0 ⟹ the point is *outside* the simplex (requires
     quantum/post-quantum geometry to express).

3. **Quantum-ness score** = fraction of activations with ≥ 1 negative barycentric
   coordinate.

4. **Density-matrix test.** Compute the empirical covariance of the cluster's
   projected activations, normalise to unit trace (density-matrix convention),
   and check the minimum eigenvalue. A valid classical state has all
   eigenvalues ≥ 0; negative values are evidence of non-classical geometry.

5. **Classify** each cluster:
   - **classical** : quantum-ness < 0.10 AND min eigenvalue > -0.05
   - **quantum**   : quantum-ness > 0.30 OR  min eigenvalue < -0.10
   - **ambiguous** : everything else

## Success criterion

> Spearman r > 0.4, p < 0.05 between cluster quantum-ness score and cluster
> FVU contribution on **gemma-2-2b** layer 12.

If this holds, flip the next model's `active` flag to `True` in `config.py`
and rerun.

## Dependencies

See `requirements.txt`:

```
torch, transformers, sae_lens, datasets,
scikit-learn, scipy, numpy, matplotlib, tqdm
```
