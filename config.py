"""
Configuration for the quantum SAE geometry experiment.

Riechers et al. (2025) shows transformers develop quantum/post-quantum belief
geometries. This experiment tests whether that geometry predicts SAE reconstruction
error at the cluster level.

FINDING: U-shaped layer pattern on Gemma-2-2B — quantum geometry predicts SAE
reconstruction failure at boundary layers (0 and 24) but not middle layers.
Intervention test confirms perfect separation (U=100, p=0.0001) at both layers.
"""

RANDOM_SEED = 42

MODELS = [
    # ── Gemma-2-2B — DONE, keep inactive ────────────────────────────────────
    {
        "name": "gemma-2-2b-it-layer-0",
        "hf_name": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_0/width_16k/average_l0_105",   # fixed from auto-select
        "target_layer": 0,
        "dtype": "bfloat16",
        "active": False,
    },
    {
        "name": "gemma-2-2b-it-layer-6",
        "hf_name": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_6/width_16k/average_l0_144",   # fixed from auto-select
        "target_layer": 6,
        "dtype": "bfloat16",
        "active": False,
    },
    {
        "name": "gemma-2-2b-it-layer-12",
        "hf_name": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_12/width_16k/average_l0_176",  # fixed from auto-select
        "target_layer": 12,
        "dtype": "bfloat16",
        "active": False,
    },
    {
        "name": "gemma-2-2b-it-layer-18",
        "hf_name": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_18/width_16k/average_l0_138",  # fixed from auto-select
        "target_layer": 18,
        "dtype": "bfloat16",
        "active": False,
    },
    {
        "name": "gemma-2-2b-it-layer-24",
        "hf_name": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_24/width_16k/average_l0_158",  # fixed from auto-select
        "target_layer": 24,
        "dtype": "bfloat16",
        "active": False,
    },

    # ── Gemma-2-9B — boundary layers only, ACTIVATE NOW ─────────────────────
    {
        "name": "gemma-2-9b-it-layer-0",
        "hf_name": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-pt-res",
        "sae_id": "layer_0/width_16k/average_l0_71",
        "target_layer": 0,
        "dtype": "bfloat16",
        "active": True,
    },
    {
        "name": "gemma-2-9b-it-layer-41",
        "hf_name": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-pt-res",
        "sae_id": "layer_41/width_16k/average_l0_71",
        "target_layer": 41,
        "dtype": "bfloat16",
        "active": True,
    },

    # ── Llama-3.1-8B — boundary layers only, activate after 9B confirms ─────
    {
        "name": "llama-3.1-8b-layer-0",
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "sae_release": "llama_scope_lxr_8x",
        "sae_id": "layer_0/width_16k/average_l0_71",
        "target_layer": 0,
        "dtype": "bfloat16",
        "active": True,
    },
    {
        "name": "llama-3.1-8b-layer-31",
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "sae_release": "llama_scope_lxr_8x",
        "sae_id": "layer_31/width_16k/average_l0_71",
        "target_layer": 31,
        "dtype": "bfloat16",
        "active": True,
    },
]

DATASET = "akariasai/PopQA"
N_PROMPTS = 400
N_CLUSTERS = 50
MIN_CLUSTER_SIZE = 10
RECONSTRUCTION_METRIC = "fvu"
RESULTS_DIR = "./results"
CHECKPOINT_EVERY = 10


def get_active_models():
    """Return only models with active=True."""
    return [m for m in MODELS if m["active"]]


if __name__ == "__main__":
    active = get_active_models()
    print(f"Active models ({len(active)}):")
    for m in active:
        print(f"  {m['name']:<35} hf_name={m['hf_name']}  layer={m['target_layer']}")
    print(f"\nDataset: {DATASET}, N_PROMPTS: {N_PROMPTS}")
    print(f"N_CLUSTERS: {N_CLUSTERS}, SEED: {RANDOM_SEED}")