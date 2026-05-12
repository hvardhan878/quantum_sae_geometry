"""
Configuration for the quantum SAE geometry experiment.

Riechers et al. (2025) shows transformers develop quantum/post-quantum belief
geometries. This experiment tests whether that geometry predicts SAE reconstruction
error at the cluster level.
"""

MODELS = [
    {
        "name": "gemma-2-2b-it",
        "hf_name": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_12/width_16k/average_l0_71",
        "target_layer": 12,
        "dtype": "bfloat16",
        "active": True,
    },
    {
        "name": "gemma-2-9b-it",
        "hf_name": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-pt-res",
        "sae_id": "layer_20/width_16k/average_l0_71",
        "target_layer": 20,
        "dtype": "bfloat16",
        "active": False,
    },
    {
        "name": "llama-3.1-8b",
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "sae_release": "llama_scope_lxr_8x",
        "sae_id": "layer_16/width_16k/average_l0_71",
        "target_layer": 16,
        "dtype": "bfloat16",
        "active": False,
    },
    {
        "name": "mistral-7b",
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "sae_release": None,
        "sae_id": None,
        "target_layer": 16,
        "dtype": "bfloat16",
        "active": False,
    },
]

DATASET = "akariasai/PopQA"
N_PROMPTS = 400
N_CLUSTERS = 50
MIN_CLUSTER_SIZE = 10
SIMPLEX_K_VALUES = [3, 4, 5]
AANET_ITERS = 500
AANET_N_ARCHETYPES = 4
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
        print(f"  {m['name']} — layer {m['target_layer']} — SAE: {m['sae_id']}")
    print(f"\nDataset: {DATASET}, N_PROMPTS: {N_PROMPTS}")
    print(f"N_CLUSTERS: {N_CLUSTERS}, MIN_CLUSTER_SIZE: {MIN_CLUSTER_SIZE}")
    print(f"AANET: {AANET_N_ARCHETYPES} archetypes, {AANET_ITERS} iters")
