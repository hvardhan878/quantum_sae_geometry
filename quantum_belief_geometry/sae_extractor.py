"""
SAE feature extractor.

For each prompt:
  1. Forward-pass the LLM and grab last-token hidden states at target_layer
     via a register_forward_hook on model.model.layers[target_layer].
  2. Run the SAELens SAE on those hidden states to obtain:
       - feature_activations  : (n_prompts, n_features)   sparse SAE codes
       - sae_reconstruction   : (n_prompts, d_model)       SAE output
       - residual             : (n_prompts, d_model)       hidden - reconstruction
       - fvu_per_prompt       : (n_prompts,)               fraction of variance unexplained
  3. Checkpoint every CHECKPOINT_EVERY batches so the run is resumable.
  4. Save final tensors to results/{model_name}/activations.pt.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_belief_geometry.config import (
    RESULTS_DIR,
    CHECKPOINT_EVERY,
    RECONSTRUCTION_METRIC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_results_dir(model_name: str, results_dir: str = RESULTS_DIR) -> str:
    path = os.path.join(results_dir, model_name)
    os.makedirs(path, exist_ok=True)
    return path


def _checkpoint_path(model_name: str, results_dir: str) -> str:
    return os.path.join(_get_results_dir(model_name, results_dir), "activations_ckpt.pt")


def _final_path(model_name: str, results_dir: str) -> str:
    return os.path.join(_get_results_dir(model_name, results_dir), "activations.pt")


def _fvu(original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """
    Fraction of Variance Unexplained per sample.

    FVU = ||x - x_hat||^2 / ||x - mean(x)||^2

    Returns a 1-D tensor of shape (batch,).
    """
    residual = original - reconstruction
    var_residual = (residual ** 2).sum(dim=-1)
    mean_x = original.mean(dim=0, keepdim=True)
    var_total = ((original - mean_x) ** 2).sum(dim=-1)
    # Guard against zero-variance inputs
    var_total = var_total.clamp(min=1e-8)
    return var_residual / var_total


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model_cfg: dict,
    dataloader,
    results_dir: str = RESULTS_DIR,
    device: str | None = None,
) -> dict:
    """
    Run the full extraction pipeline for one model config.

    Returns a dict with keys:
        feature_activations, sae_reconstruction, residual, fvu_per_prompt
    """
    final_path = _final_path(model_cfg["name"], results_dir)
    if os.path.exists(final_path):
        print(f"[sae_extractor] Final activations already exist at {final_path}, loading.")
        return torch.load(final_path, weights_only=False)

    from transformers import AutoModelForCausalLM
    from sae_lens import SAE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sae_extractor] Using device: {device}")

    # ---- Load LLM ----
    print(f"[sae_extractor] Loading model {model_cfg['hf_name']} ...")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg["dtype"], torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_name"],
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ---- Load SAE ----
    print(f"[sae_extractor] Loading SAE {model_cfg['sae_release']} / {model_cfg['sae_id']} ...")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=model_cfg["sae_release"],
        sae_id=model_cfg["sae_id"],
    )
    sae = sae.to(device)
    sae.eval()
    print(f"[sae_extractor] SAE loaded. d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # ---- Hook setup ----
    target_layer = model_cfg["target_layer"]
    _hook_storage: dict = {}

    def _hook_fn(module, input, output):
        # output is typically a tuple; first element is the hidden state tensor
        hidden = output[0] if isinstance(output, tuple) else output
        _hook_storage["hidden"] = hidden.detach()

    hook_handle = model.model.layers[target_layer].register_forward_hook(_hook_fn)

    # ---- Checkpoint resume ----
    ckpt_path = _checkpoint_path(model_cfg["name"], results_dir)
    all_feat_acts = []
    all_recons = []
    all_residuals = []
    all_fvus = []
    start_batch = 0

    if os.path.exists(ckpt_path):
        print(f"[sae_extractor] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        all_feat_acts = ckpt["feature_activations"]
        all_recons = ckpt["sae_reconstruction"]
        all_residuals = ckpt["residual"]
        all_fvus = ckpt["fvu_per_prompt"]
        start_batch = ckpt["next_batch"]
        print(f"[sae_extractor] Resuming from batch {start_batch} "
              f"({len(all_fvus)} prompts already processed)")

    # ---- Extraction loop ----
    batches = list(enumerate(dataloader))
    first_batch_done = False

    for batch_idx, (input_ids, attention_mask) in tqdm(
        batches[start_batch:], desc="Extracting", initial=start_batch, total=len(batches)
    ):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # hidden: (batch, seq_len, d_model) — take last non-pad token
        hidden = _hook_storage["hidden"]  # (batch, seq_len, d_model)

        # Find last non-padding token index per sample
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        last_hidden = hidden[
            torch.arange(hidden.size(0), device=device), seq_lengths
        ]  # (batch, d_model)
        last_hidden_f32 = last_hidden.float()

        # ---- SAE forward ----
        with torch.no_grad():
            sae_out = sae(last_hidden.to(sae.dtype if hasattr(sae, "dtype") else torch.float32))

        # SAELens SAE.forward returns an SAEOutput namedtuple-like object
        # with fields: feature_acts, sae_out (reconstruction)
        if hasattr(sae_out, "feature_acts"):
            feat_acts = sae_out.feature_acts.float()       # (batch, n_features)
            recon = sae_out.sae_out.float()                # (batch, d_model)
        else:
            # Fallback: tuple output (feature_acts, reconstruction)
            feat_acts = sae_out[0].float()
            recon = sae_out[1].float()

        residual = last_hidden_f32 - recon                 # (batch, d_model)
        fvu = _fvu(last_hidden_f32, recon)                 # (batch,)

        if not first_batch_done:
            print(f"\n[sae_extractor] First batch shapes:")
            print(f"  last_hidden:    {last_hidden_f32.shape}")
            print(f"  feat_acts:      {feat_acts.shape}")
            print(f"  recon:          {recon.shape}")
            print(f"  residual:       {residual.shape}")
            print(f"  fvu:            {fvu.shape}")
            print(f"  fvu mean:       {fvu.mean().item():.4f}")
            first_batch_done = True

        all_feat_acts.append(feat_acts.cpu())
        all_recons.append(recon.cpu())
        all_residuals.append(residual.cpu())
        all_fvus.append(fvu.cpu())

        # Checkpoint
        global_batch = start_batch + batch_idx
        if (global_batch + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(
                ckpt_path,
                all_feat_acts, all_recons, all_residuals, all_fvus,
                next_batch=global_batch + 1,
            )

    hook_handle.remove()

    # ---- Concatenate and save ----
    results = {
        "feature_activations": torch.cat(all_feat_acts, dim=0),   # (N, n_features)
        "sae_reconstruction": torch.cat(all_recons, dim=0),        # (N, d_model)
        "residual": torch.cat(all_residuals, dim=0),               # (N, d_model)
        "fvu_per_prompt": torch.cat(all_fvus, dim=0),              # (N,)
    }

    print(f"\n[sae_extractor] Final tensor shapes:")
    for k, v in results.items():
        print(f"  {k}: {v.shape}")

    torch.save(results, final_path)
    print(f"[sae_extractor] Saved activations to {final_path}")

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return results


def _save_checkpoint(path, feat_acts, recons, residuals, fvus, next_batch):
    ckpt = {
        "feature_activations": feat_acts,
        "sae_reconstruction": recons,
        "residual": residuals,
        "fvu_per_prompt": fvus,
        "next_batch": next_batch,
    }
    torch.save(ckpt, path)
    print(f"[sae_extractor] Checkpoint saved at batch {next_batch}")


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from quantum_belief_geometry.config import get_active_models
    from quantum_belief_geometry.data_loader import build_dataloader

    models = get_active_models()
    if not models:
        print("No active models.")
        sys.exit(1)

    m = models[0]
    print(f"=== Smoke test: {m['name']} (10 prompts) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = build_dataloader(
            model_name=m["name"],
            hf_name=m["hf_name"],
            n_prompts=10,
            max_length=64,
            batch_size=4,
            results_dir=tmpdir,
        )
        results = extract_activations(m, loader, results_dir=tmpdir)
        for k, v in results.items():
            print(f"  {k}: {v.shape}, dtype={v.dtype}")

    print("[sae_extractor] Smoke test passed.")
