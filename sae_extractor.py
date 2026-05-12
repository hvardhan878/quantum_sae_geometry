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

from config import (
    RESULTS_DIR,
    CHECKPOINT_EVERY,
    RECONSTRUCTION_METRIC,
)


# ---------------------------------------------------------------------------
# SAE ID resolution — handles version differences in SAELens release catalogs
# ---------------------------------------------------------------------------

def _resolve_sae_id(release: str, configured_id: str, target_layer: int) -> str:
    """
    Return the best matching SAE ID for the given release.

    1. Try the configured ID first (fast path, no network).
    2. If it's not in the catalog, query SAELens for all IDs in the release
       and pick the one that best matches target_layer + configured width/l0.
    3. Print a clear diff so the user can update config.py.
    """
    # Attempt to get the catalog without triggering a download
    catalog = _get_saelens_catalog(release)
    if catalog is None:
        # Catalog unavailable — pass the configured ID through and let SAELens
        # raise a descriptive error if it is wrong
        return configured_id

    if configured_id in catalog:
        return configured_id

    # ---- Configured ID not in catalog — auto-select best match ----
    layer_prefix = f"layer_{target_layer}/"
    layer_ids = [i for i in catalog if i.startswith(layer_prefix)]

    if not layer_ids:
        # No IDs for this layer at all — print full catalog and raise
        print(f"[sae_extractor] ERROR: No SAE IDs found for layer {target_layer} "
              f"in release '{release}'.")
        print(f"[sae_extractor] Full catalog ({len(catalog)} entries):")
        for cid in sorted(catalog):
            print(f"  {cid}")
        raise ValueError(
            f"Release '{release}' has no SAEs for layer {target_layer}. "
            f"Run `python list_saes.py {release}` to see what is available."
        )

    # Score by how closely each candidate matches the configured_id tokens
    configured_tokens = set(configured_id.replace("/", "_").split("_"))

    def _score(cid: str) -> int:
        tokens = set(cid.replace("/", "_").split("_"))
        return len(tokens & configured_tokens)

    best = max(layer_ids, key=_score)
    print(f"[sae_extractor] WARNING: SAE ID '{configured_id}' not found in release '{release}'.")
    print(f"[sae_extractor] Auto-selected closest match: '{best}'")
    print(f"[sae_extractor] To silence this warning update sae_id in config.py to: '{best}'")
    print(f"[sae_extractor] All layer-{target_layer} options:")
    for cid in sorted(layer_ids):
        marker = "  >>> " if cid == best else "      "
        print(f"{marker}{cid}")
    return best


def _get_saelens_catalog(release: str) -> list[str] | None:
    """
    Return list of SAE IDs for `release`, or None if catalog is unavailable.

    Strategy 1: read pretrained_saes.yaml directly from the installed package.
    Strategy 2: use the directory API (older SAELens).
    Strategy 3: probe with a sentinel ID and parse the truncated ValueError.
    """
    # Strategy 1 — read the YAML file that ships with every SAELens version
    try:
        import sae_lens as _sl
        yaml_path = os.path.join(os.path.dirname(_sl.__file__), "pretrained_saes.yaml")
        if os.path.exists(yaml_path):
            ids = _ids_from_yaml(yaml_path, release)
            if ids is not None:
                return ids
    except Exception:
        pass

    # Strategy 2 — directory API (older SAELens)
    for import_path in (
        "sae_lens.toolkit.pretrained_saes_directory",
        "sae_lens.pretrained_saes_directory",
    ):
        try:
            mod = __import__(import_path, fromlist=["get_pretrained_saes_directory"])
            fn = getattr(mod, "get_pretrained_saes_directory", None)
            if fn is None:
                continue
            directory = fn()
            if release not in directory:
                return []
            entry = directory[release]
            saes_map = getattr(entry, "saes_map", None) or {}
            return list(saes_map.keys())
        except Exception:
            continue

    # Strategy 3 — parse the (potentially truncated) ValueError from a probe
    import re
    try:
        from sae_lens import SAE
        SAE.from_pretrained(release=release, sae_id="__probe__")
    except ValueError as e:
        msg = str(e)
        ids = re.findall(r"'((?:layer|embedding)[^']+)'", msg)
        if ids:
            return ids
    except Exception:
        pass

    return None


def _ids_from_yaml(yaml_path: str, release: str) -> list[str] | None:
    """Parse pretrained_saes.yaml and return IDs for the given release."""
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except ImportError:
        # PyYAML unavailable — minimal line parser
        data = {}
        current = None
        in_saes = False
        with open(yaml_path) as f:
            for line in f:
                s = line.rstrip()
                if not s or s.startswith("#"):
                    continue
                indent = len(s) - len(s.lstrip())
                c = s.lstrip()
                if indent == 0 and c.endswith(":"):
                    current = c[:-1]
                    data[current] = []
                    in_saes = False
                elif indent == 2 and c == "saes:":
                    in_saes = True
                elif indent == 2 and c != "saes:":
                    in_saes = False
                elif in_saes and c.startswith("- id:"):
                    if current:
                        data[current].append(c[len("- id:"):].strip())
    except Exception:
        return None

    if release not in data:
        return None

    entry = data[release]
    if isinstance(entry, list):
        return [str(i) for i in entry]
    elif isinstance(entry, dict):
        saes = entry.get("saes", [])
        return [s["id"] for s in saes if isinstance(s, dict) and "id" in s]
    return None


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
    sae_release = model_cfg["sae_release"]
    sae_id = _resolve_sae_id(sae_release, model_cfg["sae_id"], model_cfg["target_layer"])
    print(f"[sae_extractor] Loading SAE {sae_release} / {sae_id} ...")
    sae, cfg_dict, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
    sae = sae.to(device)
    sae.eval()
    # Probe the SAE's expected input dtype by looking at its decoder weight
    sae_dtype = sae.W_dec.dtype
    print(f"[sae_extractor] SAE loaded. d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, dtype={sae_dtype}")

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

        # ---- SAE forward (use encode/decode for stable API across SAELens versions) ----
        with torch.no_grad():
            sae_input = last_hidden.to(sae_dtype)
            feat_acts = sae.encode(sae_input)              # (batch, n_features)
            recon = sae.decode(feat_acts)                  # (batch, d_model)
        feat_acts = feat_acts.float()
        recon = recon.float()

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
    from config import get_active_models
    from data_loader import build_dataloader

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
