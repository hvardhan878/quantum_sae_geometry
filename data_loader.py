"""
Data loader for the quantum SAE geometry experiment.

Loads PopQA from HuggingFace, samples N_PROMPTS questions, tokenizes them,
and returns a DataLoader. Caches tokenized dataset to disk.
"""

import os
import hashlib
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import sys
from config import DATASET, N_PROMPTS, RESULTS_DIR


def get_cache_path(model_name: str, results_dir: str = RESULTS_DIR) -> str:
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, "tokenized_dataset.pt")


def load_popqa_prompts(n_prompts: int = N_PROMPTS, seed: int = 42) -> list[str]:
    """Download PopQA and extract the first n_prompts question strings."""
    print(f"[data_loader] Loading {DATASET} ...")
    dataset = load_dataset(DATASET, split="test")
    print(f"[data_loader] Dataset size: {len(dataset)} rows")

    # Shuffle deterministically then take n_prompts
    dataset = dataset.shuffle(seed=seed)
    questions = dataset["question"][:n_prompts]
    print(f"[data_loader] Sampled {len(questions)} questions")
    return questions


def build_dataloader(
    model_name: str,
    hf_name: str,
    n_prompts: int = N_PROMPTS,
    max_length: int = 128,
    batch_size: int = 16,
    results_dir: str = RESULTS_DIR,
    force_reload: bool = False,
) -> DataLoader:
    """
    Tokenize the PopQA questions for the given model and return a DataLoader.
    Caches the tokenized tensors to disk so subsequent runs skip tokenization.
    """
    cache_path = get_cache_path(model_name, results_dir)

    if not force_reload and os.path.exists(cache_path):
        print(f"[data_loader] Loading tokenized dataset from cache: {cache_path}")
        saved = torch.load(cache_path, weights_only=True)
        input_ids = saved["input_ids"]
        attention_mask = saved["attention_mask"]

        # Validate that the cache was built with right-padding. If any row's
        # first attention_mask entry is 0 then it was built with left padding
        # (under the old buggy data_loader) and must be regenerated.
        first_col = attention_mask[:, 0]
        if (first_col == 0).any():
            print(f"[data_loader] Cache appears to be LEFT-padded "
                  f"(stale from old data_loader) — regenerating.")
            os.remove(cache_path)
        else:
            print(f"[data_loader] input_ids shape: {input_ids.shape}")
            print(f"[data_loader] attention_mask shape: {attention_mask.shape}")
            dataset = TensorDataset(input_ids, attention_mask)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    questions = load_popqa_prompts(n_prompts=n_prompts)

    print(f"[data_loader] Loading tokenizer for {hf_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)

    # Many instruction-tuned models lack a dedicated pad token; reuse EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force right padding — Gemma-2-it tokenizers default to LEFT padding, which
    # makes "last token" extraction fragile. With right padding the last real
    # token is consistently at attention_mask.sum(dim=1) - 1.
    tokenizer.padding_side = "right"

    print(f"[data_loader] Tokenizing {len(questions)} prompts "
          f"(max_length={max_length}, padding_side=right) ...")
    encoded = tokenizer(
        list(questions),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]          # (n_prompts, max_length)
    attention_mask = encoded["attention_mask"] # (n_prompts, max_length)

    print(f"[data_loader] input_ids shape:      {input_ids.shape}")
    print(f"[data_loader] attention_mask shape: {attention_mask.shape}")

    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, cache_path)
    print(f"[data_loader] Saved tokenized dataset to {cache_path}")

    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    # Quick smoke-test on 10 prompts using the first active model
    from config import get_active_models
    import tempfile

    models = get_active_models()
    if not models:
        print("No active models — check config.py")
        sys.exit(1)

    m = models[0]
    print(f"=== Smoke test: {m['name']} ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = build_dataloader(
            model_name=m["name"],
            hf_name=m["hf_name"],
            n_prompts=10,
            max_length=64,
            batch_size=4,
            results_dir=tmpdir,
        )
        for batch_idx, (ids, mask) in enumerate(loader):
            print(f"  batch {batch_idx}: ids={ids.shape}, mask={mask.shape}")

    print("[data_loader] Smoke test passed.")
