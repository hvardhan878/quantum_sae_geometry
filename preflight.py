"""
Preflight environment check.

Run this on a fresh machine (e.g. Paperspace) before launching a long experiment.
Verifies:

  1. Python version
  2. Required packages installed
  3. CUDA / GPU available with enough free memory
  4. HuggingFace token configured (gated Gemma access)
  5. Each active model's HF repo is accessible (tokenizer download)
  6. Each active model's SAE release is registered in SAELens
  7. PopQA dataset is accessible
  8. Disk space available

Exits with code 0 on success, 1 if any check fails.

Usage:
    python preflight.py
    python preflight.py --skip_dataset  # skip the dataset round-trip
"""

import argparse
import os
import shutil
import sys
import traceback


# -------------------------- pretty printing --------------------------

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}   {msg}")


def fail(check_name: str, msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")
    failures.append(f"{check_name}: {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# -------------------------- checks --------------------------

def check_python() -> None:
    section("Python version")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        fail("python", f"Python {major}.{minor} too old, need >= 3.9")
    else:
        ok(f"Python {major}.{minor}")


def check_packages() -> None:
    section("Required packages")
    required = [
        "torch", "transformers", "sae_lens", "datasets",
        "sklearn", "scipy", "numpy", "matplotlib", "tqdm",
    ]
    for pkg in required:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg:15s} {ver}")
        except ImportError:
            fail("packages", f"missing: {pkg}  (pip install -r requirements.txt)")


def check_cuda() -> None:
    section("CUDA / GPU")
    try:
        import torch
    except ImportError:
        fail("cuda", "torch not installed")
        return

    if not torch.cuda.is_available():
        warn("CUDA not available — will fall back to CPU (will be SLOW).")
        return

    n_gpu = torch.cuda.device_count()
    ok(f"CUDA available, {n_gpu} device(s)")
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1e9
        free_b, _ = torch.cuda.mem_get_info(i)
        free_gb = free_b / 1e9
        ok(f"  GPU {i}: {name}  |  total {total_gb:.1f} GB  |  free {free_gb:.1f} GB")
        if total_gb < 8:
            warn(f"  GPU {i} has < 8 GB — Gemma-2-2b in bfloat16 needs ~5 GB plus overhead.")


def check_hf_auth() -> None:
    section("HuggingFace authentication")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        ok(f"HF_TOKEN env var set (length {len(token)})")
        return

    try:
        from huggingface_hub import HfFolder
        saved = HfFolder.get_token()
        if saved:
            ok("HF token found on disk (~/.cache/huggingface/token)")
            return
    except Exception:
        pass

    fail("hf_auth",
         "No HF token found. Run `huggingface-cli login` "
         "or `export HF_TOKEN=hf_...`. Required for gated Gemma models.")


def check_models() -> None:
    section("Active model access (tokenizers)")
    try:
        from config import get_active_models
        from transformers import AutoTokenizer
    except ImportError as e:
        fail("models", f"import failed: {e}")
        return

    active = get_active_models()
    if not active:
        fail("models", "No active models in config.py.")
        return

    for m in active:
        try:
            _ = AutoTokenizer.from_pretrained(m["hf_name"], trust_remote_code=True)
            ok(f"{m['name']:20s} tokenizer accessible")
        except Exception as e:
            fail("models", f"{m['name']}: cannot load tokenizer ({e.__class__.__name__}: {e})")


def check_sae_releases() -> None:
    section("SAELens release availability")
    try:
        import sae_lens  # noqa: F401
        from sae_lens import SAE  # noqa: F401
        from config import get_active_models
    except ImportError as e:
        fail("sae_releases", f"import failed: {e}")
        return

    # SAELens has moved the directory module across versions — try several paths.
    get_dir = None
    for import_path in (
        "sae_lens.toolkit.pretrained_saes_directory",
        "sae_lens.pretrained_saes_directory",
        "sae_lens",
    ):
        try:
            mod = __import__(import_path, fromlist=["get_pretrained_saes_directory"])
            get_dir = getattr(mod, "get_pretrained_saes_directory", None)
            if get_dir is not None:
                break
        except ImportError:
            continue

    directory = {}
    if get_dir is None:
        warn("Could not locate SAELens pretrained-SAE directory function "
             "(API may have changed). Skipping release-name validation.")
    else:
        try:
            directory = get_dir()
        except Exception as e:
            warn(f"Could not query SAELens directory: {e}")

    for m in get_active_models():
        if not m.get("sae_release"):
            warn(f"{m['name']}: no SAE configured, will be skipped.")
            continue
        if not directory:
            ok(f"{m['name']:20s} release={m['sae_release']} (not validated — directory unavailable)")
            continue
        if m["sae_release"] in directory:
            entry = directory[m["sae_release"]]
            sae_ids = list(getattr(entry, "saes_map", {}).keys()) if hasattr(entry, "saes_map") else []
            if m["sae_id"] in sae_ids or not sae_ids:
                ok(f"{m['name']:20s} release={m['sae_release']}")
            else:
                fail("sae_releases",
                     f"{m['name']}: sae_id '{m['sae_id']}' not in release '{m['sae_release']}'")
        else:
            warn(f"{m['name']}: release '{m['sae_release']}' not in SAELens directory "
                 f"(this may still work — directory lookup is best-effort)")


def check_dataset() -> None:
    section("PopQA dataset")
    try:
        from datasets import load_dataset
        ds = load_dataset("akariasai/PopQA", split="test")
        ok(f"PopQA loaded, {len(ds)} rows, columns={list(ds.column_names)}")
        if "question" not in ds.column_names:
            fail("dataset", "PopQA missing 'question' column")
    except Exception as e:
        fail("dataset", f"PopQA load failed: {e.__class__.__name__}: {e}")


def check_disk() -> None:
    section("Disk space")
    cwd = os.getcwd()
    total, used, free = shutil.disk_usage(cwd)
    free_gb = free / 1e9
    print(f"  cwd={cwd}")
    print(f"  free: {free_gb:.1f} GB / total: {total/1e9:.1f} GB")
    if free_gb < 20:
        warn(f"Less than 20 GB free — Gemma-2-2b + SAE + activations need ~10–15 GB")
    else:
        ok(f"{free_gb:.1f} GB free")


# -------------------------- main --------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Skip the PopQA round-trip (fastest network step).")
    parser.add_argument("--skip_models", action="store_true",
                        help="Skip downloading tokenizers (needed for gated models).")
    args = parser.parse_args()

    print("=" * 60)
    print("  Quantum SAE Geometry — Preflight Check")
    print("=" * 60)

    check_python()
    check_packages()
    check_cuda()
    check_hf_auth()
    if not args.skip_models:
        check_models()
    check_sae_releases()
    if not args.skip_dataset:
        check_dataset()
    check_disk()

    print("\n" + "=" * 60)
    if failures:
        print(f"  {RED}PREFLIGHT FAILED{RESET}  —  {len(failures)} issue(s):")
        for f_msg in failures:
            print(f"    - {f_msg}")
    else:
        print(f"  {GREEN}PREFLIGHT PASSED{RESET}")
    if warnings:
        print(f"  {YELLOW}Warnings ({len(warnings)}):{RESET}")
        for w_msg in warnings:
            print(f"    - {w_msg}")
    print("=" * 60)

    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(3)
