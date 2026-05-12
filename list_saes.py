"""
List all SAE IDs available for a given release in the installed SAELens version.

Usage:
    python list_saes.py                              # list all releases
    python list_saes.py gemma-scope-2b-pt-res        # list IDs in one release
    python list_saes.py gemma-scope-2b-pt-res 12     # filter to layer 12
"""

import sys


def get_directory():
    """Try every known import path across SAELens versions."""
    for path in (
        "sae_lens.toolkit.pretrained_saes_directory",
        "sae_lens.pretrained_saes_directory",
        "sae_lens",
    ):
        try:
            mod = __import__(path, fromlist=["get_pretrained_saes_directory"])
            fn = getattr(mod, "get_pretrained_saes_directory", None)
            if fn:
                return fn()
        except ImportError:
            continue

    # SAELens ≥ 4 may expose it differently
    try:
        from sae_lens import pretrained_saes
        fn = getattr(pretrained_saes, "get_pretrained_saes_directory", None)
        if fn:
            return fn()
    except Exception:
        pass

    raise RuntimeError(
        "Cannot locate get_pretrained_saes_directory in this SAELens install.\n"
        "Try: python -c \"from sae_lens import SAE; help(SAE.from_pretrained)\""
    )


def list_releases(directory: dict) -> None:
    print(f"\nAvailable releases ({len(directory)}):")
    for r in sorted(directory.keys()):
        print(f"  {r}")


def list_ids(directory: dict, release: str, layer_filter: str | None = None) -> None:
    if release not in directory:
        print(f"Release '{release}' not found. Available:")
        list_releases(directory)
        sys.exit(1)

    entry = directory[release]
    saes_map = getattr(entry, "saes_map", None) or {}
    ids = sorted(saes_map.keys())

    if layer_filter:
        ids = [i for i in ids if f"layer_{layer_filter}/" in i]
        print(f"\nIDs in '{release}' matching layer {layer_filter} ({len(ids)}):")
    else:
        print(f"\nIDs in '{release}' ({len(ids)}):")

    for sae_id in ids:
        print(f"  {sae_id}")

    if not ids:
        print("  (none — try without the layer filter)")


def main():
    args = sys.argv[1:]
    directory = get_directory()

    if not args:
        list_releases(directory)
        return

    release = args[0]
    layer_filter = args[1] if len(args) > 1 else None
    list_ids(directory, release, layer_filter)


if __name__ == "__main__":
    main()
