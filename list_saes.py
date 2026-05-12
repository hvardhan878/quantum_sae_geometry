"""
List all SAE IDs available for a given release in the installed SAELens version.

Works by probing SAELens with a sentinel ID and parsing the ValueError that
comes back, which always contains the full list of valid IDs.

Usage:
    python list_saes.py gemma-scope-2b-pt-res          # all IDs
    python list_saes.py gemma-scope-2b-pt-res 12       # filter to layer 12
    python list_saes.py gemma-scope-2b-pt-res 12 16k   # filter to layer 12 + 16k width
"""

import re
import sys


def get_valid_ids_via_probe(release: str) -> list[str]:
    """
    Call SAE.from_pretrained with a deliberately invalid ID.
    SAELens raises a ValueError whose message contains the full valid-ID list.
    Parse and return it.
    """
    from sae_lens import SAE

    try:
        SAE.from_pretrained(release=release, sae_id="__probe__")
    except ValueError as e:
        msg = str(e)
        # The error message contains something like:
        # "Valid IDs are ['id1', 'id2', ...]"
        match = re.search(r"Valid IDs are \[(.+?)\]", msg, re.DOTALL)
        if match:
            raw = match.group(1)
            ids = re.findall(r"'([^']+)'", raw)
            if ids:
                return ids
        # Fallback: try to extract any quoted strings that look like paths
        ids = re.findall(r"'(layer_[^']+)'", msg)
        if ids:
            return ids
        print(f"Could not parse IDs from error message. Full message:\n{msg}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error probing release '{release}': {e}")
        sys.exit(1)

    # If no exception was raised the probe ID accidentally exists — very unlikely
    print(f"Probe succeeded unexpectedly. Release '{release}' may not exist.")
    sys.exit(1)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python list_saes.py <release> [layer_number] [width_filter]")
        print("\nExample releases to try:")
        print("  gemma-scope-2b-pt-res")
        print("  gemma-scope-9b-pt-res")
        print("  llama_scope_lxr_8x")
        sys.exit(0)

    release = args[0]
    layer_filter = args[1] if len(args) > 1 else None
    extra_filter = args[2] if len(args) > 2 else None

    print(f"Probing release '{release}' ...")
    ids = get_valid_ids_via_probe(release)

    if layer_filter:
        ids = [i for i in ids if f"layer_{layer_filter}/" in i]
    if extra_filter:
        ids = [i for i in ids if extra_filter in i]

    filter_desc = f" (layer={layer_filter}" + (f", filter={extra_filter})" if extra_filter else ")") if layer_filter else ""
    print(f"\n{len(ids)} IDs in '{release}'{filter_desc}:\n")
    for sae_id in sorted(ids):
        print(f"  {sae_id}")

    if layer_filter and not ids:
        print(f"  (no matches — try without the layer filter)")
        print(f"\nRe-running without filter:")
        ids_all = get_valid_ids_via_probe(release)
        for sae_id in sorted(ids_all):
            print(f"  {sae_id}")


if __name__ == "__main__":
    main()
