"""
List all SAE IDs available for a given release in the installed SAELens version.

Reads pretrained_saes.yaml directly from the installed SAELens package —
works regardless of which SAELens API version is installed.

Usage:
    python list_saes.py                              # list all releases
    python list_saes.py gemma-scope-2b-pt-res        # all IDs in that release
    python list_saes.py gemma-scope-2b-pt-res 12     # filter to layer 12
    python list_saes.py gemma-scope-2b-pt-res 12 16k # layer 12 + width filter
"""

import os
import sys


def _find_yaml() -> str:
    """Return path to SAELens pretrained_saes.yaml."""
    try:
        import sae_lens
        pkg_dir = os.path.dirname(sae_lens.__file__)
        candidate = os.path.join(pkg_dir, "pretrained_saes.yaml")
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError(
        "Could not locate SAELens pretrained_saes.yaml. "
        "Is sae_lens installed? (pip install sae_lens)"
    )


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        # PyYAML not available — use a minimal line-by-line parser
        return _parse_yaml_minimal(path)
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_yaml_minimal(path: str) -> dict:
    """
    Minimal YAML parser that extracts release→IDs without PyYAML.
    Only handles the flat structure of pretrained_saes.yaml.
    """
    releases: dict = {}
    current_release = None
    in_saes = False

    with open(path) as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(stripped) - len(stripped.lstrip())
            content = stripped.lstrip()

            if indent == 0 and content.endswith(":"):
                current_release = content[:-1]
                releases[current_release] = []
                in_saes = False
            elif indent == 2 and content == "saes:":
                in_saes = True
            elif indent == 2 and content != "saes:":
                in_saes = False
            elif in_saes and content.startswith("- id:"):
                sae_id = content[len("- id:"):].strip()
                if current_release is not None:
                    releases[current_release].append(sae_id)

    return releases


def get_ids_for_release(release: str) -> list[str]:
    yaml_path = _find_yaml()
    data = _load_yaml(yaml_path)

    if release not in data:
        print(f"Release '{release}' not found in pretrained_saes.yaml.")
        print(f"\nAvailable releases ({len(data)}):")
        for r in sorted(data.keys()):
            print(f"  {r}")
        sys.exit(1)

    entry = data[release]
    # YAML structure: either a list already (from minimal parser)
    # or a dict with a 'saes' key containing list of dicts
    if isinstance(entry, list):
        return [str(i) for i in entry]
    elif isinstance(entry, dict):
        saes = entry.get("saes", [])
        return [s["id"] for s in saes if isinstance(s, dict) and "id" in s]
    return []


def main() -> None:
    args = sys.argv[1:]

    yaml_path = _find_yaml()
    print(f"SAELens catalog: {yaml_path}\n")

    if not args:
        data = _load_yaml(yaml_path)
        print(f"Available releases ({len(data)}):")
        for r in sorted(data.keys()):
            print(f"  {r}")
        return

    release = args[0]
    layer_filter = args[1] if len(args) > 1 else None
    width_filter = args[2] if len(args) > 2 else None

    ids = get_ids_for_release(release)

    if layer_filter:
        ids = [i for i in ids if f"layer_{layer_filter}/" in i]
    if width_filter:
        ids = [i for i in ids if width_filter in i]

    filter_parts = []
    if layer_filter:
        filter_parts.append(f"layer={layer_filter}")
    if width_filter:
        filter_parts.append(f"width={width_filter}")
    filter_desc = f" ({', '.join(filter_parts)})" if filter_parts else ""

    print(f"{len(ids)} IDs in '{release}'{filter_desc}:\n")
    for sae_id in sorted(ids):
        print(f"  {sae_id}")


if __name__ == "__main__":
    main()
