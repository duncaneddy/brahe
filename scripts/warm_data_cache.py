#!/usr/bin/env python3
"""Pre-populate ~/.cache/brahe with the kernels/models CI depends on.

Reads `.github/brahe-data-manifest.txt` (one artifact per line) and downloads
each entry via brahe's own caching downloaders, so a warm run is idempotent:
already-cached artifacts fast-path without hitting the network.

Two manifest line forms are supported:
    <kernel-name>              -> brahe.load_kernel(name), which downloads and
                                   caches known DE/PCK/satellite kernel names
                                   via `download_spice_kernel`'s name
                                   resolution.
    icgem:<body>:<model-name>  -> brahe.datasets.icgem.download_model(body, model-name)

This is the download-only counterpart to `scripts/warm_cartopy.py`: CI
workflows restore/save `~/.cache/brahe` via `actions/cache`, keyed on the
manifest's hash, and either run this script directly (the weekly keep-warm
workflow) or rely on the test suite's own fixtures to populate the same
cache (regular test/integration runs).
"""

from pathlib import Path

import brahe as bh
import brahe.datasets as datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / ".github" / "brahe-data-manifest.txt"


def _read_manifest() -> list[str]:
    lines = MANIFEST_PATH.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _warm_entry(entry: str) -> str:
    """Download one manifest entry, returning a description of what was cached."""
    if entry.startswith("icgem:"):
        _, body, model_name = entry.split(":", 2)
        path = datasets.icgem.download_model(body, model_name)
        return path

    bh.load_kernel(entry)
    assert entry in bh.loaded_kernels(), (
        f"{entry!r} not in loaded_kernels() after load_kernel()"
    )
    return "loaded"


def main() -> None:
    entries = _read_manifest()
    print(f"Warming brahe data cache from {MANIFEST_PATH.relative_to(REPO_ROOT)}...")

    for entry in entries:
        result = _warm_entry(entry)
        print(f"  {entry:<28} -> {result}")

    print(f"Brahe data cache warm: {len(entries)} artifact(s) verified.")


if __name__ == "__main__":
    main()
