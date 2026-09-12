#!/usr/bin/env python3
"""Copy the committed Starlink fixtures into the local brahe cache.

Installs ``test_assets/starlink/MANIFEST.txt`` and the two ephemeris files it
names under ``$BRAHE_CACHE/starlink/`` so the Starlink examples and the
documentation build run without contacting api.starlink.com. Used by
``just download-resources`` and the CI example and documentation jobs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "test_assets" / "starlink"


def cache_dir() -> Path:
    base = os.environ.get("BRAHE_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "brahe"
    return root / "starlink"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    target = args.cache_dir or cache_dir()
    target.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in ASSETS.iterdir() if p.suffix == ".txt")
    if not any(p.name == "MANIFEST.txt" for p in files):
        print(f"error: {ASSETS / 'MANIFEST.txt'} is missing", file=sys.stderr)
        return 1

    for path in files:
        shutil.copy2(path, target / path.name)
    print(f"Seeded {len(files)} Starlink files into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
