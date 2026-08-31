#!/usr/bin/env python3
"""Copy the committed Celestrak snapshots into the local brahe cache.

Committed under readable names, installed under the names the client resolves.
Used by the CI example and documentation jobs and by `just download-resources`
so the offline runs never contact Celestrak.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from celestrak_snapshots import cache_name, manifest_groups, snapshot_path


def cache_dir() -> Path:
    base = os.environ.get("BRAHE_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "brahe"
    return root / "celestrak"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    target = args.cache_dir or cache_dir()
    target.mkdir(parents=True, exist_ok=True)

    missing = [g for g in manifest_groups() if not snapshot_path(g).exists()]
    if missing:
        print(
            f"error: missing committed snapshots: {', '.join(missing)}", file=sys.stderr
        )
        return 1

    for group in manifest_groups():
        shutil.copy2(snapshot_path(group), target / cache_name(group))
    print(f"Seeded {len(manifest_groups())} Celestrak groups into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
