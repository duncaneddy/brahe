#!/usr/bin/env python3
"""Refresh the committed Celestrak GP snapshots in `test_assets/celestrak`.

The examples, doc plots, and the offline example runs in CI read these groups
from `~/.cache/brahe/celestrak`, seeded from `test_assets/celestrak`. Fetching
through `CelestrakClient` rather than `curl` means the on-disk file names are
produced by the client's own cache-key derivation, so a change to that
derivation cannot silently desynchronize the committed snapshots from the names
the client looks up.

Groups are read from `.github/brahe-data-manifest.txt` so the manifest stays the
single source of truth for which groups CI depends on.

Run when the snapshots should track current orbital data:

    just refresh-celestrak-snapshots
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / ".github" / "brahe-data-manifest.txt"
SNAPSHOT_DIR = REPO_ROOT / "test_assets" / "celestrak"

# Minimum plausible size for a GP group response. Celestrak answers a throttled
# or errored request with a short HTML body, which is a valid file but not
# usable data; committing one would fail later as a confusing parse error.
MIN_BYTES = 500


def manifest_groups() -> list[str]:
    """Return the `celestrak:group:<name>` entries from the data manifest."""
    groups = []
    for line in MANIFEST_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("celestrak:group:"):
            groups.append(line.split(":", 2)[2])
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        help="Refresh only this group (repeatable). Defaults to every manifest group.",
    )
    args = parser.parse_args()

    groups = args.group or manifest_groups()
    if not groups:
        print("error: no celestrak groups found in the manifest", file=sys.stderr)
        return 1

    # Fetch into a scratch cache so a stale or partial entry in the developer's
    # own cache cannot be mistaken for a freshly downloaded response.
    with tempfile.TemporaryDirectory() as scratch:
        os.environ["BRAHE_CACHE"] = scratch
        os.environ["BRAHE_NETWORK_MODE"] = "online"

        import brahe as bh

        client = bh.celestrak.CelestrakClient(cache_max_age=0.0)

        failed = []
        for group in groups:
            try:
                records = client.get_gp(group=group)
            except Exception as exc:  # noqa: BLE001 - reported per group below
                print(f"  {group:<22} FAILED: {exc}", file=sys.stderr)
                failed.append(group)
                continue
            print(f"  {group:<22} {len(records):>6} records")

        if failed:
            print(
                f"\nerror: {len(failed)} group(s) failed: {', '.join(failed)}",
                file=sys.stderr,
            )
            return 1

        fetched = sorted(Path(scratch, "celestrak").glob("*"))
        undersized = [p.name for p in fetched if p.stat().st_size < MIN_BYTES]
        if undersized:
            print(
                f"\nerror: implausibly small responses: {', '.join(undersized)}",
                file=sys.stderr,
            )
            return 1

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for stale in SNAPSHOT_DIR.glob("*"):
            stale.unlink()
        for src in fetched:
            shutil.copy2(src, SNAPSHOT_DIR / src.name)

    total = sum(p.stat().st_size for p in SNAPSHOT_DIR.glob("*"))
    print(
        f"\nWrote {len(list(SNAPSHOT_DIR.glob('*')))} snapshots to {SNAPSHOT_DIR} ({total / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
