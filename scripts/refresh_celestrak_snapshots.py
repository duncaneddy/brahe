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

from celestrak_snapshots import (
    SNAPSHOT_DIR,
    cache_name,
    manifest_groups,
    snapshot_path,
)

# Minimum plausible size for a GP group response. Celestrak answers a throttled
# or errored request with a short HTML body, which is a valid file but not
# usable data; committing one would fail later as a confusing parse error.
MIN_BYTES = 500


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

        # The client writes each response under its cache-key name; map those
        # back to the readable committed name so the repository stays legible.
        by_cache_name = {cache_name(g): g for g in groups}
        fetched = sorted(Path(scratch, "celestrak").glob("*"))
        unexpected = [p.name for p in fetched if p.name not in by_cache_name]
        if unexpected:
            print(
                f"\nerror: unexpected cache entries: {', '.join(unexpected)}",
                file=sys.stderr,
            )
            return 1
        undersized = [p.name for p in fetched if p.stat().st_size < MIN_BYTES]
        if undersized:
            print(
                f"\nerror: implausibly small responses: {', '.join(undersized)}",
                file=sys.stderr,
            )
            return 1

        # Install each fetched snapshot over its own destination only. A
        # partial refresh (`--group`) must not disturb the snapshots it did not
        # fetch, and every replacement goes through `os.replace` so a failure
        # mid-run leaves each destination either wholly old or wholly new.
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for src in fetched:
            dest = snapshot_path(by_cache_name[src.name])
            staged = dest.with_name(f".{dest.name}.incoming")
            shutil.copy2(src, staged)
            os.replace(staged, dest)

        # A full refresh also drops snapshots whose group has left the
        # manifest. A partial refresh has no view of the full set, so it never
        # removes anything.
        if not args.group:
            keep = {snapshot_path(by_cache_name[src.name]).name for src in fetched}
            for stale in SNAPSHOT_DIR.glob("*"):
                if stale.name not in keep:
                    print(
                        f"  removing snapshot no longer in the manifest: {stale.name}"
                    )
                    stale.unlink()

    total = sum(p.stat().st_size for p in SNAPSHOT_DIR.glob("*"))
    print(
        f"\nWrote {len(list(SNAPSHOT_DIR.glob('*')))} snapshots to {SNAPSHOT_DIR} ({total / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
