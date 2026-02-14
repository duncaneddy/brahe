#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "brahe>=1.1.2",
# ]
# ///
"""Fetch satellite ephemeris data from Space-Track and Celestrak.

Downloads GP elements, CDM messages, and active satellite data, saving
gzip-compressed JSON files to a shared NFS mount. Designed to run as a
cron job every 8 hours. Be sure to create the log entry and set the required
permissions the log file.

Cron entry (runs at minute 17 every 8 hours):
    17 */8 * * * SPACETRACK_USER='user@example.com' SPACETRACK_PASSWORD='password' /usr/bin/env uv run /PATH_TO/brahe/scripts/fetch_ephemeris.py >> /var/log/fetch_ephemeris.log 2>&1

Environment variables:
    SPACETRACK_USER      Space-Track.org username
    SPACETRACK_PASSWORD  Space-Track.org password
"""

import gzip
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import brahe as bh
from brahe.spacetrack import operators as op

BASE_DIR = Path("/nfs/datasets/satellite_ephemeris")

DIRS = {
    "spacetrack_gp": BASE_DIR / "spacetrack" / "gp",
    "spacetrack_cdm": BASE_DIR / "spacetrack" / "cdm",
    "celestrak_gp_active": BASE_DIR / "celestrak" / "gp_active",
}


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes >= 1 << 30:
        return f"{size_bytes / (1 << 30):.2f} GB"
    if size_bytes >= 1 << 20:
        return f"{size_bytes / (1 << 20):.2f} MB"
    if size_bytes >= 1 << 10:
        return f"{size_bytes / (1 << 10):.2f} KB"
    return f"{size_bytes} B"


def write_gzipped(path: Path, data: str) -> int:
    """Write string data as gzip-compressed file. Returns compressed size in bytes."""
    encoded = data.encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(encoded)
    return path.stat().st_size


def fetch_spacetrack_gp(client: bh.SpaceTrackClient, timestamp: str) -> None:
    """Download all non-decayed GP elements from Space-Track."""
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("DECAY_DATE", op.null_val())
        .order_by("NORAD_CAT_ID", bh.SortOrder.ASC)
    )
    raw_json = client.query_raw(query)

    out_path = DIRS["spacetrack_gp"] / f"gp_{timestamp}.json.gz"
    compressed_size = write_gzipped(out_path, raw_json)

    # Count records (top-level JSON array)
    count = raw_json.count('"NORAD_CAT_ID"')
    print(f"  Space-Track GP: {count} records, {format_size(compressed_size)} -> {out_path}")


def fetch_spacetrack_cdm(client: bh.SpaceTrackClient, timestamp: str) -> None:
    """Download latest CDM public messages from Space-Track."""
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.CDM_PUBLIC)
        .filter("CREATED", op.greater_than(op.now_offset(-7)))
        .order_by("TCA", bh.SortOrder.DESC)
    )
    raw_json = client.query_raw(query)

    out_path = DIRS["spacetrack_cdm"] / f"cdm_{timestamp}.json.gz"
    compressed_size = write_gzipped(out_path, raw_json)

    count = raw_json.count('"CDM_ID"')
    print(f"  Space-Track CDM: {count} records, {format_size(compressed_size)} -> {out_path}")


def fetch_celestrak_gp_active(timestamp: str) -> None:
    """Download active spacecraft GP elements from Celestrak."""
    ct_client = bh.celestrak.CelestrakClient(cache_max_age=0.0)
    query = (
        bh.celestrak.CelestrakQuery.gp
        .group("active")
        .format(bh.celestrak.CelestrakOutputFormat.JSON)
    )
    raw_json = ct_client.query_raw(query)

    out_path = DIRS["celestrak_gp_active"] / f"gp_active_{timestamp}.json.gz"
    compressed_size = write_gzipped(out_path, raw_json)

    count = raw_json.count('"NORAD_CAT_ID"')
    print(f"  Celestrak GP Active: {count} records, {format_size(compressed_size)} -> {out_path}")


def main() -> None:
    # Validate credentials
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not user or not password:
        print(
            "Error: SPACETRACK_USER and SPACETRACK_PASSWORD environment variables are required.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create output directories
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    print(f"Fetching ephemeris data at {timestamp}")

    errors = []

    # Fetch Space-Track data
    try:
        client = bh.SpaceTrackClient(user, password)
    except Exception as e:
        print(f"Error: Failed to create Space-Track client: {e}", file=sys.stderr)
        errors.append("SpaceTrack client creation")
        client = None

    if client is not None:
        try:
            fetch_spacetrack_gp(client, timestamp)
        except Exception as e:
            print(f"Error: Space-Track GP fetch failed: {e}", file=sys.stderr)
            errors.append("SpaceTrack GP")

        try:
            fetch_spacetrack_cdm(client, timestamp)
        except Exception as e:
            print(f"Error: Space-Track CDM fetch failed: {e}", file=sys.stderr)
            errors.append("SpaceTrack CDM")

    # Fetch Celestrak data
    try:
        fetch_celestrak_gp_active(timestamp)
    except Exception as e:
        print(f"Error: Celestrak GP Active fetch failed: {e}", file=sys.stderr)
        errors.append("Celestrak GP Active")

    # Summary
    if errors:
        print(f"Completed with {len(errors)} error(s): {', '.join(errors)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("All fetches completed successfully.")


if __name__ == "__main__":
    main()
