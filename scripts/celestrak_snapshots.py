"""Shared naming rules for the committed Celestrak GP snapshots.

Snapshots are committed under a readable name (`active.json`), but the client
looks them up in `~/.cache/brahe/celestrak` under a name derived from the
request URL. Both the refresh command and the cache-seeding step need to agree
on that mapping, so it is defined once here.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / ".github" / "brahe-data-manifest.txt"
SNAPSHOT_DIR = REPO_ROOT / "test_assets" / "celestrak"

# `query_gp` forces JSON internally whatever the caller asks for, so this is the
# URL the client actually caches against.
GP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=JSON"


def manifest_groups() -> list[str]:
    """Return the `celestrak:group:<name>` entries from the data manifest."""
    groups = []
    for line in MANIFEST_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("celestrak:group:"):
            groups.append(line.split(":", 2)[2])
    return groups


def cache_key_for_url(url: str) -> str:
    """Mirror `CelestrakClient::cache_key_for_url` (src/celestrak/client.rs).

    Every character that is not alphanumeric and not `.` becomes `_`. A change
    to the Rust implementation without a matching change here would leave the
    seeded files under names the client never looks up, which
    `tests/scripts/test_celestrak_snapshots.py` guards against.
    """
    return "".join(c if (c.isalnum() or c == ".") else "_" for c in url)


def cache_name(group: str) -> str:
    """Return the cache file name the client resolves for a GP group query."""
    return cache_key_for_url(GP_URL.format(group=group))


def snapshot_path(group: str) -> Path:
    """Return the committed snapshot path for a GP group."""
    return SNAPSHOT_DIR / f"{group}.json"
