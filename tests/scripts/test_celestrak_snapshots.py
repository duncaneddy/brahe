"""Tests for the committed Celestrak snapshot naming rules.

The committed name is readable; the cache name is whatever the client resolves.
If those drift apart the snapshots are seeded under names nothing looks up, and
examples fall back to the network — which is the failure this whole arrangement
exists to prevent.
"""

import importlib.util
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "celestrak_snapshots", _SCRIPTS / "celestrak_snapshots.py"
)
snaps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snaps)


def test_manifest_groups_are_read():
    groups = snaps.manifest_groups()
    assert "active" in groups
    assert all(":" not in g for g in groups)


def test_cache_key_replaces_every_non_alphanumeric_except_dot():
    assert (
        snaps.cache_key_for_url("https://a.org/b?C=d&E=F") == "https___a.org_b_C_d_E_F"
    )


def test_cache_name_matches_the_client_resolved_name():
    """The exact strings CelestrakClient looks up for each GP group query."""
    assert snaps.cache_name("active") == (
        "https___celestrak.org_NORAD_elements_gp.php_GROUP_active_FORMAT_JSON"
    )
    # Hyphens in a group name become underscores in the cache name, while the
    # committed file keeps the hyphens.
    assert snaps.cache_name("gps-ops") == (
        "https___celestrak.org_NORAD_elements_gp.php_GROUP_gps_ops_FORMAT_JSON"
    )


def test_every_manifest_group_has_a_committed_snapshot():
    missing = [
        g for g in snaps.manifest_groups() if not snaps.snapshot_path(g).exists()
    ]
    assert not missing, f"missing committed snapshots: {missing}"


def test_committed_snapshots_are_named_for_their_group():
    for group in snaps.manifest_groups():
        assert snaps.snapshot_path(group).name == f"{group}.json"
