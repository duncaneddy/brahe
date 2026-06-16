"""CLI tests for `brahe datasets icgem`."""

import json
import shutil
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brahe.cli.datasets import app


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    icgem_dir = tmp_path / "icgem"
    icgem_dir.mkdir()
    (icgem_dir / "index_earth.json").write_text(
        json.dumps(
            {
                "fetched_at": int(time.time()),
                "entries": [
                    {
                        "body": "Earth",
                        "name": "JGM3",
                        "year": 1996,
                        "degree": 70,
                        "download_path": "/getmodel/gfc/seed/JGM3.gfc",
                    }
                ],
            }
        )
    )
    models_dir = icgem_dir / "models" / "earth"
    models_dir.mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "data" / "gravity_models" / "JGM3.gfc",
        models_dir / "JGM3-70-seed.gfc",
    )
    return tmp_path


@pytest.mark.integration
def test_cli_list(isolated_cache):
    runner = CliRunner()
    result = runner.invoke(app, ["icgem", "list", "--body", "earth"])
    assert result.exit_code == 0, result.output
    assert "JGM3" in result.output


@pytest.mark.integration
def test_cli_download(isolated_cache):
    runner = CliRunner()
    result = runner.invoke(app, ["icgem", "download", "JGM3", "--body", "earth"])
    assert result.exit_code == 0, result.output
    assert "JGM3-70-seed.gfc" in result.output


@pytest.mark.integration
def test_cli_download_rejects_body_all(isolated_cache):
    runner = CliRunner()
    result = runner.invoke(app, ["icgem", "download", "JGM3", "--body", "all"])
    assert result.exit_code != 0
    # Error may appear in stderr or output depending on typer version.
    combined = result.output + (result.stderr if result.stderr else "")
    assert "not valid" in combined or "BadParameter" in combined or "all" in combined


@pytest.mark.integration
def test_cli_list_single_body_failure_exits_nonzero(tmp_path, monkeypatch):
    """A single-body `list` failure must exit non-zero, not silently print '0 model(s)'.

    Previously, the loop swallowed exceptions even in single-body mode, which
    hid offline/missing-cache errors behind a successful empty result.
    """
    # Point at an empty cache dir AND an unreachable URL so the listing has
    # neither a cache file to fall back to nor a network to fetch from.
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    # Force the in-process icgem module to fail by monkeypatching the lookup.
    import brahe.datasets as datasets

    def boom(_body):
        raise RuntimeError("simulated ICGEM lookup failure")

    monkeypatch.setattr(datasets.icgem, "list_models", boom)

    runner = CliRunner()
    result = runner.invoke(app, ["icgem", "list", "--body", "earth"])
    assert result.exit_code == 1, result.output
    combined = result.output + (result.stderr if result.stderr else "")
    assert "simulated ICGEM lookup failure" in combined


@pytest.mark.integration
def test_cli_refresh_body_all_dispatches_to_all_indexes(tmp_path, monkeypatch):
    """`refresh --body all` must dispatch to refresh_all_indexes(), not silently
    refresh only the celestial index (which is what ICGEMBody::Other("all") would do).

    Mirrors `list --body all` semantics so users don't get a stale Earth index when
    they expected both indexes refreshed.
    """
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    import brahe.datasets as datasets

    calls: dict[str, list] = {"refresh_index": [], "refresh_all_indexes": []}

    def fake_refresh_index(body):
        calls["refresh_index"].append(body)

    def fake_refresh_all_indexes():
        calls["refresh_all_indexes"].append(True)

    monkeypatch.setattr(datasets.icgem, "refresh_index", fake_refresh_index)
    monkeypatch.setattr(datasets.icgem, "refresh_all_indexes", fake_refresh_all_indexes)

    runner = CliRunner()
    result = runner.invoke(app, ["icgem", "refresh", "--body", "all"])

    assert result.exit_code == 0, result.output
    assert calls["refresh_all_indexes"] == [True], (
        f"expected refresh_all_indexes() to be called once, got {calls}"
    )
    assert "all" not in calls["refresh_index"], (
        f"refresh_index('all') would silently refresh only the celestial index; got {calls}"
    )


@pytest.mark.integration
def test_cli_list_body_all_tolerates_per_body_failure(tmp_path, monkeypatch):
    """In --body all mode, an individual body's failure is logged and skipped,
    not raised — the multi-body listing still succeeds for the bodies that work.
    """
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))
    import brahe.datasets as datasets

    real_list = datasets.icgem.list_models

    def selective_fail(body):
        if body == "mars":
            raise RuntimeError("mars cache cannot be read")
        return real_list(body)

    monkeypatch.setattr(datasets.icgem, "list_models", selective_fail)

    runner = CliRunner()
    result = runner.invoke(app, ["icgem", "list", "--body", "all"])
    # Multi-body should not abort on a single body's failure.
    assert result.exit_code == 0, result.output
