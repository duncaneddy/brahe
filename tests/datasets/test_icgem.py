"""CI-gated tests for brahe.datasets.icgem and GravityModelType.icgem(...)."""

import json
import shutil
import time
from pathlib import Path

import pytest

import brahe
import brahe.datasets as datasets


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point BRAHE_CACHE at a tempdir and pre-seed an icgem entry + gfc file."""
    monkeypatch.setenv("BRAHE_CACHE", str(tmp_path))

    icgem_dir = tmp_path / "icgem"
    icgem_dir.mkdir()
    models_dir = icgem_dir / "models" / "earth"
    models_dir.mkdir(parents=True)

    # Pre-place a known gfc so we never hit the network.
    src_gfc = REPO_ROOT / "data" / "gravity_models" / "JGM3.gfc"
    shutil.copy(src_gfc, models_dir / "JGM3-70-seed.gfc")

    # Pre-place a fresh index pointing at JGM3.
    index = {
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
    (icgem_dir / "index_earth.json").write_text(json.dumps(index))

    yield tmp_path


@pytest.mark.ci
def test_list_models_returns_seeded_entry(isolated_cache):
    entries = datasets.icgem.list_models("earth")
    assert any(e.name == "JGM3" for e in entries)


@pytest.mark.ci
def test_download_model_returns_cached_path(isolated_cache):
    path = datasets.icgem.download_model("earth", "JGM3")
    assert Path(path).exists()
    assert path.endswith("JGM3-70-seed.gfc")


@pytest.mark.ci
def test_gravity_model_type_icgem_classmethod(isolated_cache):
    t = brahe.GravityModelType.icgem("earth", "JGM3")
    model = brahe.GravityModel.from_model_type(t)
    assert model.n_max == 70


@pytest.mark.ci
def test_gravity_model_type_icgem_equality_distinguishes_body_and_name():
    """Two ICGEMModel values must compare equal only when body and name both match.
    """
    a = brahe.GravityModelType.icgem("earth", "JGM3")
    b = brahe.GravityModelType.icgem("earth", "JGM3")
    c = brahe.GravityModelType.icgem("earth", "GEM6")
    d = brahe.GravityModelType.icgem("moon", "GRGM1200B")

    # Same body + same name → equal.
    assert a == b
    assert not (a != b)

    # Same body, different names → not equal.
    assert a != c
    assert not (a == c)

    # Different body, different name → not equal.
    assert a != d
    assert not (a == d)

    # ICGEMModel must not collapse to equality with non-ICGEM variants either.
    assert a != brahe.GravityModelType.JGM3
