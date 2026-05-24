import os
from pathlib import Path

import pytest

from benchmarks.gpu_comparison.config import (
    BRAHE_EOP_FILE,
    BRAHE_GRAVITY_FILE,
    BRAHE_SPACE_WEATHER_FILE,
    REPO_ROOT,
    collect_system_info,
    data_alignment_record,
    set_data_alignment_env,
    sha256_of,
)


def test_paths_point_inside_repo():
    assert BRAHE_EOP_FILE.is_relative_to(REPO_ROOT)
    assert BRAHE_SPACE_WEATHER_FILE.is_relative_to(REPO_ROOT)
    assert BRAHE_GRAVITY_FILE.is_relative_to(REPO_ROOT)


def test_data_files_exist():
    assert BRAHE_EOP_FILE.exists(), f"missing: {BRAHE_EOP_FILE}"
    assert BRAHE_SPACE_WEATHER_FILE.exists(), f"missing: {BRAHE_SPACE_WEATHER_FILE}"
    assert BRAHE_GRAVITY_FILE.exists(), f"missing: {BRAHE_GRAVITY_FILE}"


def test_collect_system_info_returns_required_fields():
    info = collect_system_info()
    assert info.cpu_logical_cores >= 1
    assert info.cpu_physical_cores >= 1
    assert info.ram_gb >= 1
    assert info.os
    assert info.python_version
    assert isinstance(info.gpus, list)


def test_set_data_alignment_env_sets_env_vars():
    for var in ("BRAHE_EOP_FILE", "BRAHE_SPACE_WEATHER_FILE", "BRAHE_GRAVITY_FILE"):
        os.environ.pop(var, None)
    set_data_alignment_env()
    assert os.environ["BRAHE_EOP_FILE"] == str(BRAHE_EOP_FILE)
    assert os.environ["BRAHE_SPACE_WEATHER_FILE"] == str(BRAHE_SPACE_WEATHER_FILE)
    assert os.environ["BRAHE_GRAVITY_FILE"] == str(BRAHE_GRAVITY_FILE)


def test_data_alignment_record_includes_hashes():
    rec = data_alignment_record()
    assert "eop_file" in rec
    assert "eop_sha256" in rec
    assert len(rec["eop_sha256"]) == 64


def test_sha256_of_tmpfile(tmp_path: Path):
    p = tmp_path / "x"
    p.write_text("hello")
    assert sha256_of(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
