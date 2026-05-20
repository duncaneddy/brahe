"""Unit tests for benchmarks.comparative.implementations.gmat.base.

These tests target pure helpers and precondition validation. The actual
gmatpy import is exercised in module integration tests; here we mock
sys.path / environment to validate the precondition logic.
"""

import os
import sys

import numpy as np
import pytest

from benchmarks.comparative.implementations.gmat import base


@pytest.fixture(autouse=True)
def _reset_init_flag():
    """Ensure each test sees a fresh _GMAT_INITIALIZED state."""
    base._GMAT_INITIALIZED = False
    yield
    base._GMAT_INITIALIZED = False


def test_ensure_gmat_raises_when_env_unset(monkeypatch):
    monkeypatch.delenv("GMAT_ROOT_PATH", raising=False)
    with pytest.raises(ImportError, match="GMAT_ROOT_PATH not set"):
        base._ensure_gmat()


def test_ensure_gmat_raises_when_so_missing(monkeypatch, tmp_path):
    """If the path is set but the matching _pyXY/_gmat_py.so is absent."""
    fake_root = tmp_path / "fake_gmat"
    (fake_root / "bin" / "gmatpy" / "_py999").mkdir(parents=True)
    # NOTE: no _gmat_py.so written.
    monkeypatch.setenv("GMAT_ROOT_PATH", str(fake_root))
    with pytest.raises(ImportError, match="gmatpy missing binary for Python"):
        base._ensure_gmat()


def test_ensure_gmat_raises_when_startup_missing(monkeypatch, tmp_path):
    """If gmatpy binary is present but api_startup_file.txt is missing."""
    fake_root = tmp_path / "fake_gmat"
    py_tag = f"_py{sys.version_info.major}{sys.version_info.minor}"
    so_dir = fake_root / "bin" / "gmatpy" / py_tag
    so_dir.mkdir(parents=True)
    (so_dir / "_gmat_py.so").touch()
    # NOTE: api_startup_file.txt not created.
    monkeypatch.setenv("GMAT_ROOT_PATH", str(fake_root))
    with pytest.raises(ImportError, match="api_startup_file.txt missing"):
        base._ensure_gmat()


def test_km_to_m_state_scales_all_six_components():
    """km_to_m_state multiplies [r_x, r_y, r_z, v_x, v_y, v_z] by 1000."""
    in_state = [7.0, 0.0, 0.0, 0.0, 7.5, 0.0]  # km, km/s
    out = base.km_to_m_state(in_state)
    assert out == pytest.approx([7000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])


def test_m_to_km_state_is_inverse():
    in_state = [7000.0, 0.0, 0.0, 0.0, 7500.0, 0.0]
    out = base.m_to_km_state(in_state)
    assert out == pytest.approx([7.0, 0.0, 0.0, 0.0, 7.5, 0.0])


def test_mu_si_to_gmat_scales_by_1e9():
    """SI mu (m^3/s^2) -> GMAT mu (km^3/s^2) factor of 1e-9."""
    mu_si = 3.986004418e14
    expected_gmat = 3.986004418e5
    assert base.mu_si_to_gmat(mu_si) == pytest.approx(expected_gmat, rel=1e-15)


def test_time_iterations_returns_times_and_first_result():
    """time_iterations returns (list-of-floats, first-call-output)."""
    call_count = {"n": 0}

    def func():
        call_count["n"] += 1
        return ["result", call_count["n"]]

    times, first = base.time_iterations(func, iterations=3)
    assert len(times) == 3
    assert all(isinstance(t, float) and t >= 0 for t in times)
    assert first == ["result", 1]
    assert call_count["n"] == 3


def test_quat_brahe_to_gmat_moves_scalar_to_end():
    """[w, x, y, z] -> [x, y, z, w]."""
    q_brahe = [0.7071, 0.7071, 0.0, 0.0]  # 90° about x
    q_gmat = base.quat_brahe_to_gmat(q_brahe)
    assert q_gmat == pytest.approx([0.7071, 0.0, 0.0, 0.7071])


def test_quat_gmat_to_brahe_moves_scalar_to_front():
    """[q1, q2, q3, q4] -> [q4, q1, q2, q3]."""
    q_gmat = [0.7071, 0.0, 0.0, 0.7071]
    q_brahe = base.quat_gmat_to_brahe(q_gmat)
    assert q_brahe == pytest.approx([0.7071, 0.7071, 0.0, 0.0])


def test_quat_roundtrip_preserves_value():
    q_brahe = [0.1, 0.2, 0.3, 0.9273618]
    rt = base.quat_gmat_to_brahe(base.quat_brahe_to_gmat(q_brahe))
    assert rt == pytest.approx(q_brahe)
