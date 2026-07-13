"""Tests for the native SPICE kernel registry and generic SPK/PCK queries."""

import numpy as np
import pytest

import brahe as bh


@pytest.fixture(autouse=True)
def ensure_kernel():
    """Load the default kernel once per test module (cached on disk)."""
    try:
        bh.initialize_ephemeris()
    except Exception as e:
        pytest.skip(f"Could not initialize ephemeris: {e}")


def test_loaded_kernels():
    assert "de440s" in bh.loaded_kernels()


def test_load_kernel_idempotent():
    bh.load_kernel("de440s")
    bh.load_kernel("de440s")
    assert bh.loaded_kernels().count("de440s") == 1


def test_kernel_is_loaded():
    """Mirrors registry::tests::test_kernel_is_loaded."""
    bh.load_kernel("de440s")
    assert bh.kernel_is_loaded("de440s")
    assert bh.kernel_is_loaded("de440")
    assert not bh.kernel_is_loaded("nonexistent_kernel")


def test_spk_position():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r = bh.spk_position(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    assert r.shape == (3,)
    assert 3.5e8 < np.linalg.norm(r) < 4.1e8


def test_spk_velocity():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    v = bh.spk_velocity(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    assert v.shape == (3,)
    assert 9.0e2 < np.linalg.norm(v) < 1.2e3


def test_spk_state_consistent():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    x = bh.spk_state(bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
    r = bh.spk_position(bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
    v = bh.spk_velocity(bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
    assert x.shape == (6,)
    np.testing.assert_allclose(x[:3], r, atol=1e-9)
    np.testing.assert_allclose(x[3:], v, atol=1e-12)


def test_spk_no_path_error():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    with pytest.raises(Exception, match="12345"):
        bh.spk_position(bh.NAIFId.MOON, 12345, epc)


def test_pck_query_without_pck_errors():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    with pytest.raises(Exception, match="31006"):
        bh.pck_euler_angles(31006, epc)


def test_naif_id_enum():
    assert bh.NAIFId.SUN == 10
    assert bh.NAIFId.EARTH == 399
    assert bh.NAIFId.TITAN == 606
    assert bh.FrameId.MOON_PA_DE440 == 31008
    assert not hasattr(bh, "NAIF_SUN")


def test_spk_position_accepts_enum_and_int(eop):
    bh.load_kernel("test_assets/de440s.bsp")
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r_enum = bh.spk_position(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    r_int = bh.spk_position(301, 399, epc)
    np.testing.assert_array_equal(r_enum, r_int)


def test_spk_position_from_kernel_shape():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r = bh.spk_position_from_kernel("de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    assert r.shape == (3,)
    assert 3.5e8 < np.linalg.norm(r) < 4.1e8


def test_spk_velocity_from_kernel_shape():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    v = bh.spk_velocity_from_kernel("de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    assert v.shape == (3,)
    assert 9.0e2 < np.linalg.norm(v) < 1.2e3


def test_spk_state_from_kernel_shape():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    x = bh.spk_state_from_kernel("de440s", bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
    assert x.shape == (6,)


def test_spk_position_from_kernel_agrees_with_pooled():
    """With only de440s loaded, the kernel-scoped query must agree exactly
    with the pooled registry query."""
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r_pooled = bh.spk_position(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    r_from_kernel = bh.spk_position_from_kernel(
        "de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc
    )
    np.testing.assert_allclose(r_from_kernel, r_pooled, atol=0.0)


def test_spk_velocity_from_kernel_agrees_with_pooled():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    v_pooled = bh.spk_velocity(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    v_from_kernel = bh.spk_velocity_from_kernel(
        "de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc
    )
    np.testing.assert_allclose(v_from_kernel, v_pooled, atol=0.0)


def test_spk_state_from_kernel_agrees_with_pooled():
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    x_pooled = bh.spk_state(bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
    x_from_kernel = bh.spk_state_from_kernel(
        "de440s", bh.NAIFId.SUN, bh.NAIFId.EARTH, epc
    )
    np.testing.assert_allclose(x_from_kernel, x_pooled, atol=0.0)


@pytest.mark.integration
def test_load_common_kernels():
    bh.clear_kernels()
    bh.load_common_kernels()
    loaded = bh.loaded_kernels()
    assert "de440s" in loaded
    assert "moon_pa_de440" in loaded

    # Restore baseline registry state for other tests.
    bh.clear_kernels()
    bh.initialize_ephemeris()


def test_spk_position_from_kernel_auto_loads():
    """The _from_kernel query auto-loads the named kernel if not already
    present in the registry."""
    bh.clear_kernels()
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r = bh.spk_position_from_kernel("de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    assert r.shape == (3,)
    assert "de440s" in bh.loaded_kernels()


@pytest.mark.integration
def test_pck_typed_returns(eop):
    bh.load_kernel("moon_pa_de440")
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    e = bh.pck_euler_angle(31008, epc)
    assert isinstance(e, bh.EulerAngle)
    r = bh.pck_rotation_matrix(31008, epc)
    assert isinstance(r, bh.RotationMatrix)
    q = bh.pck_quaternion(31008, epc)
    assert isinstance(q, bh.Quaternion)
    np.testing.assert_allclose(
        e.to_rotation_matrix().to_matrix(), r.to_matrix(), atol=1e-12
    )

    rates = bh.pck_euler_rates(31008, epc)
    e2, rates2 = bh.pck_euler_angle_and_rates(31008, epc)
    assert isinstance(e2, bh.EulerAngle)
    assert e2 == e
    np.testing.assert_allclose(rates2, rates, atol=0.0)

    # Restore baseline registry state for other tests.
    bh.clear_kernels()
    bh.initialize_ephemeris()
