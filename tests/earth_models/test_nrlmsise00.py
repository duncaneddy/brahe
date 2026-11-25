"""
Tests for NRLMSISE-00 atmospheric density model.

Tests mirror the Rust implementation in src/earth_models/nrlmsise00.rs
"""

import inspect

import numpy as np
import pytest

import brahe as bh


class TestNrlmsise00:
    """Tests for NRLMSISE-00 atmospheric density model."""

    @pytest.mark.integration
    def test_nrlmsise00_ecef_basic(self, eop):
        """Test NRLMSISE-00 model with ECEF coordinates."""
        bh.initialize_sw()

        # Test at 400 km altitude over equator
        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        x_ecef = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])

        density = bh.density_nrlmsise00(epc, x_ecef)

        # Should return positive density
        assert density > 0.0
        # Typical density at 400 km is on the order of 1e-12 kg/m³
        assert 1e-14 < density < 1e-10

    @pytest.mark.integration
    def test_nrlmsise00_ecef_matches_geod(self, eop):
        """Test that ECEF and geodetic versions produce same results."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)

        # Define geodetic position
        geod = np.array([0.0, 45.0, 400e3])  # 45° lat, 400 km altitude

        # Convert to ECEF
        x_ecef = bh.position_geodetic_to_ecef(geod, bh.AngleFormat.DEGREES)

        # Compute density both ways
        density_ecef = bh.density_nrlmsise00(epc, x_ecef)
        density_geod = bh.density_nrlmsise00_geod(epc, geod)

        # Should match within numerical precision
        assert density_ecef == pytest.approx(density_geod, rel=1e-10, abs=0.0)

    @pytest.mark.integration
    def test_nrlmsise00_geod_basic(self, eop):
        """Test NRLMSISE-00 model basic functionality with geodetic coordinates."""
        bh.initialize_sw()

        # Test at 400 km altitude
        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        geod = np.array([-74.0, 40.7, 400e3])  # NYC area, 400 km altitude

        density = bh.density_nrlmsise00_geod(epc, geod)

        # Should return positive density
        assert density > 0.0
        # Typical density at 400 km is on the order of 1e-12 kg/m³
        assert 1e-14 < density < 1e-10

    @pytest.mark.integration
    def test_nrlmsise00_altitude_variation(self, eop):
        """Test NRLMSISE-00 density decreases with altitude."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        lon, lat = 0.0, 0.0

        # Test at different altitudes
        altitudes = [200e3, 400e3, 600e3, 800e3]
        densities = []

        for alt in altitudes:
            geod = np.array([lon, lat, alt])
            density = bh.density_nrlmsise00_geod(epc, geod)
            densities.append(density)

        # Density should decrease with altitude
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Density at {altitudes[i] / 1e3}km should be > at {altitudes[i + 1] / 1e3}km"
            )

    @pytest.mark.integration
    def test_nrlmsise00_different_epochs(self, eop):
        """Test NRLMSISE-00 produces different results for different epochs."""
        bh.initialize_sw()

        geod = np.array([0.0, 0.0, 400e3])

        # Test at different epochs
        epc1 = bh.Epoch.from_date(2020, 1, 1, bh.TimeSystem.UTC)
        epc2 = bh.Epoch.from_date(2020, 7, 1, bh.TimeSystem.UTC)

        density1 = bh.density_nrlmsise00_geod(epc1, geod)
        density2 = bh.density_nrlmsise00_geod(epc2, geod)

        # Densities should be different (seasonal variation)
        assert density1 != density2
        # But both should be positive and reasonable
        assert density1 > 0.0
        assert density2 > 0.0

    @pytest.mark.integration
    def test_nrlmsise00_different_locations(self, eop):
        """Test NRLMSISE-00 produces different results for different locations."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        alt = 400e3

        # Test at different locations
        geod1 = np.array([0.0, 0.0, alt])  # Equator
        geod2 = np.array([0.0, 60.0, alt])  # High latitude

        density1 = bh.density_nrlmsise00_geod(epc, geod1)
        density2 = bh.density_nrlmsise00_geod(epc, geod2)

        # Both should be positive and reasonable
        assert density1 > 0.0
        assert density2 > 0.0

    @pytest.mark.integration
    def test_nrlmsise00_invalid_input(self, eop):
        """Test NRLMSISE-00 error handling for invalid inputs."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)

        # Test with wrong array length (ECEF version)
        with pytest.raises(ValueError):
            bh.density_nrlmsise00(epc, np.array([0.0, 0.0]))  # Only 2 elements

        # Test with wrong array length (geodetic version)
        with pytest.raises(ValueError):
            bh.density_nrlmsise00_geod(epc, np.array([0.0, 0.0]))  # Only 2 elements

    @pytest.mark.integration
    def test_nrlmsise00_leo_altitudes(self, eop):
        """Test NRLMSISE-00 at typical LEO altitudes."""
        bh.initialize_sw()

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)

        # ISS altitude (~400 km)
        geod_iss = np.array([0.0, 0.0, 400e3])
        density_iss = bh.density_nrlmsise00_geod(epc, geod_iss)

        # Starlink altitude (~550 km)
        geod_starlink = np.array([0.0, 0.0, 550e3])
        density_starlink = bh.density_nrlmsise00_geod(epc, geod_starlink)

        # ISS should have higher density than Starlink
        assert density_iss > density_starlink

        # Both should be in reasonable LEO range
        assert 1e-14 < density_iss < 1e-10
        assert 1e-15 < density_starlink < 1e-11

    def test_nrlmsise00_lunar_distance_is_negligible(self, eop, sw_test_filepath):
        """Density at cislunar distances stays finite and effectively zero
        (mirrors the Rust test of the same name)."""
        sw = bh.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")
        bh.set_global_space_weather_provider(sw)

        epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
        x_ecef = np.array([3.8e8, 0.0, 0.0])  # ~lunar distance

        density = bh.density_nrlmsise00(epc, x_ecef)

        assert np.isfinite(density)
        assert density < 1e-15


@pytest.fixture
def sw_test(sw_test_filepath):
    bh.set_global_space_weather_provider(
        bh.FileSpaceWeatherProvider.from_file(sw_test_filepath, "Hold")
    )


def test_nrlmsise00config_default():
    s = bh.NRLMSISE00Config().to_switches()
    assert s[0] == 1 and s[9] == -1
    assert all(v == 1 for i, v in enumerate(s) if i not in (0, 9))


def test_nrlmsise00config_kwargs_and_setters():
    cfg = bh.NRLMSISE00Config(
        diurnal=bh.NRLMSISE00Switch.OFF, ap_mode=bh.NRLMSISE00ApMode.DAILY
    )
    assert cfg.to_switches()[7] == 0 and cfg.to_switches()[9] == 1
    cfg.terdiurnal = bh.NRLMSISE00Switch.CROSS_TERMS_ONLY
    assert cfg.terdiurnal == bh.NRLMSISE00Switch.CROSS_TERMS_ONLY
    assert cfg.to_switches()[14] == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, bh.NRLMSISE00ApMode.THREE_HOURLY),
        (0, bh.NRLMSISE00ApMode.OFF),
        (1, bh.NRLMSISE00ApMode.DAILY),
    ],
)
def test_nrlmsise00config_from_switches_ap_mode(value, expected):
    switches = [1] * 24
    switches[9] = value
    cfg = bh.NRLMSISE00Config.from_switches(switches)
    assert cfg.ap_mode == expected
    assert cfg.to_switches()[9] == value


def test_nrlmsise00config_from_switches_c_reference_array():
    a = [1] * 24
    a[0] = 0
    a[9] = -1
    cfg = bh.NRLMSISE00Config.from_switches(a)
    assert cfg == bh.NRLMSISE00Config() and cfg.to_switches()[0] == 1


def test_nrlmsise00config_from_switches_numpy():
    cfg = bh.NRLMSISE00Config.from_switches(np.ones(24, dtype=np.int64))
    assert cfg.ap_mode == bh.NRLMSISE00ApMode.DAILY


@pytest.mark.parametrize(
    "bad",
    [
        [1] * 23,
        [1] * 25,
        [2] + [1] * 23,
        [1] * 9 + [2] + [1] * 14,
        [1] * 5 + [-1] + [1] * 18,
        [1] * 23 + [3],
        [1] * 8 + [2**40] + [1] * 15,
    ],
)
def test_nrlmsise00config_from_switches_rejects(bad):
    with pytest.raises(ValueError):
        bh.NRLMSISE00Config.from_switches(bad)


def test_nrlmsise00config_from_switches_rejects_floats():
    with pytest.raises(TypeError):
        bh.NRLMSISE00Config.from_switches(np.ones(24))


FIELD_INDEX = [
    ("f107_mean", 1),
    ("time_independent", 2),
    ("symmetrical_annual", 3),
    ("symmetrical_semiannual", 4),
    ("asymmetrical_annual", 5),
    ("asymmetrical_semiannual", 6),
    ("diurnal", 7),
    ("semidiurnal", 8),
    ("ap_mode", 9),
    ("all_ut_longitude", 10),
    ("longitudinal", 11),
    ("ut_mixed_ut_longitude", 12),
    ("mixed_ap_ut_longitude", 13),
    ("terdiurnal", 14),
    ("diffusive_equilibrium_departures", 15),
    ("tinf_variations", 16),
    ("tlb_variations", 17),
    ("tn1_variations", 18),
    ("s_variations", 19),
    ("tn2_variations", 20),
    ("nlb_variations", 21),
    ("tn3_variations", 22),
    ("turbopause_scale_height_variations", 23),
]


@pytest.mark.parametrize("name,idx", FIELD_INDEX)
def test_nrlmsise00config_field_maps_to_switch_index(name, idx):
    if name == "ap_mode":
        value, expected = bh.NRLMSISE00ApMode.DAILY, 1
    else:
        value, expected = bh.NRLMSISE00Switch.OFF, 0

    cfg = bh.NRLMSISE00Config(**{name: value})
    switches = cfg.to_switches()
    assert switches[idx] == expected
    assert getattr(cfg, name) == value
    assert all(
        v == bh.NRLMSISE00Config().to_switches()[i]
        for i, v in enumerate(switches)
        if i != idx
    )

    via_setter = bh.NRLMSISE00Config()
    setattr(via_setter, name, value)
    assert via_setter == cfg


def test_nrlmsise00config_constructor_signature_lists_keywords():
    params = inspect.signature(bh.NRLMSISE00Config).parameters
    assert list(params) == [name for name, _ in FIELD_INDEX]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())


def test_nrlmsise00config_wrong_enum_type():
    cfg = bh.NRLMSISE00Config()
    with pytest.raises(TypeError):
        cfg.ap_mode = bh.NRLMSISE00Switch.ON
    with pytest.raises(TypeError):
        cfg.diurnal = bh.NRLMSISE00ApMode.DAILY


def test_nrlmsise00config_repr():
    assert repr(bh.NRLMSISE00Config()) == "NRLMSISE00Config()"
    cfg = bh.NRLMSISE00Config(
        diurnal=bh.NRLMSISE00Switch.OFF, ap_mode=bh.NRLMSISE00ApMode.DAILY
    )
    assert repr(cfg) == (
        "NRLMSISE00Config(diurnal=NRLMSISE00Switch.OFF, ap_mode=NRLMSISE00ApMode.DAILY)"
    )


def test_density_nrlmsise00_geod_daily_ap_matches_reference(eop, sw_test):
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    cfg = bh.NRLMSISE00Config(ap_mode=bh.NRLMSISE00ApMode.DAILY)
    rho = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 400e3]), cfg)
    assert rho == pytest.approx(1.067412479194915e-11, rel=1e-9, abs=0.0)


def test_density_nrlmsise00_geod_default_is_three_hourly(eop, sw_test):
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    rho = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 400e3]))
    assert rho == pytest.approx(1.157892733789963e-11, rel=1e-9, abs=0.0)


def test_density_nrlmsise00_config_matches_geod(eop, sw_test):
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    geod = np.array([-74.0, 40.7, 400e3])
    x_ecef = bh.position_geodetic_to_ecef(geod, bh.AngleFormat.DEGREES)
    cfg = bh.NRLMSISE00Config(ap_mode=bh.NRLMSISE00ApMode.DAILY)
    rho_daily = bh.density_nrlmsise00(epc, x_ecef, config=cfg)
    assert rho_daily == pytest.approx(
        bh.density_nrlmsise00_geod(epc, geod, cfg), rel=1e-10, abs=0.0
    )
    assert rho_daily != pytest.approx(
        bh.density_nrlmsise00(epc, x_ecef), rel=1e-6, abs=0.0
    )


def test_density_nrlmsise00_all_switches_off_is_finite(eop, sw_test):
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    kwargs = {
        name: (
            bh.NRLMSISE00ApMode.OFF if name == "ap_mode" else bh.NRLMSISE00Switch.OFF
        )
        for name, _ in FIELD_INDEX
    }
    cfg = bh.NRLMSISE00Config(**kwargs)
    rho = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 400e3]), cfg)
    assert np.isfinite(rho) and rho > 0.0


def test_density_nrlmsise00_geod_at_zn3_boundary_is_finite(eop, sw_test):
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    rho = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 32500.0]))
    below = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 32499.9]))
    assert np.isfinite(rho) and rho > 0.0
    assert abs(rho - below) / below < 1e-3


@pytest.mark.parametrize(
    "setting", [bh.NRLMSISE00Switch.OFF, bh.NRLMSISE00Switch.CROSS_TERMS_ONLY]
)
def test_density_nrlmsise00_geod_below_mesosphere_requires_diffusive_switch(
    eop, sw_test, setting
):
    """Mirrors test_density_nrlmsise00_geod_below_mesosphere_requires_diffusive_switch"""
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    cfg = bh.NRLMSISE00Config(diffusive_equilibrium_departures=setting)
    with pytest.raises(RuntimeError, match="switch 15"):
        bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 50e3]), cfg)
    x_ecef = bh.position_geodetic_to_ecef(
        np.array([-74.0, 40.7, 50e3]), bh.AngleFormat.DEGREES
    )
    with pytest.raises(RuntimeError, match="switch 15"):
        bh.density_nrlmsise00(epc, x_ecef, cfg)
    rho_boundary = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 72.5e3]), cfg)
    assert np.isfinite(rho_boundary) and rho_boundary > 0.0


def test_density_nrlmsise00_geod_below_mesosphere_with_diffusive_switch_on(
    eop, sw_test
):
    """Mirrors test_density_nrlmsise00_geod_below_mesosphere_with_diffusive_switch_on"""
    epc = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    rho = bh.density_nrlmsise00_geod(epc, np.array([-74.0, 40.7, 50e3]))
    assert np.isfinite(rho) and rho > 0.0
