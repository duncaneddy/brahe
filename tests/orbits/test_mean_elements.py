"""
Tests for mean-osculating Keplerian element conversions.

Tests the Brouwer-Lyddane first-order J2 perturbation mapping between
mean and osculating orbital elements.
"""

import pytest
import numpy as np
import brahe

BROUWER_LYDDANE = brahe.MeanElementMethod.brouwer_lyddane()


class TestMeanOsculatingConversions:
    """Tests for mean-osculating Keplerian element conversions."""

    def test_round_trip_mean_to_osc_to_mean_radians(self):
        """Test round-trip: mean -> osc -> mean (radians)."""
        # Define mean elements for a typical LEO satellite
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,  # a = ~6878 km
                0.01,  # e = 0.01 (slightly eccentric)
                np.radians(45.0),  # i = 45 degrees
                np.radians(30.0),  # Ω = 30 degrees
                np.radians(60.0),  # ω = 60 degrees
                np.radians(90.0),  # M = 90 degrees
            ]
        )

        # Convert mean -> osc -> mean
        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        mean_recovered = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # Check that recovered elements are close to original
        # Note: First-order approximation means small errors of order J2² are expected
        # Tolerances match Rust tests accounting for this limitation
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)  # 100 m for SMA
        assert mean[1] == pytest.approx(
            mean_recovered[1], abs=1e-4
        )  # Eccentricity (J2² errors)
        assert mean[2] == pytest.approx(
            mean_recovered[2], abs=0.01
        )  # Inclination (~0.6 deg)
        assert mean[3] == pytest.approx(mean_recovered[3], abs=0.01)  # RAAN (~0.6 deg)
        assert mean[4] == pytest.approx(mean_recovered[4], abs=0.01)  # AoP (~0.6 deg)
        assert mean[5] == pytest.approx(
            mean_recovered[5], abs=0.01
        )  # Mean anomaly (~0.6 deg)

    def test_round_trip_mean_to_osc_to_mean_degrees(self):
        """Test round-trip: mean -> osc -> mean (degrees)."""
        # Define mean elements for a typical LEO satellite (angles in degrees)
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,  # a = ~6878 km
                0.01,  # e = 0.01 (slightly eccentric)
                45.0,  # i = 45 degrees
                30.0,  # Ω = 30 degrees
                60.0,  # ω = 60 degrees
                90.0,  # M = 90 degrees
            ]
        )

        # Convert mean -> osc -> mean
        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )
        mean_recovered = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )

        # Check that recovered elements are close to original
        # Tolerances in degrees (~0.01 radians = ~0.6 degrees)
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)  # 100 m for SMA
        assert mean[1] == pytest.approx(mean_recovered[1], abs=1e-4)  # Eccentricity
        assert mean[2] == pytest.approx(mean_recovered[2], abs=0.6)  # Inclination
        assert mean[3] == pytest.approx(mean_recovered[3], abs=0.6)  # RAAN
        assert mean[4] == pytest.approx(mean_recovered[4], abs=0.6)  # AoP
        assert mean[5] == pytest.approx(mean_recovered[5], abs=0.6)  # Mean anomaly

    def test_round_trip_osc_to_mean_to_osc(self):
        """Test round-trip: osc -> mean -> osc ≈ original."""
        # Define osculating elements for a typical LEO satellite
        osc = np.array(
            [
                brahe.R_EARTH + 600e3,  # a = ~6978 km
                0.02,  # e = 0.02
                np.radians(60.0),  # i = 60 degrees
                np.radians(45.0),  # Ω = 45 degrees
                np.radians(120.0),  # ω = 120 degrees
                np.radians(180.0),  # M = 180 degrees
            ]
        )

        # Convert osc -> mean -> osc
        mean = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        osc_recovered = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # Check that recovered elements are close to original
        # Tolerances match Rust tests accounting for J2² first-order approximation errors
        assert osc[0] == pytest.approx(osc_recovered[0], abs=100.0)  # 100 m for SMA
        assert osc[1] == pytest.approx(
            osc_recovered[1], abs=1e-4
        )  # Eccentricity (J2² errors)
        assert osc[2] == pytest.approx(
            osc_recovered[2], abs=0.01
        )  # Inclination (~0.6 deg)
        assert osc[3] == pytest.approx(osc_recovered[3], abs=0.01)  # RAAN (~0.6 deg)
        assert osc[4] == pytest.approx(osc_recovered[4], abs=0.01)  # AoP (~0.6 deg)
        assert osc[5] == pytest.approx(
            osc_recovered[5], abs=0.01
        )  # Mean anomaly (~0.6 deg)

    def test_near_circular_orbit(self):
        """Test near-circular orbit (small eccentricity)."""
        mean = np.array(
            [
                brahe.R_EARTH + 400e3,
                0.0001,  # Very small eccentricity
                np.radians(28.5),
                0.0,
                0.0,
                0.0,
            ]
        )

        # Should not raise
        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        mean_recovered = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # Semi-major axis should be close
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)

    def test_sun_synchronous_orbit(self):
        """Test sun-synchronous orbit (high inclination)."""
        mean = np.array(
            [
                brahe.R_EARTH + 700e3,
                0.001,
                np.radians(98.0),  # Sun-synchronous inclination
                np.radians(45.0),
                np.radians(90.0),
                np.radians(270.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        mean_recovered = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # Note: For small eccentricity (0.001), the relative J2² error can be larger
        # Tolerances match Rust tests
        assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0)
        assert mean[1] == pytest.approx(
            mean_recovered[1], abs=1e-3
        )  # Larger tolerance for very small e
        assert mean[2] == pytest.approx(mean_recovered[2], abs=1e-4)  # Inclination

    def test_osc_differs_from_mean(self):
        """Test that osculating elements differ from mean elements."""
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                np.radians(90.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # Osculating and mean should differ (J2 perturbation effect)
        # The semi-major axis should be different
        assert abs(osc[0] - mean[0]) > 1.0  # Should differ by more than 1 meter

    def test_various_mean_anomalies(self):
        """Test various mean anomaly values."""
        base_mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                0.0,  # Will be varied
            ]
        )

        for m_deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            mean = base_mean.copy()
            mean[5] = np.radians(m_deg)

            osc = brahe.state_koe_mean_to_osc(
                mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
            )
            mean_recovered = brahe.state_koe_osc_to_mean(
                osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
            )

            # Check semi-major axis recovery
            assert mean[0] == pytest.approx(mean_recovered[0], abs=100.0), (
                f"Failed at mean anomaly {m_deg} degrees"
            )

    def test_geo_orbit(self):
        """Test GEO orbit."""
        # Geostationary orbit parameters
        mean = np.array(
            [
                42164e3,  # GEO radius
                0.0001,  # Near-circular
                np.radians(0.1),  # Near-equatorial
                np.radians(45.0),
                0.0,
                0.0,
            ]
        )

        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        mean_recovered = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # At GEO altitude, J2 effects are smaller
        assert mean[0] == pytest.approx(mean_recovered[0], abs=10.0)

    def test_numpy_array_input(self):
        """Test that numpy arrays are properly accepted."""
        # Test with numpy array
        mean_np = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                np.radians(90.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(
            mean_np, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        assert isinstance(osc, np.ndarray)
        assert len(osc) == 6

    def test_list_input(self):
        """Test that Python lists are properly accepted."""
        # Test with Python list
        mean_list = [
            brahe.R_EARTH + 500e3,
            0.01,
            np.radians(45.0),
            np.radians(30.0),
            np.radians(60.0),
            np.radians(90.0),
        ]

        osc = brahe.state_koe_mean_to_osc(
            mean_list, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        assert isinstance(osc, np.ndarray)
        assert len(osc) == 6

    def test_return_type(self):
        """Test that return type is numpy array."""
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                np.radians(45.0),
                0.0,
                0.0,
                0.0,
            ]
        )

        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        mean_back = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        assert isinstance(osc, np.ndarray)
        assert isinstance(mean_back, np.ndarray)
        assert osc.dtype == np.float64
        assert mean_back.dtype == np.float64

    def test_invalid_array_length(self):
        """Test that invalid array lengths raise ValueError."""
        # Test with wrong length
        invalid = np.array([1.0, 2.0, 3.0])  # Only 3 elements

        with pytest.raises(ValueError):
            brahe.state_koe_mean_to_osc(
                invalid, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
            )

        with pytest.raises(ValueError):
            brahe.state_koe_osc_to_mean(
                invalid, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
            )

    def test_moderate_eccentricity(self):
        """Test with moderate eccentricity."""
        mean = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.1,  # Moderate eccentricity
                np.radians(45.0),
                np.radians(30.0),
                np.radians(60.0),
                np.radians(90.0),
            ]
        )

        osc = brahe.state_koe_mean_to_osc(
            mean, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )
        mean_recovered = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.RADIANS
        )

        # Should still recover reasonably well
        assert mean[0] == pytest.approx(mean_recovered[0], abs=200.0)
        assert mean[1] == pytest.approx(mean_recovered[1], abs=1e-5)

    def test_degrees_consistency(self):
        """Test that degrees input produces degrees output."""
        # Input in degrees
        mean_deg = np.array(
            [
                brahe.R_EARTH + 500e3,
                0.01,
                45.0,  # degrees
                30.0,  # degrees
                60.0,  # degrees
                90.0,  # degrees
            ]
        )

        # Convert to osculating with degrees format
        osc_deg = brahe.state_koe_mean_to_osc(
            mean_deg, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )

        # Verify output angles are in reasonable degree range (not radian range)
        assert osc_deg[2] >= 7.0 and osc_deg[2] < 180.0
        assert osc_deg[3] >= 7.0 and osc_deg[3] < 360.0  # RAAN should be in degrees
        assert osc_deg[4] >= 7.0 and osc_deg[4] < 360.0  # AoP should be in degrees
        assert osc_deg[5] >= 7.0 and osc_deg[5] < 360.0  # M should be in degrees

    def test_single_state_numerical_raises(self):
        """Numerical method is batch-only; single-state calls must raise."""
        mean = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        cfg = brahe.NumericalConfig(
            5400.0, brahe.WindowAlignment.CENTERED, brahe.EdgeHandling.TRUNCATE
        )
        with pytest.raises(Exception):
            brahe.state_koe_mean_to_osc(
                mean, brahe.MeanElementMethod.numerical(cfg), brahe.AngleFormat.DEGREES
            )

    def test_config_enum_equality_through_getter(self):
        """Enum values obtained via getters must compare equal to classattrs."""
        cfg = brahe.NumericalConfig(
            5400.0, brahe.WindowAlignment.CENTERED, brahe.EdgeHandling.TRUNCATE
        )
        assert cfg.alignment == brahe.WindowAlignment.CENTERED
        assert cfg.edge == brahe.EdgeHandling.TRUNCATE
        assert brahe.WindowAlignment.CENTERED != brahe.WindowAlignment.LEADING
        assert brahe.EdgeHandling.TRUNCATE != brahe.EdgeHandling.PRESERVE_WINDOW


class TestBatchMeanOsculatingConversions:
    """Tests for batch mean-osculating Keplerian element conversions."""

    def test_batch_analytical_osc_to_mean_shape(self):
        """Batch osc->mean with brouwer_lyddane preserves length and matches single-state."""
        e0 = brahe.Epoch.from_gps_seconds(0.0)
        epochs = [e0, e0 + 60.0, e0 + 120.0]
        s = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        states = np.vstack([s, s, s])

        out_epochs, out_states = brahe.batch_state_koe_osc_to_mean(
            epochs, states, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )

        assert len(out_epochs) == 3
        assert out_states.shape == (3, 6)

        single = brahe.state_koe_osc_to_mean(
            s, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )
        assert out_states[0] == pytest.approx(single)

    def test_batch_analytical_mean_to_osc_shape(self):
        """Batch mean->osc with brouwer_lyddane preserves length and matches single-state."""
        e0 = brahe.Epoch.from_gps_seconds(0.0)
        epochs = [e0, e0 + 60.0, e0 + 120.0]
        s = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        states = np.vstack([s, s, s])

        out_epochs, out_states = brahe.batch_state_koe_mean_to_osc(
            epochs, states, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )

        assert len(out_epochs) == 3
        assert out_states.shape == (3, 6)

        single = brahe.state_koe_mean_to_osc(
            s, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )
        assert out_states[0] == pytest.approx(single)

    def test_batch_numerical_osc_to_mean_averages(self):
        """Numerical osc->mean, windowed over one full period, recovers the seed mean a/e/i.

        Builds a synthetic osculating series by holding a/e/i/raan/argp fixed and
        analytically expanding the mean anomaly linearly with time (via
        state_koe_mean_to_osc per sample) over two full orbital periods -- no
        propagation needed. A centered window exactly one period wide, anchored near
        the middle of the series, should average out the short-period J2 oscillation
        and recover the seed mean elements.
        """
        a = brahe.R_EARTH + 500e3
        e = 0.01
        i = 45.0
        raan = 30.0
        argp = 60.0

        period = brahe.orbital_period(a)
        n_per_period = 60
        n_periods = 2
        n_samples = n_per_period * n_periods + 1
        dt = period / n_per_period

        e0 = brahe.Epoch.from_gps_seconds(0.0)
        epochs = [e0 + k * dt for k in range(n_samples)]
        osc_rows = []
        for k in range(n_samples):
            mean_anomaly = (k * (360.0 / n_per_period)) % 360.0
            mean_koe = np.array([a, e, i, raan, argp, mean_anomaly])
            osc_rows.append(
                brahe.state_koe_mean_to_osc(
                    mean_koe, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
                )
            )
        states = np.vstack(osc_rows)

        cfg = brahe.NumericalConfig(
            period, brahe.WindowAlignment.CENTERED, brahe.EdgeHandling.TRUNCATE
        )
        method = brahe.MeanElementMethod.numerical(cfg)

        out_epochs, out_states = brahe.batch_state_koe_osc_to_mean(
            epochs, states, method, brahe.AngleFormat.DEGREES
        )

        assert len(out_epochs) > 0
        mid = len(out_epochs) // 2
        assert out_states[mid, 0] == pytest.approx(a, abs=50.0)
        assert out_states[mid, 1] == pytest.approx(e, abs=1e-3)
        assert out_states[mid, 2] == pytest.approx(i, abs=0.05)

    def test_batch_numerical_mean_to_osc_requires_inverse_raises(self):
        """Numerical mean->osc without an InverseConfig must raise (batch wiring smoke test)."""
        e0 = brahe.Epoch.from_gps_seconds(0.0)
        epochs = [e0, e0 + 60.0]
        s = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        states = np.vstack([s, s])

        cfg = brahe.NumericalConfig(
            5400.0, brahe.WindowAlignment.CENTERED, brahe.EdgeHandling.TRUNCATE
        )
        method = brahe.MeanElementMethod.numerical(cfg)

        with pytest.raises(Exception):
            brahe.batch_state_koe_mean_to_osc(
                epochs, states, method, brahe.AngleFormat.DEGREES
            )


class TestStateKoeMeanOnProviders:
    """Tests for state_koe_mean methods on propagators and trajectories."""

    def test_sgp_propagator_state_koe_mean(self):
        """Test SGPPropagator.state_koe_mean returns different values from osculating."""
        # ISS TLE
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        prop = brahe.SGPPropagator.from_tle(line1, line2)

        epoch = prop.epoch
        # state_koe_osc returns osculating elements
        osc = prop.state_koe_osc(epoch, brahe.AngleFormat.DEGREES)
        mean = prop.state_koe_mean(epoch, brahe.AngleFormat.DEGREES)

        # Mean and osculating should differ due to J2 effects
        assert len(osc) == 6
        assert len(mean) == 6
        # Semi-major axis should differ (J2 effect)
        assert osc[0] != pytest.approx(mean[0], abs=0.1)

    def test_sgp_propagator_states_koe_mean(self):
        """Test SGPPropagator.states_koe_mean returns list of mean elements."""
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        prop = brahe.SGPPropagator.from_tle(line1, line2)

        epoch0 = prop.epoch
        epochs = [epoch0 + i * 3600.0 for i in range(3)]
        mean_list = prop.states_koe_mean(epochs, brahe.AngleFormat.DEGREES)

        assert len(mean_list) == 3
        for mean in mean_list:
            assert len(mean) == 6

    def test_keplerian_propagator_state_koe_mean(self):
        """Test KeplerianPropagator.state_koe_mean returns different values from osculating."""
        epc = brahe.Epoch.from_datetime(
            2024, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC
        )
        oe = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        state = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
        prop = brahe.KeplerianPropagator.from_eci(epc, state, 60.0)

        # state_koe_osc returns osculating elements
        osc = prop.state_koe_osc(epc, brahe.AngleFormat.DEGREES)
        mean = prop.state_koe_mean(epc, brahe.AngleFormat.DEGREES)

        # Mean and osculating should differ due to J2 effects
        assert len(osc) == 6
        assert len(mean) == 6
        # Semi-major axis should differ (J2 effect)
        assert osc[0] != pytest.approx(mean[0], abs=0.1)

    def test_keplerian_propagator_states_koe_mean(self):
        """Test KeplerianPropagator.states_koe_mean returns list of mean elements."""
        epc = brahe.Epoch.from_datetime(
            2024, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC
        )
        oe = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        state = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
        prop = brahe.KeplerianPropagator.from_eci(epc, state, 60.0)

        epochs = [epc + i * 3600.0 for i in range(3)]
        mean_list = prop.states_koe_mean(epochs, brahe.AngleFormat.DEGREES)

        assert len(mean_list) == 3
        for mean in mean_list:
            assert len(mean) == 6

    def test_orbit_trajectory_state_koe_mean(self):
        """Test OrbitTrajectory.state_koe_mean returns different values from osculating."""
        traj = brahe.OrbitTrajectory(
            6,
            brahe.OrbitFrame.ECI,
            brahe.OrbitRepresentation.CARTESIAN,
            None,
        )
        epc = brahe.Epoch.from_datetime(
            2024, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC
        )
        oe = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        state = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
        traj.add(epc, state)

        # state_koe_osc returns osculating elements
        osc = traj.state_koe_osc(epc, brahe.AngleFormat.DEGREES)
        mean = traj.state_koe_mean(epc, brahe.AngleFormat.DEGREES)

        # Mean and osculating should differ due to J2 effects
        assert len(osc) == 6
        assert len(mean) == 6
        # Semi-major axis should differ (J2 effect)
        assert osc[0] != pytest.approx(mean[0], abs=0.1)

    def test_mean_elements_consistent_with_standalone_function(self):
        """Test that state_koe_mean produces same results as standalone conversion."""
        epc = brahe.Epoch.from_datetime(
            2024, 1, 1, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC
        )
        oe = np.array([brahe.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        state = brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)
        prop = brahe.KeplerianPropagator.from_eci(epc, state, 60.0)

        # Get mean elements via propagator method
        mean_via_prop = prop.state_koe_mean(epc, brahe.AngleFormat.DEGREES)

        # Get mean elements via standalone function (osc -> mean conversion)
        # state_koe_osc returns osculating elements
        osc = prop.state_koe_osc(epc, brahe.AngleFormat.DEGREES)
        mean_via_func = brahe.state_koe_osc_to_mean(
            osc, BROUWER_LYDDANE, brahe.AngleFormat.DEGREES
        )

        # Should be identical
        np.testing.assert_array_almost_equal(mean_via_prop, mean_via_func, decimal=10)
