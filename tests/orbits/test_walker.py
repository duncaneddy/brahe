"""Tests for WalkerConstellationGenerator."""

import math
import pytest

import brahe as bh
from brahe import AngleFormat, TimeSystem, WalkerPattern


@pytest.fixture
def epoch():
    """Create a test epoch."""
    return bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem.UTC)


class TestWalkerConstellationGeneratorBasic:
    """Basic tests for WalkerConstellationGenerator."""

    def test_new_basic(self, epoch):
        """Test basic construction with degrees."""
        walker = bh.WalkerConstellationGenerator(
            t=12,
            p=3,
            f=1,
            semi_major_axis=7000e3,
            eccentricity=0.001,
            inclination=98.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        assert walker.total_satellites == 12
        assert walker.num_planes == 3
        assert walker.phasing == 1
        assert walker.satellites_per_plane == 4
        assert walker.semi_major_axis == pytest.approx(7000e3)
        assert walker.pattern == WalkerPattern.DELTA

    def test_new_radians(self, epoch):
        """Test construction with radians."""
        walker = bh.WalkerConstellationGenerator(
            t=12,
            p=3,
            f=1,
            semi_major_axis=7000e3,
            eccentricity=0.001,
            inclination=math.radians(98.0),
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.RADIANS,
            pattern=WalkerPattern.DELTA,
        )

        assert walker.total_satellites == 12
        assert walker.satellites_per_plane == 4

    def test_invalid_not_divisible(self, epoch):
        """Test that T not divisible by P raises error."""
        with pytest.raises(bh.PanicException):
            bh.WalkerConstellationGenerator(
                t=10,
                p=3,
                f=1,
                semi_major_axis=7000e3,
                eccentricity=0.001,
                inclination=98.0,
                argument_of_perigee=0.0,
                reference_raan=0.0,
                reference_mean_anomaly=0.0,
                epoch=epoch,
                angle_format=AngleFormat.DEGREES,
                pattern=WalkerPattern.DELTA,
            )

    def test_invalid_phasing(self, epoch):
        """Test that phasing >= P raises error."""
        with pytest.raises(bh.PanicException):
            bh.WalkerConstellationGenerator(
                t=12,
                p=3,
                f=3,  # Invalid: must be < 3
                semi_major_axis=7000e3,
                eccentricity=0.001,
                inclination=98.0,
                argument_of_perigee=0.0,
                reference_raan=0.0,
                reference_mean_anomaly=0.0,
                epoch=epoch,
                angle_format=AngleFormat.DEGREES,
                pattern=WalkerPattern.DELTA,
            )

    def test_invalid_zero_planes(self, epoch):
        """Test that P=0 raises error."""
        with pytest.raises(bh.PanicException):
            bh.WalkerConstellationGenerator(
                t=12,
                p=0,
                f=0,
                semi_major_axis=7000e3,
                eccentricity=0.001,
                inclination=98.0,
                argument_of_perigee=0.0,
                reference_raan=0.0,
                reference_mean_anomaly=0.0,
                epoch=epoch,
                angle_format=AngleFormat.DEGREES,
                pattern=WalkerPattern.DELTA,
            )


class TestWalkerConstellationGeneratorElements:
    """Tests for satellite element generation."""

    def test_raan_spacing_delta(self, epoch):
        """Test RAAN spacing between planes for Walker Delta (360 degree spread)."""
        # 6 satellites in 3 planes -> planes at RAAN 0, 120, 240 degrees
        walker = bh.WalkerConstellationGenerator(
            t=6,
            p=3,
            f=0,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        elem0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)
        elem1 = walker.satellite_elements(1, 0, AngleFormat.DEGREES)
        elem2 = walker.satellite_elements(2, 0, AngleFormat.DEGREES)

        # RAAN is index 3
        assert elem0[3] == pytest.approx(0.0, abs=1e-10)
        assert elem1[3] == pytest.approx(120.0, abs=1e-10)
        assert elem2[3] == pytest.approx(240.0, abs=1e-10)

    def test_raan_spacing_star(self, epoch):
        """Test RAAN spacing between planes for Walker Star (180 degree spread)."""
        # 6 satellites in 3 planes -> planes at RAAN 0, 60, 120 degrees
        walker = bh.WalkerConstellationGenerator(
            t=6,
            p=3,
            f=0,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.STAR,
        )

        elem0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)
        elem1 = walker.satellite_elements(1, 0, AngleFormat.DEGREES)
        elem2 = walker.satellite_elements(2, 0, AngleFormat.DEGREES)

        # RAAN is index 3 - Star pattern uses 180/3 = 60 degree spacing
        assert elem0[3] == pytest.approx(0.0, abs=1e-10)
        assert elem1[3] == pytest.approx(60.0, abs=1e-10)
        assert elem2[3] == pytest.approx(120.0, abs=1e-10)

    def test_ma_spacing_within_plane(self, epoch):
        """Test mean anomaly spacing within a plane."""
        # 6 satellites in 2 planes -> 3 per plane, MA spacing = 120 degrees
        walker = bh.WalkerConstellationGenerator(
            t=6,
            p=2,
            f=0,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        elem0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)
        elem1 = walker.satellite_elements(0, 1, AngleFormat.DEGREES)
        elem2 = walker.satellite_elements(0, 2, AngleFormat.DEGREES)

        # MA is index 5
        assert elem0[5] == pytest.approx(0.0, abs=1e-10)
        assert elem1[5] == pytest.approx(120.0, abs=1e-10)
        assert elem2[5] == pytest.approx(240.0, abs=1e-10)

    def test_phasing(self, epoch):
        """Test phasing factor between planes."""
        # 12:3:1 constellation
        # Phase offset per plane = 1 * 360/12 = 30 degrees
        walker = bh.WalkerConstellationGenerator(
            t=12,
            p=3,
            f=1,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        elem_p0_s0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)
        elem_p1_s0 = walker.satellite_elements(1, 0, AngleFormat.DEGREES)
        elem_p2_s0 = walker.satellite_elements(2, 0, AngleFormat.DEGREES)

        # Plane 0: MA = 0
        # Plane 1: MA = 0 + 1*1*(360/12) = 30 degrees
        # Plane 2: MA = 0 + 2*1*(360/12) = 60 degrees
        assert elem_p0_s0[5] == pytest.approx(0.0, abs=1e-10)
        assert elem_p1_s0[5] == pytest.approx(30.0, abs=1e-10)
        assert elem_p2_s0[5] == pytest.approx(60.0, abs=1e-10)

    def test_reference_offsets(self, epoch):
        """Test non-zero reference RAAN and MA."""
        walker = bh.WalkerConstellationGenerator(
            t=4,
            p=2,
            f=0,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=30.0,
            reference_mean_anomaly=15.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        elem0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)

        # First satellite should have the reference values
        assert elem0[3] == pytest.approx(30.0, abs=1e-10)  # RAAN
        assert elem0[5] == pytest.approx(15.0, abs=1e-10)  # MA


class TestWalkerConstellationGeneratorPropagators:
    """Tests for propagator generation methods."""

    def test_as_keplerian_propagators_count(self, epoch):
        """Test that correct number of propagators is created."""
        walker = bh.WalkerConstellationGenerator(
            t=6,
            p=2,
            f=1,
            semi_major_axis=7000e3,
            eccentricity=0.001,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        props = walker.as_keplerian_propagators(60.0)

        assert len(props) == 6

    def test_as_keplerian_propagators_with_name(self, epoch):
        """Test propagator naming with base name."""
        walker = bh.WalkerConstellationGenerator(
            t=6,
            p=2,
            f=1,
            semi_major_axis=7000e3,
            eccentricity=0.001,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        ).with_base_name("Sat")

        props = walker.as_keplerian_propagators(60.0)

        assert props[0].get_name() == "Sat-P0-S0"
        assert props[0].get_id() == 0
        assert props[3].get_name() == "Sat-P1-S0"
        assert props[3].get_id() == 3

    def test_as_sgp_propagators(self, epoch):
        """Test SGP propagator generation."""
        walker = bh.WalkerConstellationGenerator(
            t=6,
            p=2,
            f=1,
            semi_major_axis=bh.R_EARTH + 780e3,
            eccentricity=0.001,
            inclination=98.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        props = walker.as_sgp_propagators(60.0, bstar=0.0, ndt2=0.0, nddt6=0.0)

        assert len(props) == 6

    def test_as_numerical_propagators(self, epoch):
        """Test numerical propagator generation."""
        walker = bh.WalkerConstellationGenerator(
            t=4,
            p=2,
            f=0,
            semi_major_axis=bh.R_EARTH + 500e3,
            eccentricity=0.001,
            inclination=45.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        prop_config = bh.NumericalPropagationConfig.default()
        force_config = bh.ForceModelConfig.earth_gravity()

        props = walker.as_numerical_propagators(prop_config, force_config)

        assert len(props) == 4


class TestWalkerPatternComparison:
    """Tests comparing Walker Delta and Walker Star patterns."""

    def test_delta_vs_star_raan_spread(self, epoch):
        """Test that Delta and Star have different RAAN spreads."""
        # Create both patterns with same parameters
        delta = bh.WalkerConstellationGenerator(
            t=6,
            p=6,
            f=0,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=86.4,  # Iridium-like inclination
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        )

        star = bh.WalkerConstellationGenerator(
            t=6,
            p=6,
            f=0,
            semi_major_axis=7000e3,
            eccentricity=0.0,
            inclination=86.4,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.STAR,
        )

        # Delta: 360/6 = 60 degree RAAN spacing
        delta_elem5 = delta.satellite_elements(5, 0, AngleFormat.DEGREES)
        # Last plane RAAN = 5 * 60 = 300 degrees
        assert delta_elem5[3] == pytest.approx(300.0, abs=1e-10)

        # Star: 180/6 = 30 degree RAAN spacing
        star_elem5 = star.satellite_elements(5, 0, AngleFormat.DEGREES)
        # Last plane RAAN = 5 * 30 = 150 degrees
        assert star_elem5[3] == pytest.approx(150.0, abs=1e-10)


class TestWalkerConstellationGeneratorGPSExample:
    """Test a GPS-like Walker constellation."""

    def test_gps_constellation(self, epoch):
        """Test GPS-like 24:6:2 Walker Delta constellation."""
        walker = bh.WalkerConstellationGenerator(
            t=24,
            p=6,
            f=2,
            semi_major_axis=bh.R_EARTH + 20200e3,
            eccentricity=0.0,
            inclination=55.0,
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.DELTA,
        ).with_base_name("GPS")

        assert walker.total_satellites == 24
        assert walker.num_planes == 6
        assert walker.satellites_per_plane == 4
        assert walker.pattern == WalkerPattern.DELTA

        # Check RAAN spacing (60 degrees between planes)
        elem_p0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)
        elem_p1 = walker.satellite_elements(1, 0, AngleFormat.DEGREES)

        assert elem_p1[3] - elem_p0[3] == pytest.approx(60.0, abs=1e-10)

        # Check phasing (2 * 360/24 = 30 degrees per plane)
        assert elem_p1[5] - elem_p0[5] == pytest.approx(30.0, abs=1e-10)

        # Generate propagators
        props = walker.as_keplerian_propagators(60.0)
        assert len(props) == 24
        assert props[0].get_name() == "GPS-P0-S0"
        assert props[23].get_name() == "GPS-P5-S3"


class TestWalkerConstellationGeneratorIridiumExample:
    """Test an Iridium-like Walker Star constellation."""

    def test_iridium_constellation(self, epoch):
        """Test Iridium-like 66:6:2 Walker Star constellation."""
        walker = bh.WalkerConstellationGenerator(
            t=66,
            p=6,
            f=2,
            semi_major_axis=bh.R_EARTH + 780e3,  # ~780 km altitude
            eccentricity=0.0,
            inclination=86.4,  # Near-polar
            argument_of_perigee=0.0,
            reference_raan=0.0,
            reference_mean_anomaly=0.0,
            epoch=epoch,
            angle_format=AngleFormat.DEGREES,
            pattern=WalkerPattern.STAR,
        ).with_base_name("IRIDIUM")

        assert walker.total_satellites == 66
        assert walker.num_planes == 6
        assert walker.satellites_per_plane == 11
        assert walker.pattern == WalkerPattern.STAR

        # Check RAAN spacing (180/6 = 30 degrees between planes for Star)
        elem_p0 = walker.satellite_elements(0, 0, AngleFormat.DEGREES)
        elem_p1 = walker.satellite_elements(1, 0, AngleFormat.DEGREES)

        assert elem_p1[3] - elem_p0[3] == pytest.approx(30.0, abs=1e-10)

        # Generate propagators
        props = walker.as_keplerian_propagators(60.0)
        assert len(props) == 66
        assert props[0].get_name() == "IRIDIUM-P0-S0"
        assert props[65].get_name() == "IRIDIUM-P5-S10"
