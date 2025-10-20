"""
Validation tests comparing brahe access computation against STK outputs.

These tests use known spacecraft/station configurations and compare computed
access windows against ground truth data from STK (Systems Tool Kit).
"""

import pytest
import brahe as bh

# STK Ground Truth Data
# Access windows for Svalbard station over 2020-01-01
# Format: (start_epoch, end_epoch, duration_seconds)
STK_SVALBARD_ACCESS = [
    (
        bh.Epoch.from_string("2020-01-01T00:19:59.776Z"),
        bh.Epoch.from_string("2020-01-01T00:30:39.293Z"),
        639.517,
    ),
    (
        bh.Epoch.from_string("2020-01-01T01:55:16.331Z"),
        bh.Epoch.from_string("2020-01-01T02:06:35.211Z"),
        678.880,
    ),
    (
        bh.Epoch.from_string("2020-01-01T03:30:13.201Z"),
        bh.Epoch.from_string("2020-01-01T03:41:55.695Z"),
        702.494,
    ),
    (
        bh.Epoch.from_string("2020-01-01T05:04:54.081Z"),
        bh.Epoch.from_string("2020-01-01T05:16:30.793Z"),
        696.712,
    ),
    (
        bh.Epoch.from_string("2020-01-01T06:39:19.083Z"),
        bh.Epoch.from_string("2020-01-01T06:50:22.980Z"),
        663.897,
    ),
    (
        bh.Epoch.from_string("2020-01-01T08:13:24.935Z"),
        bh.Epoch.from_string("2020-01-01T08:23:45.935Z"),
        621.000,
    ),
    (
        bh.Epoch.from_string("2020-01-01T09:47:07.465Z"),
        bh.Epoch.from_string("2020-01-01T09:57:01.775Z"),
        594.310,
    ),
    (
        bh.Epoch.from_string("2020-01-01T11:20:29.103Z"),
        bh.Epoch.from_string("2020-01-01T11:30:31.919Z"),
        602.816,
    ),
    (
        bh.Epoch.from_string("2020-01-01T12:53:45.755Z"),
        bh.Epoch.from_string("2020-01-01T13:04:25.698Z"),
        639.943,
    ),
    (
        bh.Epoch.from_string("2020-01-01T14:27:20.723Z"),
        bh.Epoch.from_string("2020-01-01T14:38:41.177Z"),
        680.454,
    ),
    (
        bh.Epoch.from_string("2020-01-01T16:01:32.672Z"),
        bh.Epoch.from_string("2020-01-01T16:13:14.011Z"),
        701.339,
    ),
    (
        bh.Epoch.from_string("2020-01-01T17:36:30.019Z"),
        bh.Epoch.from_string("2020-01-01T17:48:02.441Z"),
        692.423,
    ),
    (
        bh.Epoch.from_string("2020-01-01T19:12:09.407Z"),
        bh.Epoch.from_string("2020-01-01T19:23:08.406Z"),
        658.999,
    ),
    (
        bh.Epoch.from_string("2020-01-01T20:48:14.588Z"),
        bh.Epoch.from_string("2020-01-01T20:58:36.775Z"),
        622.186,
    ),
    (
        bh.Epoch.from_string("2020-01-01T22:24:20.109Z"),
        bh.Epoch.from_string("2020-01-01T22:34:30.297Z"),
        610.188,
    ),
]


@pytest.fixture
def spacecraft_polar():
    """
    Polar orbit spacecraft from TLE.

    TLE Data:
        ID: 1
        Name: Spacecraft 1
        Inclination: 90.0 degrees
        Eccentricity: 0.001
        Mean Motion: 15.21936719 rev/day (~500 km altitude)
    """
    line1 = "1 00001U          20001.00000000  .00000000  00000-0  00000-0 0    07"
    line2 = "2 00001  90.0000   0.0000 0010000   0.0000   0.0000 15.21936719    07"

    # Create SGP4 propagator from TLE
    propagator = bh.SGPPropagator.from_tle(line1, line2)

    # Set identification
    propagator.set_name("Spacecraft 1")
    propagator.set_id(1)

    return propagator


@pytest.fixture
def station_svalbard():
    """
    Svalbard ground station in Norway.

    Location:
        Longitude: 15.396518°E
        Latitude: 78.230306°N
        Altitude: 0m

    Constraints:
        Minimum Elevation: 5°
    """
    # Coordinates from GeoJSON (longitude, latitude, altitude)
    lon = 15.396518
    lat = 78.230306
    alt = 0.0

    # Create point location (lon, lat, alt in degrees and meters)
    location = bh.PointLocation(lon, lat, alt)
    location.set_name("Svalbard")

    return location


def test_access_svalbard_stk_validation(spacecraft_polar, station_svalbard):
    """
    Validate access computation against STK ground truth.

    This test compares brahe's access window computation with output from
    STK (Systems Tool Kit) for a polar orbit satellite accessing the
    Svalbard ground station over a 24-hour period.

    Validation criteria:
    - Number of access windows matches (15 windows)
    - Window start times match within 0.1 seconds
    - Window end times match within 0.1 seconds
    - Window durations match within 0.1 seconds
    """
    # Set simulation duration
    t_start = bh.Epoch.from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    t_end = bh.Epoch.from_datetime(2020, 1, 2, 0, 0, 0.0, 0.0, bh.UTC)

    # Create elevation constraint (0 degrees to match STK configuration)
    constraint = bh.ElevationConstraint(min_elevation_deg=0.0, max_elevation_deg=None)

    # Configure access search with fine time resolution for STK validation
    config = bh.AccessSearchConfig(
        initial_time_step=0.1,  # 0.1-second steps for sub-second accuracy
        adaptive_step=False,
    )

    # Find access windows with tight time tolerance
    windows = bh.location_accesses(
        station_svalbard,
        spacecraft_polar,
        t_start,
        t_end,
        constraint,
        None,  # property_computers
        config,
        0.01,  # time_tolerance in seconds
    )

    # Validate number of windows
    assert len(windows) == 15, f"Expected 15 access windows, found {len(windows)}"

    # Validate each window against STK data
    for idx, window in enumerate(windows):
        stk_start, stk_end, stk_duration = STK_SVALBARD_ACCESS[idx]

        # Get window times
        window_start = window.start
        window_end = window.end
        window_duration = window.duration

        # Compute time differences in seconds (direct Epoch subtraction)
        start_diff = abs(window_start - stk_start)
        end_diff = abs(window_end - stk_end)
        duration_diff = abs(window_duration - stk_duration)

        # Assert within tolerance (0.1 seconds)
        assert start_diff < 0.1, (
            f"Window {idx}: Start time differs by {start_diff:.3f}s\n"
            f"  Expected: {stk_start}\n"
            f"  Got:      {window_start}"
        )

        assert end_diff < 0.1, (
            f"Window {idx}: End time differs by {end_diff:.3f}s\n"
            f"  Expected: {stk_end}\n"
            f"  Got:      {window_end}"
        )

        assert duration_diff < 0.1, (
            f"Window {idx}: Duration differs by {duration_diff:.3f}s\n"
            f"  Expected: {stk_duration:.3f}s\n"
            f"  Got:      {window_duration:.3f}s"
        )


def test_access_svalbard_stk_with_5deg_elevation(spacecraft_polar, station_svalbard):
    """
    Test access windows with 5-degree elevation constraint.

    This test uses the station's nominal 5-degree minimum elevation
    (as specified in the station configuration) rather than 0 degrees.
    This should produce fewer and shorter access windows.
    """
    # Set simulation duration
    t_start = bh.Epoch.from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    t_end = bh.Epoch.from_datetime(2020, 1, 2, 0, 0, 0.0, 0.0, bh.UTC)

    # Create elevation constraint with station's nominal value
    constraint = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)

    # Configure access search with fine time resolution
    config = bh.AccessSearchConfig(initial_time_step=1.0, adaptive_step=False)

    # Find access windows
    windows = bh.location_accesses(
        station_svalbard,
        spacecraft_polar,
        t_start,
        t_end,
        constraint,
        None,  # property_computers
        config,
        0.01,  # time_tolerance in seconds
    )

    # Should have same number of windows, but shorter duration
    assert len(windows) == 15, f"Expected 15 access windows, found {len(windows)}"

    # Verify all windows have reasonable durations
    for idx, window in enumerate(windows):
        duration = window.duration

        # Each window should be shorter than the 0-degree case
        stk_duration = STK_SVALBARD_ACCESS[idx][2]

        assert duration < stk_duration, (
            f"Window {idx}: Duration with 5° elevation ({duration:.1f}s) "
            f"should be less than 0° case ({stk_duration:.1f}s)"
        )

        # All windows should still be at least a few minutes
        assert duration > 300, (
            f"Window {idx}: Duration ({duration:.1f}s) seems too short"
        )


def test_access_window_properties_svalbard(spacecraft_polar, station_svalbard):
    """
    Validate that access window properties are computed correctly.

    Checks that computed properties (azimuth, elevation, etc.) are
    within valid ranges and physically reasonable.
    """
    # Set simulation duration (just first few hours for speed)
    t_start = bh.Epoch.from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    t_end = bh.Epoch.from_datetime(2020, 1, 1, 6, 0, 0.0, 0.0, bh.UTC)

    # Create elevation constraint
    constraint = bh.ElevationConstraint(min_elevation_deg=0.0, max_elevation_deg=None)

    # Configure access search with fine time resolution
    config = bh.AccessSearchConfig(initial_time_step=1.0, adaptive_step=False)

    # Find access windows
    windows = bh.location_accesses(
        station_svalbard,
        spacecraft_polar,
        t_start,
        t_end,
        constraint,
        None,  # property_computers
        config,
        0.01,  # time_tolerance in seconds
    )

    # Should find several windows in first 6 hours
    assert len(windows) >= 3, f"Expected at least 3 windows, found {len(windows)}"

    # Validate properties for each window
    for idx, window in enumerate(windows):
        props = window.properties

        # Azimuth should be in range [0, 360]
        assert 0.0 <= props.azimuth_open <= 360.0, (
            f"Window {idx}: azimuth_open={props.azimuth_open} out of range"
        )
        assert 0.0 <= props.azimuth_close <= 360.0, (
            f"Window {idx}: azimuth_close={props.azimuth_close} out of range"
        )

        # Elevation should be in range [-90, 90], and min should be >= 0 (our constraint)
        assert -90.0 <= props.elevation_min <= 90.0, (
            f"Window {idx}: elevation_min={props.elevation_min} out of range"
        )
        assert -90.0 <= props.elevation_max <= 90.0, (
            f"Window {idx}: elevation_max={props.elevation_max} out of range"
        )
        assert props.elevation_min >= -0.1, (  # Allow small numerical error
            f"Window {idx}: elevation_min={props.elevation_min} violates 0° constraint"
        )
        assert props.elevation_max > props.elevation_min, (
            f"Window {idx}: elevation_max should be > elevation_min"
        )

        # Off-nadir should be in range [0, 180]
        assert 0.0 <= props.off_nadir_min <= 180.0, (
            f"Window {idx}: off_nadir_min={props.off_nadir_min} out of range"
        )
        assert 0.0 <= props.off_nadir_max <= 180.0, (
            f"Window {idx}: off_nadir_max={props.off_nadir_max} out of range"
        )
        assert props.off_nadir_min <= props.off_nadir_max, (
            f"Window {idx}: off_nadir_min should be <= off_nadir_max"
        )

        # Local time should be in range [0, 86400] seconds
        assert 0.0 <= props.local_time <= 86400.0, (
            f"Window {idx}: local_time={props.local_time} out of range"
        )

        # Look direction should be valid
        assert props.look_direction in [
            bh.LookDirection.LEFT,
            bh.LookDirection.RIGHT,
            bh.LookDirection.EITHER,
        ], f"Window {idx}: Invalid look_direction={props.look_direction}"

        # Asc/Dsc should be valid
        assert props.asc_dsc in [
            bh.AscDsc.ASCENDING,
            bh.AscDsc.DESCENDING,
            bh.AscDsc.EITHER,
        ], f"Window {idx}: Invalid asc_dsc={props.asc_dsc}"
