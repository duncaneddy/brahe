"""Tests for SP3 satellites, chaining, and propagators — parity with
src/datafiles/sp3/{satellite,interop}.rs."""

import itertools
import math
import pathlib
import subprocess
import sys
import textwrap
import uuid

import numpy as np
import pytest

import brahe as bh
from brahe.datafiles import (
    SP3,
    SP3_DEFAULT_LAGRANGE_DEGREE,
    SP3FileType,
    SP3Record,
    SP3Satellite,
    SP3SatelliteId,
    SP3TimeSystem,
)

STEP = 900.0
N_EPOCHS = 97

TRUNC = "test_assets/sp3/gbm19500_truncated.sp3"
AFTER_DROP = "test_assets/sp3/gbm19500_after_drop.sp3"
AFTER_NO_DROP = "test_assets/sp3/gbm19500_after_no_drop.sp3"
LARGE_GAP = "test_assets/sp3/gbm19500_large_gap.sp3"
COD_SUBSET = "test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3"

ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


def sid(code):
    return SP3SatelliteId.parse(code)


def t0():
    return bh.Epoch.from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)


def gps(minute, second):
    return bh.Epoch.from_datetime(
        2024, 11, 10, 0, minute, second, 0.0, bh.TimeSystem.GPS
    )


def find(sats, code):
    return next(s for s in sats if str(s.id) == code)


def gps_propagator(raan, mean_anomaly):
    return bh.KeplerianPropagator.from_keplerian(
        t0(),
        np.array([26_560e3, 0.01, 55.0, raan, 0.0, mean_anomaly]),
        bh.AngleFormat.DEGREES,
        60.0,
    )


def build_sp3():
    """A 24 h position-only file at 900 s with four satellites: G01 absent at
    epochs 40..=43, G02 with no clock at epoch 5 and absent at epoch 7, G03
    present only at epochs 0..=10 and 20, and G04 holding only a
    zero-position record."""
    ids = ["G01", "G02", "G03", "G04"]
    propagators = [
        gps_propagator(0.0, 0.0),
        gps_propagator(60.0, 30.0),
        gps_propagator(120.0, 60.0),
        gps_propagator(180.0, 90.0),
    ]
    sp3 = SP3(ids, STEP, include_velocity=False)
    for k in range(N_EPOCHS):
        epoch = t0() + k * STEP
        index = sp3.push_epoch(epoch)
        for n, code in enumerate(ids):
            present = [
                not (40 <= k <= 43),
                k != 7,
                k <= 10 or k == 20,
                False,
            ][n]
            if not present:
                continue
            x = propagators[n].state_itrf(epoch)
            clock = None if (n == 1 and k == 5) else 1.0e-5 + 1.0e-9 * k
            sp3.set_record(code, index, SP3Record(x[:3], clock=clock))
    sp3.set_record("G04", 3, SP3Record(np.zeros(3)))
    return sp3, propagators


def build_clean_sp3():
    """A 24 h position-only file at 900 s with three gap-free satellites, and
    the propagators it was sampled from, named after the satellites."""
    names = ["G01", "G02", "G03"]
    propagators = [
        gps_propagator(raan, anomaly).with_name(name)
        for (raan, anomaly), name in zip(
            [(0.0, 0.0), (60.0, 30.0), (120.0, 60.0)], names
        )
    ]
    sp3 = SP3(names, STEP, include_velocity=False)
    for k in range(N_EPOCHS):
        epoch = t0() + k * STEP
        index = sp3.push_epoch(epoch)
        for name, prop in zip(names, propagators):
            sp3.set_record(name, index, SP3Record(prop.state_itrf(epoch)[:3]))
    return sp3, propagators


def single_satellite(code, interval, samples, include_velocity=False):
    """An SP3 file holding one satellite at the given (offset, position) samples."""
    sp3 = SP3([code], interval, include_velocity=include_velocity)
    for offset, position in samples:
        index = sp3.push_epoch(t0() + offset)
        sp3.set_record(code, index, SP3Record(np.array(position)))
    return sp3


def quadratic_position(t):
    return [1.0e7 + 100.0 * t + 0.01 * t * t, 2.0e7 - 50.0 * t, 3.0e6 + 0.002 * t * t]


def quadratic_velocity(t):
    return [100.0 + 0.02 * t, -50.0, 0.004 * t]


def record_key(r):
    def correlation(c):
        return (
            None
            if c is None
            else (c.std, c.clock_std, c.xy, c.xz, c.xc, c.yz, c.yc, c.zc)
        )

    return (
        tuple(r.position),
        r.clock,
        None if r.velocity is None else tuple(r.velocity),
        r.clock_rate,
        r.position_std,
        r.clock_std,
        r.velocity_std,
        r.clock_rate_std,
        r.clock_event,
        r.clock_predicted,
        r.maneuver,
        r.orbit_predicted,
        correlation(r.position_correlation),
        correlation(r.velocity_correlation),
    )


def assert_epoch_lists_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y


# ---------------------------------------------------------------------------
# satellite.rs
# ---------------------------------------------------------------------------


def test_SP3Satellite_gap_split():
    sp3, _ = build_sp3()
    g01 = sp3.satellite("G01")
    assert len(g01) == 93
    assert g01.has_gaps()
    assert repr(g01) == "SP3Satellite(id=G01, samples=93, segments=2)"
    segments = g01.segments()
    assert segments[0] == (t0(), t0() + 39.0 * 900.0)
    assert segments[1] == (t0() + 44.0 * 900.0, t0() + 96.0 * 900.0)
    with pytest.raises(bh.BraheError) as excinfo:
        g01.to_trajectory()
    err = str(excinfo.value)
    assert "G01" in err and "to_trajectory_segments" in err
    assert str(t0() + 39.0 * 900.0) in err and str(t0() + 44.0 * 900.0) in err
    segs = g01.to_trajectory_segments()
    assert len(segs) == 2
    assert segs[0].get_name() == "G01"
    assert segs[0].frame == bh.CelestialFrame.ITRF
    assert len(segs[0]) == 40
    assert len(segs[1]) == 53
    assert segs[1].epochs()[0] == t0() + 44.0 * 900.0


def test_SP3Satellite_accessors_skip_invalid_and_keep_bad_clocks():
    sp3, _ = build_sp3()
    g02 = sp3.satellite(sid("G02"))
    assert g02.id == sid("G02")
    assert len(g02) == N_EPOCHS - 1
    n = len(g02.epochs())
    assert g02.positions().shape == (n, 3)
    assert g02.velocities().shape == (n, 3)
    assert g02.clocks().shape == (n,)
    assert g02.clock_rates().shape == (n,)
    assert len(g02.records()) == n
    assert np.isnan(g02.clocks()[5])
    assert g02.clocks()[4] == 1.0e-5 + 4.0e-9
    assert np.isnan(g02.clock_rates()).all()
    assert all(e != t0() + 7.0 * 900.0 for e in g02.epochs())
    assert not g02.has_file_velocities()
    assert g02.start_epoch == t0()
    assert g02.end_epoch == t0() + 96.0 * 900.0
    expected = sp3.record("G02", 8)
    assert np.array_equal(g02.positions()[7], expected.position)
    assert record_key(g02.records()[7]) == record_key(expected)


def test_SP3Satellite_single_sample_run():
    sp3, _ = build_sp3()
    g03 = sp3.satellite("G03")
    assert len(g03) == 12
    assert g03.segments() == [
        (t0(), t0() + 10.0 * 900.0),
        (t0() + 20.0 * 900.0, t0() + 20.0 * 900.0),
    ]
    assert len(g03.to_trajectory_segments()) == 1
    with pytest.raises(bh.BraheError, match="G03"):
        g03.to_trajectory()
    assert np.isnan(g03.velocities()[11]).all()
    assert np.isfinite(g03.velocities()[10]).all()


def test_SP3Satellite_to_trajectory_single_sample():
    sat = single_satellite("G07", 900.0, [(0.0, [1.0e7, 2.0e7, 3.0e6])]).satellite(
        "G07"
    )
    assert not sat.has_gaps()
    assert sat.segments() == [(t0(), t0())]
    assert sat.to_trajectory_segments() == []
    with pytest.raises(bh.BraheError) as excinfo:
        sat.to_trajectory()
    err = str(excinfo.value)
    assert "G07" in err and str(t0()) in err


def test_SP3_satellite_without_valid_sample():
    sp3, _ = build_sp3()
    with pytest.raises(bh.BraheError, match="G04"):
        sp3.satellite("G04")
    with pytest.raises(bh.BraheError):
        sp3.satellite("G05")
    assert [str(s.id) for s in sp3.satellites()] == ["G01", "G02", "G03"]


def test_SP3Satellite_InterpolationConfig_propagates_to_trajectories():
    sp3, _ = build_sp3()
    g01 = sp3.satellite("G01")
    default_method = bh.InterpolationMethod.lagrange(SP3_DEFAULT_LAGRANGE_DEGREE)
    assert g01.get_interpolation_method() == default_method
    for traj in g01.to_trajectory_segments():
        assert traj.get_interpolation_method() == default_method

    method = bh.InterpolationMethod.lagrange(5)
    g01 = g01.with_interpolation_method(method)
    assert g01.get_interpolation_method() == method
    for traj in g01.to_trajectory_segments():
        assert traj.get_interpolation_method() == method

    g02 = sp3.satellite("G02")
    g02.set_interpolation_method(bh.InterpolationMethod.HERMITE_CUBIC)
    assert g02.get_interpolation_method() == bh.InterpolationMethod.HERMITE_CUBIC
    for traj in g02.to_trajectory_segments():
        assert traj.get_interpolation_method() == bh.InterpolationMethod.HERMITE_CUBIC


def test_SP3Satellite_with_interpolation_method_mutates_in_place():
    sp3, _ = build_sp3()
    g01 = sp3.satellite("G01")
    returned = g01.with_interpolation_method(bh.InterpolationMethod.LINEAR)
    assert returned is g01
    assert g01.get_interpolation_method() == bh.InterpolationMethod.LINEAR


def test_SP3Satellite_short_run_limits_lagrange_degree():
    samples = [(900.0 * k, quadratic_position(900.0 * k)) for k in range(3)]
    sat = (
        single_satellite("G08", 900.0, samples)
        .satellite("G08")
        .with_interpolation_method(bh.InterpolationMethod.lagrange(7))
    )
    assert sat.get_interpolation_method() == bh.InterpolationMethod.lagrange(7)
    assert sat.to_trajectory().get_interpolation_method() == (
        bh.InterpolationMethod.lagrange(2)
    )


def test_SP3Satellite_derived_velocity_matches_propagator():
    sp3, propagators = build_sp3()
    g01 = sp3.satellite("G01")
    half = math.ceil(SP3_DEFAULT_LAGRANGE_DEGREE / 2)
    epochs = g01.epochs()
    velocities = g01.velocities()
    for start, end in [(0, 40), (40, 93)]:
        for i in range(start + half, end - half):
            truth = propagators[0].state_itrf(epochs[i])
            np.testing.assert_allclose(velocities[i], truth[3:], rtol=0.0, atol=1e-3)


def test_SP3Satellite_derived_velocity_exact_for_polynomials():
    offsets = [300.0 * k for k in range(13)]
    sat = single_satellite(
        "G09", 300.0, [(t, quadratic_position(t)) for t in offsets]
    ).satellite("G09")
    assert not sat.has_file_velocities()
    for i, t in enumerate(offsets):
        np.testing.assert_allclose(
            sat.velocities()[i], quadratic_velocity(t), rtol=0.0, atol=1e-6
        )

    pair = single_satellite(
        "G10",
        60.0,
        [(0.0, [0.0, 1.0e7, 2.0e7]), (60.0, [600.0, 1.0e7 - 120.0, 2.0e7])],
    ).satellite("G10")
    for velocity in pair.velocities():
        np.testing.assert_allclose(velocity, [10.0, -2.0, 0.0], rtol=0.0, atol=1e-9)


def test_SP3Satellite_file_velocities():
    def file(skip_velocity_at=None):
        sp3 = SP3(["G11"], 900.0, include_velocity=True)
        for k in range(5):
            t = 900.0 * k
            index = sp3.push_epoch(t0() + t)
            velocity = None if k == skip_velocity_at else np.array([1.0, 2.0, 3.0 + t])
            sp3.set_record(
                "G11",
                index,
                SP3Record(np.array(quadratic_position(t)), velocity=velocity),
            )
        return sp3

    sat = file().satellite("G11")
    assert sat.has_file_velocities()
    assert np.array_equal(sat.velocities()[2], [1.0, 2.0, 1803.0])

    sat = file(skip_velocity_at=3).satellite("G11")
    assert not sat.has_file_velocities()
    for i, epoch in enumerate(sat.epochs()):
        np.testing.assert_allclose(
            sat.velocities()[i], quadratic_velocity(epoch - t0()), rtol=0.0, atol=1e-6
        )
    assert np.array_equal(sat.records()[2].velocity, [1.0, 2.0, 1803.0])


def test_SP3Satellite_gap_threshold_uses_larger_interval():
    p = [1.0e7, 2.0e7, 3.0e6]
    files = [
        single_satellite("G12", 300.0, [(0.0, p), (300.0, p)]),
        single_satellite("G12", 900.0, [(1200.0, p), (2100.0, p)]),
        single_satellite("G12", 900.0, [(3500.0, p), (4400.0, p)]),
    ]
    sat = find(SP3Satellite.from_files(files), "G12")
    assert sat.segments() == [
        (t0(), t0() + 2100.0),
        (t0() + 3500.0, t0() + 4400.0),
    ]
    assert sat.has_gaps()


def test_SP3Satellite_DOrbitStateProvider_returns_file_positions_at_run_edges():
    sp3, _ = build_sp3()
    g01 = sp3.satellite("G01")
    positions = g01.positions()
    velocities = g01.velocities()
    epochs = g01.epochs()
    for i in [0, 39, 40, len(g01) - 1]:
        state = g01.state_itrf(epochs[i])
        np.testing.assert_allclose(state[:3], positions[i], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(state[3:], velocities[i], rtol=0.0, atol=1e-6)
    assert g01.state_dim == 6
    assert len(g01.state(epochs[0])) == 6


def test_SP3Satellite_DOrbitStateProvider_errors_outside_runs():
    sp3, _ = build_sp3()
    g01 = sp3.satellite("G01")

    gap_midpoint = t0() + 41.5 * STEP
    with pytest.raises(bh.BraheError) as excinfo:
        g01.state_itrf(gap_midpoint)
    err = str(excinfo.value)
    assert "SP3:" in err
    assert "G01" in err
    assert str(t0() + 39.0 * STEP) in err and str(t0() + 44.0 * STEP) in err

    with pytest.raises(bh.BraheError) as excinfo:
        g01.state_itrf(t0() - 1.0)
    assert "G01" in str(excinfo.value) and str(t0()) in str(excinfo.value)
    end = t0() + 96.0 * STEP
    with pytest.raises(bh.BraheError) as excinfo:
        g01.state_itrf(end + 1.0)
    assert "G01" in str(excinfo.value) and str(end) in str(excinfo.value)

    gcrf = bh.CelestialFrame.GCRF
    degrees = bh.AngleFormat.DEGREES
    single_calls = [
        lambda: g01.state(gap_midpoint),
        lambda: g01.state_eci(gap_midpoint),
        lambda: g01.state_gcrf(gap_midpoint),
        lambda: g01.state_ecef(gap_midpoint),
        lambda: g01.state_eme2000(gap_midpoint),
        lambda: g01.state_bci(gap_midpoint),
        lambda: g01.state_bcbf(gap_midpoint),
        lambda: g01.state_in_frame(gcrf, gap_midpoint),
        lambda: g01.state_koe_osc(gap_midpoint, degrees),
        lambda: g01.state_koe_mean(gap_midpoint, degrees),
    ]
    pair = [t0(), gap_midpoint]
    batch_calls = [
        lambda: g01.states(pair),
        lambda: g01.states_eci(pair),
        lambda: g01.states_ecef(pair),
        lambda: g01.states_gcrf(pair),
        lambda: g01.states_itrf(pair),
        lambda: g01.states_eme2000(pair),
        lambda: g01.states_bci(pair),
        lambda: g01.states_bcbf(pair),
        lambda: g01.states_in_frame(gcrf, pair),
        lambda: g01.states_koe_osc(pair, degrees),
        lambda: g01.states_koe_mean(pair, degrees),
    ]
    for call in single_calls + batch_calls:
        with pytest.raises(bh.BraheError, match="G01"):
            call()


def test_SP3Satellite_DOrbitStateProvider_isolated_sample_errors():
    sp3, _ = build_sp3()
    g03 = sp3.satellite("G03")
    isolated = t0() + 20.0 * STEP
    with pytest.raises(bh.BraheError) as excinfo:
        g03.state_itrf(isolated)
    assert "G03" in str(excinfo.value) and str(isolated) in str(excinfo.value)
    with pytest.raises(bh.BraheError):
        g03.state_itrf(t0() + 15.0 * STEP)
    assert len(g03.state_itrf(t0() + 5.0 * STEP)) == 6


def test_SP3Satellite_DOrbitStateProvider_matches_segment_trajectory():
    sp3, _ = build_sp3()
    g01 = sp3.satellite("G01")
    segments = g01.to_trajectory_segments()
    t = t0() + 10.0 * STEP + 137.0
    np.testing.assert_allclose(
        g01.state_eci(t), segments[0].state_eci(t), rtol=0.0, atol=1e-9
    )

    t = t0() + 60.0 * STEP + 11.0
    np.testing.assert_allclose(
        g01.state_itrf(t), segments[1].state_itrf(t), rtol=0.0, atol=1e-9
    )

    epochs = [t0() + 100.0, t0() + 60.0 * STEP + 11.0]
    for batch, single in [
        (g01.states_gcrf, g01.state_gcrf),
        (g01.states_eci, g01.state_eci),
        (g01.states_ecef, g01.state_ecef),
        (g01.states_itrf, g01.state_itrf),
        (g01.states_eme2000, g01.state_eme2000),
        (g01.states_bci, g01.state_bci),
        (g01.states_bcbf, g01.state_bcbf),
        (g01.states, g01.state),
    ]:
        for state, epoch in zip(batch(epochs), epochs):
            assert np.array_equal(state, single(epoch))
    frame = bh.CelestialFrame.GCRF
    for state, epoch in zip(g01.states_in_frame(frame, epochs), epochs):
        assert np.array_equal(state, g01.state_in_frame(frame, epoch))
    degrees = bh.AngleFormat.DEGREES
    for state, epoch in zip(g01.states_koe_osc(epochs, degrees), epochs):
        assert np.array_equal(state, g01.state_koe_osc(epoch, degrees))
    for state, epoch in zip(g01.states_koe_mean(epochs, degrees), epochs):
        assert np.array_equal(state, g01.state_koe_mean(epoch, degrees))


def test_SP3Satellite_DOrbitStateProvider_follows_interpolation_method():
    sp3, _ = build_sp3()
    t = t0() + 10.0 * STEP + 450.0
    default = sp3.satellite("G01")
    low = sp3.satellite("G01").with_interpolation_method(
        bh.InterpolationMethod.lagrange(1)
    )
    assert np.linalg.norm(default.state_itrf(t) - low.state_itrf(t)) > 1.0


def test_SP3Satellite_Identifiable():
    sp3, _ = build_sp3()
    sat = sp3.satellite("G01")
    assert sat.get_name() == "G01"
    assert sat.get_id() is None
    assert sat.get_uuid() is None

    sat.set_name("Block IIF")
    assert sat.get_name() == "Block IIF"
    sat.set_id(7)
    assert sat.get_id() == 7
    value = str(uuid.uuid4())
    sat = sat.with_uuid(value)
    assert sat.get_uuid() == value
    assert (sat.get_name(), sat.get_id()) == ("Block IIF", 7)

    renamed = sp3.satellite("G02").with_name("spare")
    assert renamed.get_name() == "spare"
    for traj in renamed.to_trajectory_segments():
        assert traj.get_name() == "spare"

    sat = sp3.satellite("G02").with_new_uuid().with_id(3)
    assert sat.get_uuid() is not None
    sat.generate_uuid()
    sat.set_identity("a", None, 9)
    assert (sat.get_name(), sat.get_uuid(), sat.get_id()) == ("a", None, 9)
    sat = sat.with_identity(None, value, None)
    assert (sat.get_name(), sat.get_uuid(), sat.get_id()) == (None, value, None)

    with pytest.raises(ValueError):
        sat.with_uuid("not-a-uuid")
    with pytest.raises(ValueError):
        sat.set_identity(None, "not-a-uuid", None)


def assert_windows_match(sp3_windows, reference):
    assert len(reference) > 0
    assert len(sp3_windows) == len(reference)
    for a, b in zip(sp3_windows, reference):
        assert abs(a.window_open - b.window_open) < 1.0
        assert abs(a.window_close - b.window_close) < 1.0
        assert a.satellite_name == b.satellite_name


def test_SP3Satellite_location_accesses_matches_propagator():
    sp3, propagators = build_clean_sp3()
    station = bh.PointLocation(10.0, 45.0, 0.0)
    constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
    start, end = t0(), t0() + 86400.0

    g01 = sp3.satellite("G01")
    windows = bh.location_accesses(station, g01, start, end, constraint)
    reference = bh.location_accesses(station, propagators[0], start, end, constraint)
    assert_windows_match(windows, reference)
    assert windows[0].satellite_name == "G01"


def test_SP3Satellite_location_accesses_constellation():
    sp3, propagators = build_clean_sp3()
    station = bh.PointLocation(10.0, 45.0, 0.0)
    constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
    start, end = t0(), t0() + 86400.0

    satellites = sp3.satellites()
    assert len(satellites) == 3
    windows = bh.location_accesses(station, satellites, start, end, constraint)
    reference = bh.location_accesses(station, propagators, start, end, constraint)
    assert_windows_match(windows, reference)
    for name in ["G01", "G02", "G03"]:
        assert any(w.satellite_name == name for w in windows), name


def test_SP3Satellite_location_accesses_polygon_and_mixed_list():
    sp3, propagators = build_clean_sp3()
    polygon = bh.PolygonLocation(
        np.array(
            [[9.0, 44.0, 0.0], [11.0, 44.0, 0.0], [11.0, 46.0, 0.0], [9.0, 46.0, 0.0]]
        )
    )
    constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
    start, end = t0(), t0() + 86400.0
    windows = bh.location_accesses([polygon], sp3.satellites(), start, end, constraint)
    assert len(windows) > 0

    with pytest.raises(TypeError, match="SP3Satellite"):
        bh.location_accesses(
            polygon, [sp3.satellite("G01"), propagators[0]], start, end, constraint
        )


def test_sp3_satellite_location_accesses():
    sp3 = SP3.from_file(COD_SUBSET)
    sats = sp3.satellites()
    station = bh.PointLocation(-122.17, 37.43, 30.0)
    windows = bh.location_accesses(
        station,
        sats,
        sp3.start_epoch,
        sp3.end_epoch,
        bh.ElevationConstraint(min_elevation_deg=10.0),
    )
    assert len(windows) > 0
    assert {w.satellite_name for w in windows} <= {str(s.id) for s in sats}


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

ACCESS_CALLBACK_SCRIPT = textwrap.dedent(
    """
    import sys

    import brahe as bh

    bh.set_global_eop_provider(
        bh.FileEOPProvider.from_standard_file(
            "test_assets/finals.all.iau2000.txt", True, "Hold"
        )
    )


    class AboveHorizon(bh.AccessConstraintComputer):
        def evaluate(self, epoch, satellite_state_ecef, location_ecef):
            relative = satellite_state_ecef[:3] - location_ecef
            return bool(relative @ location_ecef > 0.0)

        def name(self):
            return "AboveHorizon"


    class MidpointZ(bh.AccessPropertyComputer):
        def compute(
            self, window, sample_epochs, sample_states_ecef, location_ecef, location_geodetic
        ):
            return {"mid_z": float(sample_states_ecef[len(sample_states_ecef) // 2, 2])}

        def property_names(self):
            return ["mid_z"]


    callback, provider, path = sys.argv[1:4]
    sp3 = bh.SP3.from_file(path)
    satellites = sp3.satellites()
    propagators = satellites if provider == "sp3" else [s.to_trajectory() for s in satellites]
    elevation = bh.ElevationConstraint(min_elevation_deg=10.0)
    if callback == "constraint":
        constraint = AboveHorizon()
        computers = None
    else:
        constraint = elevation
        computers = [MidpointZ()]
    config = bh.AccessSearchConfig(initial_time_step=60.0, parallel=True)
    windows = bh.location_accesses(
        [bh.PointLocation(-122.17, 37.43, 30.0), bh.PointLocation(10.0, 45.0, 0.0)],
        propagators,
        sp3.start_epoch,
        sp3.end_epoch,
        constraint,
        property_computers=computers,
        config=config,
    )
    assert len(windows) > 0
    if computers is not None:
        assert all("mid_z" in w.properties.additional for w in windows)
    print(len(windows))
    """
)


def run_access_callback_script(callback, provider):
    return subprocess.run(
        [sys.executable, "-c", ACCESS_CALLBACK_SCRIPT, callback, provider, COD_SUBSET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("callback", ["constraint", "property_computer"])
def test_SP3Satellite_location_accesses_python_callbacks_parallel(callback):
    result = run_access_callback_script(callback, "sp3")
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0


def smooth_itrf_position(propagator, epoch):
    """ITRF position of ``propagator`` with the Earth rotation angle evaluated
    from a two-part UT1 Julian date (as ``iauEra00``), which is smooth in time
    below the 0.6 µs resolution of a single-float MJD."""
    year, month, day, hour, minute, second, nanosecond = (
        epoch.to_datetime_as_time_system(bh.TimeSystem.UT1)
    )
    mjd_day = round(
        bh.Epoch.from_date(year, month, day, bh.TimeSystem.UT1).mjd_as_time_system(
            bh.TimeSystem.UT1
        )
    )
    day_fraction = (
        3600.0 * hour + 60.0 * minute + second + 1e-9 * nanosecond
    ) / 86400.0
    d1, d2 = day_fraction, bh.MJD_ZERO + mjd_day
    t = d1 + (d2 - 2451545.0)
    f = math.fmod(d1, 1.0) + math.fmod(d2, 1.0)
    era = math.fmod(
        2.0 * math.pi * (f + 0.7790572732640 + 0.00273781191135448 * t), 2.0 * math.pi
    )
    c, s = math.cos(era), math.sin(era)
    rotation = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    gcrf = propagator.state_gcrf(epoch)
    return (
        bh.polar_motion(epoch)
        @ rotation
        @ bh.bias_precession_nutation(epoch)
        @ gcrf[:3]
    )


def reference_itrf_velocity(propagator, epoch, h):
    """Fourth-order central difference, with step ``h`` (s), of
    :func:`smooth_itrf_position`."""

    def p(dt):
        return smooth_itrf_position(propagator, epoch + dt)

    return (p(-2.0 * h) - 8.0 * p(-h) + 8.0 * p(h) - p(2.0 * h)) / (12.0 * h)


def itrf_errors(sat, propagator):
    """Position interior, position edge (m), velocity interior, and velocity
    edge (m/s) errors of ``sat`` at its default interpolation method."""
    epochs = sat.epochs()
    velocities = sat.velocities()
    n = len(epochs)
    half = SP3_DEFAULT_LAGRANGE_DEGREE // 2
    errors = [0.0, 0.0, 0.0, 0.0]
    for i, epoch in enumerate(epochs):
        reference = reference_itrf_velocity(propagator, epoch, 10.0)
        error = np.linalg.norm(velocities[i] - reference)
        slot = 2 if (i >= half and i + half < n) else 3
        errors[slot] = max(errors[slot], error)
        if i + 1 < n:
            t = epoch + 0.5 * STEP
            error = np.linalg.norm(
                sat.state(t)[:3] - smooth_itrf_position(propagator, t)
            )
            slot = 0 if (i >= half and i + 1 + half < n) else 1
            errors[slot] = max(errors[slot], error)
    return errors


def test_SP3Satellite_interpolation_accuracy_keplerian():
    sp3, propagators = build_clean_sp3()
    half = SP3_DEFAULT_LAGRANGE_DEGREE // 2
    gcrf_position = [0.0, 0.0]
    gcrf_velocity = [0.0, 0.0]
    for code, propagator in zip(["G01", "G02", "G03"], propagators):
        sat = sp3.satellite(code)
        position_interior, position_edge, velocity_interior, velocity_edge = (
            itrf_errors(sat, propagator)
        )
        assert position_interior <= 1e-3, code
        assert velocity_interior <= 1e-4, code
        assert position_edge <= 2e-2, code
        assert velocity_edge <= 5e-4, code

        gcrf = sat.to_trajectory().to_gcrf()
        epochs = sat.epochs()
        n = len(epochs)
        for i, epoch in enumerate(epochs):
            interior = i >= half and i + half < n
            error = np.linalg.norm(
                gcrf.interpolate(epoch)[3:] - propagator.state_gcrf(epoch)[3:]
            )
            slot = 0 if interior else 1
            gcrf_velocity[slot] = max(gcrf_velocity[slot], error)
            if i + 1 < n:
                t = epoch + 0.5 * STEP
                interior = i >= half and i + 1 + half < n
                error = np.linalg.norm(
                    gcrf.interpolate(t)[:3] - propagator.state_gcrf(t)[:3]
                )
                slot = 0 if interior else 1
                gcrf_position[slot] = max(gcrf_position[slot], error)
    assert gcrf_position[0] <= 1e-3 and gcrf_velocity[0] <= 1e-4
    assert gcrf_position[1] <= 2e-2 and gcrf_velocity[1] <= 5e-4


def lagrange_derivative_weights(times, node):
    """Derivatives at ``times[node]`` of the Lagrange basis polynomials."""
    xi = times[node]
    weights = []
    for j, xj in enumerate(times):
        if j == node:
            weights.append(
                sum(1.0 / (xi - xk) for k, xk in enumerate(times) if k != node)
            )
            continue
        w = 1.0 / (xj - xi)
        for k, xk in enumerate(times):
            if k not in (node, j):
                w *= (xi - xk) / (xj - xk)
        weights.append(w)
    return weights


def test_SP3Satellite_interpolation_accuracy_file_precision():
    sp3, propagators = build_clean_sp3()
    parsed = SP3.from_str(sp3.to_string())

    m = SP3_DEFAULT_LAGRANGE_DEGREE + 1
    times = [STEP * k for k in range(m)]

    def basis(j, t):
        return math.prod(
            (t - times[k]) / (times[j] - times[k]) for k in range(m) if k != j
        )

    center = 0.5 * (times[m // 2 - 1] + times[m // 2])
    lebesgue = sum(abs(basis(j, center)) for j in range(m))
    derivative_lebesgue = max(
        sum(abs(w) for w in lagrange_derivative_weights(times, node))
        for node in (m // 2 - 1, m // 2)
    )
    rounding = math.sqrt(3.0) * 0.5e-3

    for code, propagator in zip(["G01", "G02", "G03"], propagators):
        exact_sat = sp3.satellite(code)
        rounded_sat = parsed.satellite(code)
        assert (
            np.abs(rounded_sat.positions() - exact_sat.positions()).max()
            <= 0.5e-3 + 1e-9
        )
        exact = itrf_errors(exact_sat, propagator)
        rounded = itrf_errors(rounded_sat, propagator)
        assert rounded[0] <= exact[0] + lebesgue * rounding, (code, rounded)
        assert rounded[2] <= exact[2] + derivative_lebesgue * rounding, (code, rounded)


def test_SP3Satellite_interpolation_accuracy_cod():
    file = SP3.from_file(COD_SUBSET)
    ids = [str(i) for i in file.satellite_ids]
    assert ids == ["G02", "G05", "E11"]
    assert file.num_epochs == 72

    subsampled = SP3(ids, 900.0, include_velocity=False)
    file_epochs = file.epochs
    for k in range(0, file.num_epochs, 3):
        index = subsampled.push_epoch(file_epochs[k])
        for code in ids:
            subsampled.set_record(code, index, file.record(code, k))

    n = subsampled.num_epochs
    half = SP3_DEFAULT_LAGRANGE_DEGREE // 2
    max_interior_error = 0.0
    for code in ids:
        sat = subsampled.satellite(code)
        for k in range(file.num_epochs):
            if k % 3 == 0:
                continue
            before = k // 3
            if before < half or before + 1 + half >= n:
                continue
            state = sat.state(file_epochs[k])
            truth = file.record(code, k).position
            max_interior_error = max(
                max_interior_error, np.linalg.norm(state[:3] - truth)
            )
    assert max_interior_error <= 1e-2


# ---------------------------------------------------------------------------
# interop.rs: chaining
# ---------------------------------------------------------------------------


def linear_file(ids, epochs, time_system="GPS"):
    """A position-only file at 300 s holding ``ids`` at ``epochs``, with each
    satellite moving linearly along x."""
    sp3 = SP3(ids, 300.0, include_velocity=False, time_system=time_system)
    origin = t0()
    for epoch in epochs:
        index = sp3.push_epoch(epoch)
        for n, code in enumerate(ids):
            x = 2.0e7 + 1.0e6 * n + 1000.0 * (epoch - origin)
            sp3.set_record(code, index, SP3Record(np.array([x, 1.0e7, 1.0e7])))
    return sp3


def uniform_file(code, interval, epochs, velocity):
    """A file at ``interval`` holding ``code`` at ``epochs``, moving at 1000 m/s
    along x, with velocity records when ``velocity`` is true."""
    sp3 = SP3([code], interval, include_velocity=velocity)
    for epoch in epochs:
        index = sp3.push_epoch(epoch)
        x = 2.0e7 + 1000.0 * (epoch - gps(0, 0.0))
        v = np.array([1000.0, 0.0, 0.0]) if velocity else None
        sp3.set_record(code, index, SP3Record(np.array([x, 1.0e7, 1.0e7]), velocity=v))
    return sp3


def test_SP3Satellite_from_files_shared_boundary_keeps_later():
    a = SP3.from_file(TRUNC)
    b = SP3.from_file(AFTER_DROP)
    sats = SP3Satellite.from_files([a, b])
    c01 = find(sats, "C01")
    assert len(c01) == 3
    p_b = b.record("C01", 0).position
    assert np.array_equal(c01.positions()[1], p_b)
    assert not c01.has_gaps()

    p_a = a.record("C01", 1).position
    assert not np.array_equal(p_a, p_b)
    assert c01.epochs()[1] == b.epochs[0]
    assert len(sats) == 87


def test_SP3Satellite_from_files_order_and_duplicates():
    a = SP3.from_file(TRUNC)
    b = SP3.from_file(AFTER_DROP)
    c = SP3.from_file(AFTER_NO_DROP)
    fwd = SP3Satellite.from_files([a, b, c])
    rev = SP3Satellite.from_files([c, b, a])
    with pytest.raises(bh.BraheError, match="overlap"):
        SP3Satellite.from_files([a, a])

    assert len(fwd) == len(rev)
    for f, r in zip(fwd, rev):
        assert f.id == r.id
        assert_epoch_lists_equal(f.epochs(), r.epochs())
        assert np.array_equal(f.positions(), r.positions())


def test_SP3Satellite_from_files_overlap_names_both_files():
    a = SP3.from_file(TRUNC)
    with pytest.raises(bh.BraheError) as excinfo:
        SP3Satellite.from_files([a, a])
    err = str(excinfo.value)
    assert "overlap" in err
    assert err.count("GFZ") == 2
    assert str(a.start_epoch) in err and str(a.end_epoch) in err

    shifted = SP3(["G01"], 300.0, agency="ABC")
    shifted.push_epoch(a.start_epoch + 150.0)
    with pytest.raises(bh.BraheError) as excinfo:
        SP3Satellite.from_files([shifted, a])
    err = str(excinfo.value)
    assert "overlap" in err and "GFZ" in err and "ABC" in err


def test_SP3Satellite_from_files_large_gap():
    sats = SP3Satellite.from_files([SP3.from_file(TRUNC), SP3.from_file(LARGE_GAP)])
    g01 = find(sats, "G01")
    assert len(g01) == 4
    assert g01.has_gaps()
    assert len(g01.segments()) == 2

    with pytest.raises(bh.BraheError) as excinfo:
        g01.to_trajectory()
    assert "G01" in str(excinfo.value) and "gap" in str(excinfo.value)
    assert len(g01.to_trajectory_segments()) == 2

    inside_gap = bh.Epoch.from_datetime(2017, 5, 21, 0, 12, 0.0, 0.0, bh.TimeSystem.GPS)
    with pytest.raises(bh.BraheError) as excinfo:
        g01.state_itrf(inside_gap)
    assert "G01" in str(excinfo.value) and "gap" in str(excinfo.value)


def test_SP3Satellite_from_files_contiguous():
    sats = SP3Satellite.from_files(
        [SP3.from_file(TRUNC), SP3.from_file(AFTER_DROP), SP3.from_file(AFTER_NO_DROP)]
    )
    assert len(sats) == 87
    start = bh.Epoch.from_datetime(2017, 5, 21, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)
    expected = [start + 300.0 * k for k in range(4)]
    for sat in sats:
        assert len(sat) == 4, str(sat.id)
        assert_epoch_lists_equal(sat.epochs(), expected)
        assert not sat.has_gaps(), str(sat.id)


def test_SP3Satellite_from_files_missing_satellite_gives_gap():
    ids = ["G01", "G02"]
    first = linear_file(ids, [gps(0, 0.0), gps(5, 0.0)])
    middle = linear_file(ids[:1], [gps(10, 0.0), gps(15, 0.0)])
    last = linear_file(ids, [gps(20, 0.0), gps(25, 0.0)])
    sats = SP3Satellite.from_files([first, middle, last])
    assert len(sats) == 2

    g01 = find(sats, "G01")
    assert len(g01) == 6
    assert not g01.has_gaps()

    g02 = find(sats, "G02")
    assert len(g02) == 4
    assert g02.has_gaps()
    assert g02.segments() == [(gps(0, 0.0), gps(5, 0.0)), (gps(20, 0.0), gps(25, 0.0))]


def test_SP3Satellite_from_files_invalid_later_boundary_keeps_earlier():
    first = linear_file(["G01"], [gps(0, 0.0), gps(5, 0.0)])
    second = SP3(["G01"], 300.0, include_velocity=False)
    second.push_epoch(gps(5, 0.0))
    second.push_epoch(gps(10, 0.0))
    second.set_record("G01", 1, SP3Record(np.array([3.0e7, 1.0e7, 1.0e7])))
    expected = first.record("G01", 1).position

    g01 = find(SP3Satellite.from_files([first, second]), "G01")
    assert len(g01) == 3
    assert np.array_equal(g01.positions()[1], expected)
    assert not g01.has_gaps()


def test_SP3Satellite_from_files_ascending_id_order():
    header_order = ["C05", "R01", "G02", "E01", "G01"]
    first = linear_file(header_order, [gps(0, 0.0), gps(5, 0.0)])
    second = linear_file(["J01", "G03"], [gps(10, 0.0), gps(15, 0.0)])
    ids = [str(s.id) for s in SP3Satellite.from_files([first, second])]
    assert ids == ["G01", "G02", "G03", "R01", "E01", "C05", "J01"]
    assert [str(s.id) for s in first.satellites()] == header_order


def test_SP3Satellite_from_files_empty_inputs():
    assert SP3Satellite.from_files([]) == []
    a = SP3.from_file(TRUNC)
    no_epochs = SP3(["G01"], 300.0)
    sats = SP3Satellite.from_files([no_epochs, a, no_epochs])
    assert len(sats) == 87
    assert len(find(sats, "G01")) == 2


def test_SP3Satellite_from_files_mixed_time_systems_sort_by_instant():
    gps_file = linear_file(["G01"], [gps(0, 10.0), gps(5, 10.0), gps(10, 10.0)])

    def utc(minute):
        return bh.Epoch.from_datetime(
            2024, 11, 10, 0, minute, 0.0, 0.0, bh.TimeSystem.UTC
        )

    utc_file = linear_file(["G01"], [utc(10), utc(15)], time_system=SP3TimeSystem.UTC)
    assert utc_file.start_epoch > gps_file.end_epoch

    for files in ([gps_file, utc_file], [utc_file, gps_file]):
        g01 = find(SP3Satellite.from_files(files), "G01")
        assert len(g01) == 5
        epochs = g01.epochs()
        assert epochs[2] == gps(10, 10.0)
        assert epochs[3] == utc(10)
        assert all(b - a > 0.0 for a, b in itertools.pairwise(epochs))
        assert not g01.has_gaps()


def test_SP3_satellites_matches_from_files_single():
    a = SP3.from_file(TRUNC)
    single = a.satellites()
    chained = SP3Satellite.from_files([a])
    assert len(single) == len(chained)
    for sat in single:
        other = next(s for s in chained if s.id == sat.id)
        assert_epoch_lists_equal(sat.epochs(), other.epochs())
        assert [record_key(r) for r in sat.records()] == [
            record_key(r) for r in other.records()
        ]
        np.testing.assert_array_equal(sat.velocities(), other.velocities())
        one = a.satellite(sat.id)
        assert [record_key(r) for r in one.records()] == [
            record_key(r) for r in sat.records()
        ]


def test_SP3Satellite_from_files_equal_starts_independent_of_order():
    long = linear_file(["G01"], [gps(0, 0.0), gps(5, 0.0)])
    single = SP3(["G01"], 300.0, include_velocity=False)
    single.push_epoch(gps(0, 0.0))
    single.set_record("G01", 0, SP3Record(np.array([3.0e7, 1.0e7, 1.0e7])))
    expected = long.record("G01", 0).position

    forward = SP3Satellite.from_files([long, single])
    reverse = SP3Satellite.from_files([single, long])
    for sats in (forward, reverse):
        g01 = find(sats, "G01")
        assert_epoch_lists_equal(g01.epochs(), [gps(0, 0.0), gps(5, 0.0)])
        assert np.array_equal(g01.positions()[0], expected)
    assert [record_key(r) for r in find(forward, "G01").records()] == [
        record_key(r) for r in find(reverse, "G01").records()
    ]


def test_SP3Satellite_from_files_one_epoch_file_twice():
    single = linear_file(["G01"], [gps(0, 0.0)])
    g01 = find(SP3Satellite.from_files([single, single]), "G01")
    assert len(g01) == 1
    assert_epoch_lists_equal(g01.epochs(), [gps(0, 0.0)])


def test_SP3Satellite_from_files_velocity_source():
    first = [gps(0, 0.0), gps(5, 0.0), gps(10, 0.0)]
    second = [gps(15, 0.0), gps(20, 0.0), gps(25, 0.0)]

    both = SP3Satellite.from_files(
        [
            uniform_file("G01", 300.0, first, True),
            uniform_file("G01", 300.0, second, True),
        ]
    )
    g01 = find(both, "G01")
    assert g01.has_file_velocities()
    assert (g01.velocities() == [1000.0, 0.0, 0.0]).all()

    mixed = SP3Satellite.from_files(
        [
            uniform_file("G01", 300.0, first, True),
            uniform_file("G01", 300.0, second, False),
        ]
    )
    g01 = find(mixed, "G01")
    assert len(g01) == 6
    assert not g01.has_file_velocities()
    assert np.array_equal(g01.records()[0].velocity, [1000.0, 0.0, 0.0])
    np.testing.assert_allclose(
        g01.velocities(), np.tile([1000.0, 0.0, 0.0], (6, 1)), rtol=0.0, atol=1e-6
    )


def test_SP3Satellite_from_files_mixed_intervals_gap_threshold():
    fine = uniform_file("G01", 300.0, [gps(0, 0.0), gps(5, 0.0), gps(10, 0.0)], False)

    coarse = uniform_file(
        "G01", 900.0, [gps(25, 0.0), gps(40, 0.0), gps(55, 0.0)], False
    )
    g01 = find(SP3Satellite.from_files([fine, coarse]), "G01")
    assert len(g01) == 6
    assert not g01.has_gaps()

    late = uniform_file("G01", 900.0, [gps(35, 0.0), gps(50, 0.0)], False)
    g01 = find(SP3Satellite.from_files([fine, late]), "G01")
    assert g01.segments() == [(gps(0, 0.0), gps(10, 0.0)), (gps(35, 0.0), gps(50, 0.0))]


def keplerian_day(start, code, propagator):
    """A 24 h position-only file at 900 s starting at ``start``, sampled from
    ``propagator``."""
    sp3 = SP3([code], 900.0, include_velocity=False)
    for k in range(97):
        epoch = start + 900.0 * k
        index = sp3.push_epoch(epoch)
        sp3.set_record(code, index, SP3Record(propagator.state_itrf(epoch)[:3]))
    return sp3


def test_SP3Satellite_from_files_boundary_velocity_uses_centered_fit():
    day1 = t0()
    day2 = day1 + 86400.0
    propagator = bh.KeplerianPropagator.from_keplerian(
        day1,
        np.array([15_000e3, 0.01, 55.0, 30.0, 0.0, 45.0]),
        bh.AngleFormat.DEGREES,
        60.0,
    )
    first = keplerian_day(day1, "G01", propagator)
    second = keplerian_day(day2, "G01", propagator)
    truth = propagator.state_itrf(day2)[3:]

    single = first.satellite("G01")
    assert single.epochs()[-1] == day2
    err_single = np.linalg.norm(single.velocities()[-1] - truth)

    chained = find(SP3Satellite.from_files([first, second]), "G01")
    assert len(chained) == 193
    assert not chained.has_gaps()
    boundary = next(i for i, e in enumerate(chained.epochs()) if e == day2)
    err_chained = np.linalg.norm(chained.velocities()[boundary] - truth)

    assert err_chained < err_single
    assert err_chained < 0.1 * err_single


# ---------------------------------------------------------------------------
# interop.rs: from_propagators, set_clocks, set_clock_rates
# ---------------------------------------------------------------------------


def leo_propagator(epoch, raan):
    return bh.KeplerianPropagator.from_keplerian(
        epoch,
        np.array([bh.R_EARTH + 800e3, 0.001, 98.0, raan, 0.0, 0.0]),
        bh.AngleFormat.DEGREES,
        60.0,
    )


def two_leo_sp3(**options):
    a = leo_propagator(t0(), 0.0)
    b = leo_propagator(t0(), 40.0)
    sp3 = SP3.from_propagators(
        [("L01", a), (sid("L02"), b)], t0(), t0() + 86400.0, 30.0, **options
    )
    return sp3, a, b


def test_SP3_from_propagators_keplerian():
    sp3, a, b = two_leo_sp3()
    assert sp3.header.file_type == SP3FileType.LEO
    assert sp3.num_epochs == 2881
    assert sp3.header.epoch_interval == 30.0
    assert sp3.header.has_velocity
    assert sp3.epochs[0] == t0()
    assert sp3.start_epoch == t0()

    for code, prop in (("L01", a), ("L02", b)):
        for k in (0, 1, 1000, 2880):
            truth = prop.state_itrf(t0() + 30.0 * k)
            record = sp3.record(code, k)
            np.testing.assert_allclose(record.position, truth[:3], rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(record.velocity, truth[3:], rtol=0.0, atol=1e-9)
            assert record.clock is None
            assert record.clock_rate is None


def test_SP3_from_propagators_mixed_file_type():
    a = leo_propagator(t0(), 0.0)
    b = leo_propagator(t0(), 40.0)
    sp3 = SP3.from_propagators([("G01", a), ("L01", b)], t0(), t0() + 600.0, 30.0)
    assert sp3.header.file_type == SP3FileType.MIXED


def test_SP3_from_propagators_clocks_absent_and_round_trip():
    sp3, a, _ = two_leo_sp3()
    for k in range(sp3.num_epochs):
        for code in ("L01", "L02"):
            assert sp3.record(code, k).clock is None
    text = sp3.to_string()
    assert "999999.999999" in text

    sat = SP3.from_str(text).satellite("L01")
    assert len(sat) == 2881
    for k in (0, 7, 1440, 2880):
        epoch = t0() + 30.0 * k
        np.testing.assert_allclose(
            sat.state_itrf(epoch)[:3], a.state_itrf(epoch)[:3], rtol=0.0, atol=5.1e-4
        )


def test_SP3_from_propagators_grid_includes_end_within_tolerance():
    providers = [("L01", leo_propagator(t0(), 0.0))]
    assert SP3.from_propagators(providers, t0(), t0() + 100.0, 30.0).num_epochs == 4
    assert SP3.from_propagators(providers, t0(), t0() + 90.0, 30.0).num_epochs == 4
    assert SP3.from_propagators(providers, t0(), t0(), 30.0).num_epochs == 1

    with pytest.raises(bh.BraheError, match="precedes"):
        SP3.from_propagators(providers, t0(), t0() - 30.0, 30.0)
    with pytest.raises(bh.BraheError):
        SP3.from_propagators(providers, t0(), t0() + 60.0, 0.0)


def test_SP3_from_propagators_position_only_rejects_clock_rates():
    sp3, _, _ = two_leo_sp3(include_velocity=False)
    assert not sp3.header.has_velocity
    assert sp3.record("L01", 3).velocity is None
    assert "\nV" not in sp3.to_string()

    with pytest.raises(bh.BraheError, match="velocity"):
        sp3.set_clock_rates("L01", [1e-12] * sp3.num_epochs)


def test_SP3_set_clocks_and_rates():
    sp3, _, _ = two_leo_sp3()
    n = sp3.num_epochs
    clocks = [None if k == 5 else 1e-5 + 1e-9 * k for k in range(n)]
    rates = [1e-12 * k for k in range(n)]
    sp3.set_clocks("L01", clocks)
    sp3.set_clock_rates(sid("L01"), rates)

    for k in (0, 5, 6, n - 1):
        record = sp3.record("L01", k)
        assert record.clock == clocks[k]
        assert record.clock_rate == rates[k]
    assert sp3.record("L02", 6).clock is None

    parsed = SP3.from_str(sp3.to_string())
    record = parsed.record("L01", 6)
    assert record.clock == pytest.approx(clocks[6], abs=1e-12)
    assert record.clock_rate == pytest.approx(rates[6], abs=1e-16)
    assert parsed.record("L01", 5).clock is None


def test_SP3_set_clocks_nan_is_none():
    sp3, _, _ = two_leo_sp3()
    n = sp3.num_epochs
    clocks = [math.nan if k == 5 else 1e-5 for k in range(n)]
    sp3.set_clocks("L01", clocks)
    rates = np.full(n, 1e-12)
    rates[6] = np.nan
    sp3.set_clock_rates("L01", rates)
    assert sp3.record("L01", 5).clock is None
    assert sp3.record("L01", 6).clock_rate is None
    assert sp3.record("L01", 6).clock == 1e-5
    assert sp3.record("L01", 5).clock_rate == 1e-12
    parsed = SP3.from_str(sp3.to_string())
    assert parsed.record("L01", 5).clock is None
    assert parsed.record("L01", 6).clock_rate is None

    sat = sp3.satellite("L01")
    sp3.set_clocks("L02", sat.clocks())
    sp3.set_clock_rates("L02", sat.clock_rates())
    assert sp3.record("L02", 5).clock is None
    assert sp3.record("L02", 6).clock == 1e-5
    assert sp3.record("L02", 6).clock_rate is None
    sp3.to_string()


def test_SP3_set_clocks_errors():
    sp3, _, _ = two_leo_sp3()
    n = sp3.num_epochs

    with pytest.raises(bh.BraheError, match="epochs"):
        sp3.set_clocks("L01", [1e-5] * (n - 1))
    with pytest.raises(bh.BraheError):
        sp3.set_clock_rates("L01", [1e-12] * (n + 1))
    with pytest.raises(bh.BraheError):
        sp3.set_clocks("G05", [None] * n)

    sparse = SP3(["G01"], 300.0)
    sparse.push_epoch(gps(0, 0.0))
    sparse.push_epoch(gps(5, 0.0))
    sparse.set_record("G01", 0, SP3Record(np.array([2.0e7, 1.0e7, 1.0e7])))
    with pytest.raises(bh.BraheError) as excinfo:
        sparse.set_clocks("G01", [1e-5, 1e-5])
    assert "G01" in str(excinfo.value) and "no record" in str(excinfo.value)
    assert sparse.record("G01", 0).clock is None

    sparse.set_clocks("G01", [1e-5, None])
    assert sparse.record("G01", 0).clock == 1e-5
    with pytest.raises(bh.BraheError):
        sparse.set_clock_rates("G01", [None, 1e-12])


def test_SP3_from_propagators_trajectory_outside_span_errors():
    prop = leo_propagator(t0(), 0.0)
    traj = bh.OrbitTrajectory(
        6, bh.CelestialFrame.ITRF, bh.OrbitRepresentation.CARTESIAN, None
    )
    for k in range(721):
        epoch = t0() + 60.0 * k
        traj.add(epoch, prop.state_itrf(epoch))

    covered = SP3.from_propagators([("L01", traj)], t0(), t0() + 43200.0, 60.0)
    assert covered.num_epochs == 721

    with pytest.raises(bh.BraheError) as excinfo:
        SP3.from_propagators([("L01", traj)], t0(), t0() + 86400.0, 60.0)
    err = str(excinfo.value)
    assert "L01" in err and str(t0() + 43260.0) in err


def test_SP3_from_propagators_non_finite_state_errors():
    prop = leo_propagator(t0(), 0.0)
    traj = bh.OrbitTrajectory(
        6, bh.CelestialFrame.ITRF, bh.OrbitRepresentation.CARTESIAN, None
    )
    traj.add(t0(), prop.state_itrf(t0()))
    traj.add(t0() + 60.0, np.full(6, np.nan))

    with pytest.raises(bh.BraheError) as excinfo:
        SP3.from_propagators([("L01", traj)], t0(), t0() + 60.0, 60.0)
    err = str(excinfo.value)
    assert "L01" in err and "non-finite" in err and str(t0() + 60.0) in err


def test_SP3_from_propagators_resamples_sp3_satellite():
    sp3, _, _ = two_leo_sp3()
    sat = sp3.satellite("L02")
    resampled = SP3.from_propagators(
        [("L02", sat)], t0() + 3600.0, t0() + 7200.0, 45.0, agency="DEMO"
    )
    assert resampled.num_epochs == 81
    assert resampled.header.agency == "DEMO"
    assert resampled.header.epoch_interval == 45.0
    for k in (0, 40, 80):
        truth = sat.state_itrf(t0() + 3600.0 + 45.0 * k)
        record = resampled.record("L02", k)
        np.testing.assert_allclose(record.position, truth[:3], rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(record.velocity, truth[3:], rtol=0.0, atol=1e-9)


def test_SP3_from_propagators_numerical_propagator():
    start = bh.Epoch.from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    prop = bh.NumericalOrbitPropagator(
        start,
        np.array([bh.R_EARTH + 700e3, 0.0, 0.0, 0.0, 6000.0, 4500.0]),
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.earth_gravity(),
        None,
    )
    prop.propagate_to(start + 7200.0)

    sp3 = SP3.from_propagators([("L05", prop)], start, start + 7200.0, 60.0)
    assert sp3.num_epochs == 121
    for k in (0, 1, 60, 120):
        truth = prop.state_itrf(start + 60.0 * k)
        record = sp3.record("L05", k)
        np.testing.assert_allclose(record.position, truth[:3], rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(record.velocity, truth[3:], rtol=0.0, atol=1e-9)


def test_sp3_from_propagators_mixed_types():
    kep_prop = leo_propagator(t0(), 0.0)
    traj = bh.OrbitTrajectory(
        6, bh.CelestialFrame.ITRF, bh.OrbitRepresentation.CARTESIAN, None
    )
    for k in range(121):
        epoch = t0() + 30.0 * k
        traj.add(epoch, leo_propagator(t0(), 40.0).state_itrf(epoch))
    sgp = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2, 60.0)
    start, end = t0(), t0() + 3600.0

    sp3 = SP3.from_propagators(
        [("L01", kep_prop), ("L02", traj), (sid("L03"), sgp)],
        start,
        end,
        30.0,
        agency="DEMO",
    )
    assert sp3.header.agency == "DEMO"
    assert np.isnan(sp3.satellite("L01").clocks()).all()
    for code, prop in (("L01", kep_prop), ("L02", traj), ("L03", sgp)):
        truth = prop.state_itrf(start + 30.0 * 60)
        record = sp3.record(code, 60)
        np.testing.assert_allclose(record.position, truth[:3], rtol=0.0, atol=1e-9)


def test_SP3_from_propagators_header_options():
    sp3 = SP3.from_propagators(
        [("L01", leo_propagator(t0(), 0.0))],
        t0(),
        t0() + 300.0,
        60.0,
        time_system="UTC",
        coordinate_system="ITRF",
        orbit_type="FIT",
        data_used="SLR",
        comments=["generated in a test"],
    )
    header = sp3.header
    assert header.time_system == SP3TimeSystem.UTC
    assert header.coordinate_system == "ITRF"
    assert str(header.orbit_type) == "FIT"
    assert header.data_used == "SLR"
    assert header.comments == ["generated in a test"]
    assert header.agency == "BRHE"


def test_SP3_from_propagators_rejects_invalid_items():
    prop = leo_propagator(t0(), 0.0)
    start, end = t0(), t0() + 300.0
    with pytest.raises(TypeError, match="KeplerianPropagator") as excinfo:
        SP3.from_propagators([("L01", "not a provider")], start, end, 60.0)
    assert "SP3Satellite" in str(excinfo.value)
    with pytest.raises(TypeError, match="tuple"):
        SP3.from_propagators([prop], start, end, 60.0)
    with pytest.raises(TypeError, match="tuple"):
        SP3.from_propagators([("L01", prop, 1)], start, end, 60.0)
    with pytest.raises(bh.BraheError):
        SP3.from_propagators([("X01", prop)], start, end, 60.0)
