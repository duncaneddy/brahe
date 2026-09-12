"""Tests for the Modified ITC format — parity with src/itc/*.rs."""

import numpy as np
import pytest

import brahe as bh
from brahe.itc import (
    ITC,
    ITCCovarianceFrame,
    ITCHeader,
    ITCStateVector,
)
from brahe.spacetrack import SpaceTrackEphemerisFileCategory

FULL = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
TRUNCATED = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt"


def utc(y, mo, d, h, mi, s):
    return bh.Epoch(y, mo, d, h, mi, s, 0.0)


def state(sec):
    return ITCStateVector(
        utc(2026, 9, 11, 1, 42, 42.0) + sec,
        np.array([7.0e6, 0.0, 0.0]),
        np.array([0.0, 7.5e3, 0.0]),
    )


def test_itc_covariance_frame_parse_and_token():
    for token in ["UVW", "uvw", "RTN", "RSW", "RIC"]:
        assert ITCCovarianceFrame.parse(token) == ITCCovarianceFrame.RTN
    for token in ["EME2000", "J2000", "j2000"]:
        assert ITCCovarianceFrame.parse(token) == ITCCovarianceFrame.EME2000
    assert ITCCovarianceFrame.parse("ITRF") == ITCCovarianceFrame.ITRF
    with pytest.raises(bh.BraheError):
        ITCCovarianceFrame.parse("TEME")
    assert str(ITCCovarianceFrame.RTN) == "UVW"
    assert ITCCovarianceFrame.EME2000.token() == "EME2000"


def test_itc_header_defaults_and_kwargs():
    header = ITCHeader()
    assert header.state_frame == bh.CelestialFrame.EME2000
    assert header.covariance_frame == ITCCovarianceFrame.RTN
    assert (
        header.created is None
        and header.step_size is None
        and header.ephemeris_source is None
    )
    header = ITCHeader(
        created=utc(2026, 9, 11, 1, 55, 52.0),
        ephemeris_source="blend",
        state_frame=bh.CelestialFrame.TEME,
        covariance_frame=ITCCovarianceFrame.EME2000,
    )
    assert header.created == utc(2026, 9, 11, 1, 55, 52.0)
    assert header.ephemeris_source == "blend"
    assert header.state_frame == bh.CelestialFrame.TEME
    assert header.covariance_frame == ITCCovarianceFrame.EME2000
    header.step_size = 30.0
    assert header.step_size == 30.0


def test_itc_push_state_and_accessors():
    itc = ITC(ITCHeader())
    assert len(itc) == 0 and itc.start_epoch is None
    itc.push_state(state(0.0))
    itc.push_state(state(60.0))
    assert len(itc) == 2
    assert not itc.has_covariance
    assert itc.start_epoch == utc(2026, 9, 11, 1, 42, 42.0)
    assert itc.end_epoch == utc(2026, 9, 11, 1, 43, 42.0)
    with pytest.raises(bh.BraheError):
        itc.push_state(state(60.0))
    with pytest.raises(bh.BraheError):
        itc.push_state(state(0.0))


def test_itc_covariance_all_or_none():
    cov = np.eye(6)
    with_cov = ITC(ITCHeader())
    with_cov.push_state_with_covariance(state(0.0), cov)
    assert with_cov.has_covariance
    with pytest.raises(bh.BraheError):
        with_cov.push_state(state(60.0))
    with_cov.push_state_with_covariance(state(60.0), cov)
    assert len(with_cov.covariances) == 2
    without = ITC(ITCHeader())
    without.push_state(state(0.0))
    with pytest.raises(bh.BraheError):
        without.push_state_with_covariance(state(60.0), cov)
    with pytest.raises(ValueError):
        ITC(ITCHeader()).push_state_with_covariance(state(0.0), np.eye(3))


def test_itc_state_vector_to_vector():
    sv = ITCStateVector(
        utc(2026, 9, 11, 1, 42, 42.0),
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    )
    x = sv.to_vector()
    assert x.shape == (6,)
    np.testing.assert_array_equal(x[:3], sv.position)
    np.testing.assert_array_equal(x[3:], sv.velocity)


def test_itc_push_state_with_covariance_rejects_asymmetric():
    asymmetric = np.eye(6)
    asymmetric[0, 1] = 1.0
    asymmetric[1, 0] = 2.0
    with pytest.raises(bh.BraheError):
        ITC(ITCHeader()).push_state_with_covariance(state(0.0), asymmetric)


def test_parse_full_asset_header():
    itc = ITC.from_file(FULL)
    h = itc.header
    assert h.created == utc(2026, 9, 11, 1, 55, 52.0)
    assert h.ephemeris_start == utc(2026, 9, 11, 1, 42, 42.0)
    assert h.ephemeris_stop == utc(2026, 9, 14, 1, 42, 42.0)
    assert h.step_size == 60.0
    assert h.ephemeris_source == "blend"
    assert h.covariance_frame == ITCCovarianceFrame.RTN
    assert h.state_frame == bh.CelestialFrame.EME2000
    assert itc.source_name.norad_cat_id == 100001
    assert itc.source_name.object_name == "STARLINK-38128"


def test_parse_full_asset_records():
    itc = ITC.from_file(FULL)
    assert len(itc) == 4321
    states = itc.states
    covs = itc.covariances
    assert len(covs) == 4321
    first = states[0]
    assert first.epoch == utc(2026, 9, 11, 1, 42, 42.0)
    np.testing.assert_allclose(
        first.position,
        np.array([4244.3465367594, 1264.3254891872, 5043.9826441325]) * 1e3,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        first.velocity,
        np.array([3.5951547629, 5.2587956583, -4.3350352914]) * 1e3,
        atol=1e-9,
    )
    c = covs[0]
    assert c.shape == (6, 6)
    assert c[0, 0] == pytest.approx(4.6343390768e-07 * 1e6, abs=1e-12)
    assert c[1, 0] == pytest.approx(-3.7963809271e-07 * 1e6, abs=1e-12)
    assert c[3, 0] == pytest.approx(8.2617476650e-10 * 1e6, abs=1e-15)
    assert c[5, 5] == pytest.approx(5.4287251909e-12 * 1e6, abs=1e-17)
    np.testing.assert_array_equal(c, c.T)
    assert states[-1].epoch == utc(2026, 9, 14, 1, 42, 42.0)
    np.testing.assert_allclose(
        states[-1].position,
        np.array([1818.7096808241, 3525.7372127755, -5423.9171357013]) * 1e3,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        states[-1].velocity,
        np.array([-6.2948457440, -2.4380764122, -3.6949737258]) * 1e3,
        atol=1e-9,
    )
    assert states[1].epoch - states[0].epoch == pytest.approx(60.0, abs=1e-6)


def test_parse_truncated_and_from_str():
    itc = ITC.from_file(TRUNCATED)
    assert len(itc) == 50 and itc.has_covariance
    assert itc.header.created == utc(2026, 9, 11, 2, 4, 18.0)
    assert itc.source_name.norad_cat_id == 100002
    with open(TRUNCATED) as f:
        text = f.read()
    parsed = ITC.from_str(text)
    assert parsed.source_name is None
    assert parsed.header.state_frame == bh.CelestialFrame.EME2000
    assert len(parsed) == 50


HEADER = (
    "created:2026-09-11 01:55:52 UTC\n"
    "ephemeris_start:2026-09-11 01:42:42 UTC ephemeris_stop:2026-09-14 01:42:42 UTC step_size:60\n"
    "ephemeris_source:blend\n"
    "UVW\n"
)
REC0 = "2026254014242.000 4244.3465367594 1264.3254891872 5043.9826441325 3.5951547629 5.2587956583 -4.3350352914\n"
COV0 = (
    "4.6343390768e-07 -3.7963809271e-07 7.7770281867e-07 1.8398914684e-10 2.3515746188e-10 1.2050744950e-06 8.2617476650e-10\n"
    "-9.3402950908e-10 6.5056423006e-13 2.0019807553e-12 -4.6727451721e-10 4.1362166473e-10 -1.2117302280e-12 -8.4570284181e-13\n"
    "5.1950545462e-13 2.6473144477e-13 8.7817244924e-13 1.6102297802e-09 -2.1049889128e-16 -1.5343530772e-15 5.4287251909e-12\n"
)
REC1 = "2026254014342.000 4449.8462744669 1576.6138914245 4772.1198724309 3.2521129034 5.1467007555 -4.7234817117\n"


def test_parse_without_covariance_and_unrecognized_header():
    itc = ITC.from_str(
        "created:\nephemeris_start: ephemeris_stop: step_size:60\nephemeris_source:test\nUVW\n"
        + REC0
        + REC1
    )
    assert len(itc) == 2 and not itc.has_covariance
    assert itc.header.created is None and itc.header.step_size == 60.0
    itc = ITC.from_str("Generated by a tool\nSome note\nA third\nJ2000\n" + REC0)
    assert itc.header.created is None and itc.header.ephemeris_source is None
    assert itc.header.covariance_frame == ITCCovarianceFrame.EME2000


@pytest.mark.parametrize(
    "content",
    [
        "created:\nephemeris_source:x\nUVW\n",
        HEADER,
        HEADER + REC0.replace(" -4.3350352914", ""),
        HEADER + REC0 + COV0[:-20],
        HEADER + REC0 + COV0 + REC1,
        HEADER + REC1 + REC0,
        HEADER + COV0 + REC0,
        HEADER.replace("step_size:60", "step_size:abc") + REC0,
        HEADER.replace("step_size:60", "step_size:NaN") + REC0,
        HEADER.replace("step_size:60", "step_size:inf") + REC0,
        HEADER.replace("created:2026-09-11 01:55:52 UTC", "created:yesterday") + REC0,
        HEADER.replace("\nUVW\n", "\nTEME\n") + REC0,
    ],
)
def test_parse_errors(content):
    with pytest.raises(bh.BraheError):
        ITC.from_str(content)


def test_parse_missing_file_is_error():
    with pytest.raises(bh.BraheError):
        ITC.from_file("test_assets/starlink/does_not_exist.txt")


def test_parse_covariance_lower_triangle_order():
    c = ITC.from_str(HEADER + REC0 + COV0).covariances[0]
    assert c[3, 1] == pytest.approx(-9.3402950908e-10 * 1e6, abs=1e-16)
    assert c[4, 4] == pytest.approx(5.1950545462e-13 * 1e6, abs=1e-18)
    assert c[5, 0] == pytest.approx(2.6473144477e-13 * 1e6, abs=1e-18)


@pytest.mark.parametrize("path", [TRUNCATED, FULL])
def test_write_matches_source_text(path):
    with open(path) as f:
        original = f.read()
    assert ITC.from_file(path).to_string().rstrip() == original.rstrip()


def test_write_round_trip_and_derived_header(tmp_path):
    itc = ITC.from_file(TRUNCATED)
    again = ITC.from_str(itc.to_string())
    assert [s.epoch for s in again.states] == [s.epoch for s in itc.states]
    np.testing.assert_array_equal(again.covariances[0], itc.covariances[0])

    built = ITC(ITCHeader(ephemeris_source="unit"))
    for i in range(3):
        built.push_state(
            ITCStateVector(
                utc(2026, 9, 11, 1, 42, 42.0) + 30.0 * i,
                np.array([7.0e6, 1.0e3, -2.0e3]),
                np.array([1.0, 7.5e3, -3.0]),
            )
        )
    lines = built.to_string().splitlines()
    assert lines[0] == "created:"
    assert (
        lines[1]
        == "ephemeris_start:2026-09-11 01:42:42 UTC ephemeris_stop:2026-09-11 01:43:42 UTC step_size:30"
    )
    assert lines[2] == "ephemeris_source:unit"
    assert lines[3] == "UVW"
    assert (
        lines[4]
        == "2026254014242.000 7000.0000000000 1.0000000000 -2.0000000000 0.0010000000 7.5000000000 -0.0030000000"
    )
    assert len(lines) == 7

    with pytest.raises(bh.BraheError):
        ITC(ITCHeader()).to_string()
    out = tmp_path / "out.txt"
    itc.to_file(str(out))
    reread = ITC.from_file(str(out))
    assert len(reread) == 50 and reread.source_name is None


def test_write_fractional_step_and_other_frames():
    itc = ITC(ITCHeader(covariance_frame=ITCCovarianceFrame.ITRF))
    for sec in (0.0, 2.5):
        itc.push_state_with_covariance(
            ITCStateVector(
                utc(2026, 1, 1, 0, 0, sec),
                np.array([7.0e6, 0.0, 0.0]),
                np.array([0.0, 7.5e3, 0.0]),
            ),
            np.eye(6) * 1.0e6,
        )
    lines = itc.to_string().splitlines()
    assert lines[1].endswith("step_size:2.5")
    assert lines[3] == "ITRF"
    assert (
        lines[5]
        == "1.0000000000e+00 0.0000000000e+00 1.0000000000e+00 0.0000000000e+00 0.0000000000e+00 1.0000000000e+00 0.0000000000e+00"
    )
    assert len(lines) == 12


def test_from_file_infers_frame_from_data_type(tmp_path):
    with open(TRUNCATED) as f:
        text = f.read()
    teme = tmp_path / "TEME_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt"
    teme.write_text(text)
    itc = ITC.from_file(str(teme))
    assert itc.header.state_frame == bh.CelestialFrame.TEME
    assert itc.source_name.data_type == "TEME"
    itrf = tmp_path / "ITRF_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt"
    itrf.write_text(text)
    assert ITC.from_file(str(itrf)).header.state_frame == bh.CelestialFrame.ITRF
    unknown = (
        tmp_path / "GCRF_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt"
    )
    unknown.write_text(text)
    with pytest.raises(bh.BraheError):
        ITC.from_file(str(unknown))
    plain = tmp_path / "ephemeris.txt"
    plain.write_text(text)
    itc = ITC.from_file(str(plain))
    assert (
        itc.header.state_frame == bh.CelestialFrame.EME2000 and itc.source_name is None
    )


def test_itc_file_name():
    itc = ITC.from_file(TRUNCATED)
    name = itc.file_name(
        100002,
        "STARLINK-37711",
        SpaceTrackEphemerisFileCategory.OPERATIONAL,
        "1473385800",
    )
    assert (
        str(name)
        == "MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt"
    )
    teme = ITC(ITCHeader(state_frame=bh.CelestialFrame.TEME))
    teme.push_state(
        ITCStateVector(
            utc(2020, 10, 26, 12, 24, 0.0),
            np.array([7.0e6, 0.0, 0.0]),
            np.array([0.0, 7.5e3, 0.0]),
        )
    )
    assert (
        str(teme.file_name(25544, "ISS", SpaceTrackEphemerisFileCategory.SPECIAL, ""))
        == "TEME_25544_ISS_3001224_Special__UNCLASSIFIED.txt"
    )
    with pytest.raises(bh.BraheError):
        ITC(ITCHeader()).file_name(
            1, "A", SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
        )
    with pytest.raises(bh.BraheError):
        ITC(ITCHeader(state_frame=bh.CelestialFrame.GCRF)).file_name(
            1, "A", SpaceTrackEphemerisFileCategory.OPERATIONAL, ""
        )


def test_top_level_exports():
    assert bh.ITC is ITC
    assert bh.ITCHeader is ITCHeader
    assert bh.ITCStateVector is ITCStateVector
    assert bh.ITCCovarianceFrame is ITCCovarianceFrame
    assert bh.SpaceTrackEphemerisFileName is bh.spacetrack.SpaceTrackEphemerisFileName
    assert (
        bh.SpaceTrackEphemerisFileCategory
        is bh.spacetrack.SpaceTrackEphemerisFileCategory
    )


def _rtn_to_eme_block(x):
    r = x[:3]
    v = x[3:]
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v)
    n_hat = n_hat / np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    R = np.column_stack([r_hat, t_hat, n_hat])
    J = np.zeros((6, 6))
    J[:3, :3] = R
    J[3:, 3:] = R
    return J


def _assert_covariance_close(actual, expected, rel):
    """Compare covariances with each tolerance scaled by the Cauchy-Schwarz
    bound ``sqrt(P_ii P_kk)`` rather than by the element itself. Off-diagonal
    covariance entries are differences of much larger products, so an
    element-relative tolerance measures the cancellation in the input data
    rather than the accuracy of the transformation."""
    d = np.sqrt(np.diag(expected))
    scale = np.outer(d, d)
    np.testing.assert_array_less(np.abs(actual - expected), rel * scale + 1e-300)


def _strip_covariance(text):
    lines = text.splitlines()
    keep = lines[:4] + [
        l
        for l in lines[4:]
        if l.split()[0][:13].isdigit() and len(l.split()[0].split(".")[0]) == 13
    ]
    return "\n".join(keep)


def test_to_trajectory_full_asset():
    itc = ITC.from_file(FULL)
    traj = itc.to_trajectory()
    assert traj.frame == bh.CelestialFrame.EME2000
    assert len(traj) == 4321
    assert traj.get_name() == "STARLINK-38128"
    states = itc.states
    x0 = np.concatenate([states[0].position, states[0].velocity])
    J = _rtn_to_eme_block(x0)
    expected = J @ itc.covariances[0] @ J.T
    cov0 = traj.covariance(states[0].epoch)
    np.testing.assert_allclose(cov0, expected, rtol=1e-9, atol=1e-20)
    assert np.trace(cov0[:3, :3]) == pytest.approx(
        np.trace(itc.covariances[0][:3, :3]), rel=1e-9
    )
    x_mid = traj.interpolate(states[0].epoch + 30.0)
    assert 6.5e6 < np.linalg.norm(x_mid[:3]) < 7.5e6


def test_to_trajectory_to_eci_applies_frame_bias():
    traj = ITC.from_file(TRUNCATED).to_trajectory()
    eci = traj.to_eci()
    a = traj.interpolate(traj.start_epoch())[:3]
    b = eci.interpolate(eci.start_epoch())[:3]
    d = np.linalg.norm(a - b)
    assert 0.1 < d < 2.0
    R = bh.rotation_eme2000_to_gcrf()
    np.testing.assert_allclose(b, R @ a, rtol=0, atol=1e-6)


def test_to_trajectory_rotating_variant_changes_only_velocity_blocks():
    itc = ITC.from_file(TRUNCATED)
    e = itc.states[0].epoch
    a = itc.to_trajectory().covariance(e)
    b = itc.to_trajectory_with_covariance_variant(
        bh.OrbitRelativeFrameVariant.ROTATING
    ).covariance(e)
    c = itc.to_trajectory(
        covariance_variant=bh.OrbitRelativeFrameVariant.ROTATING
    ).covariance(e)
    np.testing.assert_allclose(a[:3, :3], b[:3, :3], rtol=1e-9, atol=1e-20)
    np.testing.assert_array_equal(b, c)
    assert not np.allclose(a[3:, :], b[3:, :], rtol=1e-12, atol=0.0)


def test_to_trajectory_without_covariance():
    with open(TRUNCATED) as f:
        text = f.read()
    itc = ITC.from_str(_strip_covariance(text))
    assert not itc.has_covariance
    traj = itc.to_trajectory()
    assert len(traj) == 50
    with pytest.raises(bh.BraheError):
        traj.covariance(itc.states[0].epoch)
    assert traj.get_name() is None


def test_to_trajectory_accepts_every_frame_combination():
    itc = ITC.from_file(TRUNCATED)
    for state_frame in [
        bh.CelestialFrame.EME2000,
        bh.CelestialFrame.GCRF,
        bh.CelestialFrame.TEME,
        bh.CelestialFrame.ITRF,
    ]:
        for covariance_frame in [
            ITCCovarianceFrame.RTN,
            ITCCovarianceFrame.EME2000,
            ITCCovarianceFrame.ITRF,
        ]:
            h = itc.header
            h.state_frame = state_frame
            h.covariance_frame = covariance_frame
            itc.header = h
            traj = itc.to_trajectory()
            assert traj.frame == state_frame
            cov = traj.covariance(itc.states[0].epoch)
            assert cov[0, 0] > 0.0
            identity_pair = (
                state_frame == bh.CelestialFrame.ITRF
                and covariance_frame == bh.ITCCovarianceFrame.ITRF
            ) or (
                state_frame == bh.CelestialFrame.EME2000
                and covariance_frame == bh.ITCCovarianceFrame.EME2000
            )
            if identity_pair:
                np.testing.assert_array_equal(cov, itc.covariances[0])


def test_to_trajectory_empty_message_is_error():
    with pytest.raises(bh.BraheError):
        ITC(ITCHeader()).to_trajectory()


def test_from_trajectory_round_trip_and_errors():
    itc = ITC.from_file(TRUNCATED)
    traj = itc.to_trajectory()
    back = ITC.from_trajectory(traj, ITCHeader(ephemeris_source="round-trip"))
    assert len(back) == 50
    assert back.header.ephemeris_source == "round-trip"
    assert back.header.ephemeris_start == itc.states[0].epoch
    assert back.header.step_size == pytest.approx(60.0, abs=1e-9)
    for a, b in zip(itc.states, back.states):
        assert a.epoch == b.epoch
        np.testing.assert_allclose(a.position, b.position, rtol=0, atol=1e-6)
        np.testing.assert_allclose(a.velocity, b.velocity, rtol=0, atol=1e-9)
    assert len(back.covariances) == len(itc.covariances) == 50
    for a, b in zip(itc.covariances, back.covariances):
        _assert_covariance_close(b, a, 1e-11)

    empty = bh.OrbitTrajectory(
        6, bh.CelestialFrame.EME2000, bh.OrbitRepresentation.CARTESIAN
    )
    with pytest.raises(bh.BraheError):
        ITC.from_trajectory(empty, ITCHeader())

    kep = traj.to_keplerian(bh.AngleFormat.DEGREES)
    with pytest.raises(bh.BraheError):
        ITC.from_trajectory(kep, ITCHeader())

    seven = bh.OrbitTrajectory(
        7, bh.CelestialFrame.EME2000, bh.OrbitRepresentation.CARTESIAN
    )
    seven.add(itc.states[0].epoch, np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0, 1.0]))
    with pytest.raises(bh.BraheError):
        ITC.from_trajectory(seven, ITCHeader())


def test_eme2000_covariance_frame_keeps_inertial_covariance():
    itc = ITC.from_file(TRUNCATED)
    traj = itc.to_trajectory()
    back = ITC.from_trajectory(
        traj, ITCHeader(covariance_frame=ITCCovarianceFrame.EME2000)
    )
    assert back.header.covariance_frame == ITCCovarianceFrame.EME2000
    expected = traj.covariance(itc.states[0].epoch)
    np.testing.assert_allclose(back.covariances[0], expected, rtol=1e-12, atol=1e-20)
    forward = back.to_trajectory()
    np.testing.assert_allclose(
        forward.covariance(itc.states[0].epoch), expected, rtol=1e-12, atol=1e-20
    )


def test_from_trajectory_rtn_round_trips_for_both_variants():
    itc = ITC.from_file(TRUNCATED)
    for variant in [
        bh.OrbitRelativeFrameVariant.INERTIAL,
        bh.OrbitRelativeFrameVariant.ROTATING,
    ]:
        traj = itc.to_trajectory_with_covariance_variant(variant)
        back = ITC.from_trajectory_with_covariance_variant(traj, ITCHeader(), variant)
        assert back.header.covariance_frame == ITCCovarianceFrame.RTN
        for a, b in zip(itc.covariances, back.covariances):
            _assert_covariance_close(b, a, 1e-11)


def test_itrf_covariance_round_trips_through_an_eme2000_trajectory():
    # Mirrors src/itc/interop.rs
    # test_itrf_covariance_round_trips_through_an_eme2000_trajectory.
    # Relabelling the fixture's covariance as ITRF changes what the numbers
    # mean, but the pair of rotations must still invert one another.
    itc = ITC.from_file(TRUNCATED)
    h = itc.header
    h.covariance_frame = ITCCovarianceFrame.ITRF
    itc.header = h
    traj = itc.to_trajectory()
    assert traj.frame == bh.CelestialFrame.EME2000
    cov0 = traj.covariance(itc.states[0].epoch)
    assert abs(cov0[0, 0] - itc.covariances[0][0, 0]) > 1e-6

    back = ITC.from_trajectory(
        traj, ITCHeader(covariance_frame=ITCCovarianceFrame.ITRF)
    )
    assert back.header.covariance_frame == ITCCovarianceFrame.ITRF
    for a, b in zip(itc.covariances, back.covariances):
        _assert_covariance_close(b, a, 1e-11)


def test_eme2000_covariance_round_trips_through_a_teme_trajectory():
    # Mirrors src/itc/interop.rs
    # test_eme2000_covariance_round_trips_through_a_teme_trajectory.
    itc = ITC.from_file(TRUNCATED)
    h = itc.header
    h.state_frame = bh.CelestialFrame.TEME
    h.covariance_frame = ITCCovarianceFrame.EME2000
    itc.header = h
    traj = itc.to_trajectory()
    assert traj.frame == bh.CelestialFrame.TEME

    back = ITC.from_trajectory(
        traj,
        ITCHeader(
            state_frame=bh.CelestialFrame.TEME,
            covariance_frame=ITCCovarianceFrame.EME2000,
        ),
    )
    for a, b in zip(itc.covariances, back.covariances):
        _assert_covariance_close(b, a, 1e-11)


def test_from_trajectory_teme_state_frame_carries_covariance():
    # Mirrors src/itc/interop.rs
    # test_from_trajectory_teme_state_frame_carries_covariance.
    itc = ITC.from_file(TRUNCATED)
    traj = itc.to_trajectory()
    teme = ITC.from_trajectory(traj, ITCHeader(state_frame=bh.CelestialFrame.TEME))
    assert teme.header.state_frame == bh.CelestialFrame.TEME
    assert teme.has_covariance
    for a, b in zip(itc.covariances, teme.covariances):
        _assert_covariance_close(b, a, 1e-11)


def test_from_trajectory_gcrf_covariance_rotates_through_bias():
    itc = ITC.from_file(TRUNCATED)
    eme = itc.to_trajectory()
    R = bh.rotation_eme2000_to_gcrf()
    R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    epochs = eme.epochs()
    states = eme.states()
    covs = np.array([eme.covariance(e) for e in epochs])
    gcrf_states = (R6 @ states.T).T
    gcrf_covs = np.array([R6 @ p @ R6.T for p in covs])
    gcrf = bh.OrbitTrajectory.from_orbital_data(
        epochs,
        gcrf_states,
        bh.CelestialFrame.GCRF,
        bh.OrbitRepresentation.CARTESIAN,
        None,
        covariances=gcrf_covs,
    )
    back = ITC.from_trajectory(gcrf, ITCHeader())
    for a, b in zip(itc.states, back.states):
        np.testing.assert_allclose(a.position, b.position, rtol=0, atol=1e-5)
        np.testing.assert_allclose(a.velocity, b.velocity, rtol=0, atol=1e-8)
    assert len(back.covariances) == len(itc.covariances) == 50
    for a, b in zip(itc.covariances, back.covariances):
        _assert_covariance_close(b, a, 1e-10)


def test_to_trajectory_teme_without_covariance_is_allowed():
    with open(TRUNCATED) as f:
        text = f.read()
    itc = ITC.from_str(_strip_covariance(text))
    h = itc.header
    h.state_frame = bh.CelestialFrame.TEME
    itc.header = h
    traj = itc.to_trajectory()
    assert traj.frame == bh.CelestialFrame.TEME


def test_from_trajectory_without_covariance_and_frame_conversion():
    with open(TRUNCATED) as f:
        text = f.read()
    itc = ITC.from_str(_strip_covariance(text))
    traj = itc.to_trajectory()
    eci = traj.to_eci()
    back = ITC.from_trajectory(eci, ITCHeader())
    assert not back.has_covariance
    for a, b in zip(itc.states, back.states):
        np.testing.assert_allclose(a.position, b.position, rtol=0, atol=1e-5)

    teme = ITC.from_trajectory(eci, ITCHeader(state_frame=bh.CelestialFrame.TEME))
    assert teme.header.state_frame == bh.CelestialFrame.TEME
    d = abs(teme.states[0].position[0] - itc.states[0].position[0])
    assert d > 1.0e3


def test_to_trajectory_feeds_location_accesses():
    itc = ITC.from_file(FULL)
    traj = itc.to_trajectory()
    station = bh.PointLocation(-122.4194, 37.7749, 0.0)
    start = itc.start_epoch
    end = start + 12.0 * 3600.0
    windows = bh.location_accesses(
        [station], [traj], start, end, bh.ElevationConstraint(min_elevation_deg=10.0)
    )
    assert len(windows) > 0
    for w in windows:
        assert w.window_open >= start and w.window_close <= end
