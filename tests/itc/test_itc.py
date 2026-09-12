"""Tests for the Modified ITC format — parity with src/itc/*.rs."""

import numpy as np
import pytest

import brahe as bh
from brahe.itc import (
    ITC,
    ITCCovarianceFrame,
    ITCHeader,
    ITCStateVector,
    data_type_for_state_frame,
    state_frame_for_data_type,
)
from brahe.spacetrack import EphemerisFileCategory

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


def test_state_frame_data_type_mapping():
    for token in ["MEME", "meme", "EME2000", "J2000"]:
        assert state_frame_for_data_type(token) == bh.CelestialFrame.EME2000
    assert state_frame_for_data_type("TEME") == bh.CelestialFrame.TEME
    assert state_frame_for_data_type("ITRF") == bh.CelestialFrame.ITRF
    with pytest.raises(bh.BraheError):
        state_frame_for_data_type("GCRF")
    assert data_type_for_state_frame(bh.CelestialFrame.EME2000) == "MEME"
    assert data_type_for_state_frame(bh.CelestialFrame.TEME) == "TEME"
    assert data_type_for_state_frame(bh.CelestialFrame.ITRF) == "ITRF"
    with pytest.raises(bh.BraheError):
        data_type_for_state_frame(bh.CelestialFrame.GCRF)


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
        100002, "STARLINK-37711", EphemerisFileCategory.OPERATIONAL, "1473385800"
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
        str(teme.file_name(25544, "ISS", EphemerisFileCategory.SPECIAL, ""))
        == "TEME_25544_ISS_3001224_Special__UNCLASSIFIED.txt"
    )
    with pytest.raises(bh.BraheError):
        ITC(ITCHeader()).file_name(1, "A", EphemerisFileCategory.OPERATIONAL, "")
    with pytest.raises(bh.BraheError):
        ITC(ITCHeader(state_frame=bh.CelestialFrame.GCRF)).file_name(
            1, "A", EphemerisFileCategory.OPERATIONAL, ""
        )


def test_top_level_exports():
    assert bh.ITC is ITC
    assert bh.ITCHeader is ITCHeader
    assert bh.ITCStateVector is ITCStateVector
    assert bh.ITCCovarianceFrame is ITCCovarianceFrame
    assert bh.state_frame_for_data_type is state_frame_for_data_type
    assert bh.data_type_for_state_frame is data_type_for_state_frame
    assert bh.EphemerisFileName is bh.spacetrack.EphemerisFileName
    assert bh.EphemerisFileCategory is bh.spacetrack.EphemerisFileCategory
