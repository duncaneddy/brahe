"""Tests for the SP3 format — parity with src/datafiles/sp3/{types,parse,write}.rs."""

import math
import pathlib

import numpy as np
import pytest

import brahe as bh
from brahe.datafiles import (
    SP3,
    GNSSSystem,
    SP3Correlation,
    SP3FileType,
    SP3Header,
    SP3OrbitType,
    SP3Record,
    SP3SatelliteId,
    SP3TimeSystem,
    SP3Version,
)

GPS_ZERO_MJD = 44244
SECONDS_PER_DAY = 86400.0


def fixture(name):
    return f"test_assets/sp3/{name}"


def read_fixture(name):
    with open(fixture(name)) as f:
        return f.read()


def sat(code):
    return SP3SatelliteId.parse(code)


def gps_t0():
    return bh.Epoch.from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)


def assert_rel(actual, expected, tol):
    assert abs((actual - expected) / expected) <= tol, (
        f"{actual} differs from {expected} by more than {tol} relative"
    )


def assert_epoch(actual, expected):
    assert actual.time_system == expected.time_system
    assert actual - expected == pytest.approx(0.0, abs=1e-9)


def parse_error(content):
    with pytest.raises(bh.BraheError) as excinfo:
        SP3.from_str(content)
    return str(excinfo.value)


def file_error(name):
    with pytest.raises(bh.BraheError) as excinfo:
        SP3.from_file(fixture(name))
    return str(excinfo.value)


def write_error(sp3):
    with pytest.raises(bh.BraheError) as excinfo:
        sp3.to_string()
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# types.rs: identifiers and enumerations
# ---------------------------------------------------------------------------


def test_SP3SatelliteId_parse():
    assert SP3SatelliteId.parse("G05") == SP3SatelliteId(GNSSSystem.GPS, 5)
    assert str(SP3SatelliteId.parse("G 5")) == "G05"
    assert SP3SatelliteId.parse("  5").system == GNSSSystem.GPS
    assert SP3SatelliteId.parse(" 05").number == 5
    assert SP3SatelliteId.parse("R12").system == GNSSSystem.GLONASS
    assert SP3SatelliteId.parse("L01").system == GNSSSystem.LEO
    with pytest.raises(bh.BraheError):
        SP3SatelliteId.parse("X01")
    with pytest.raises(bh.BraheError):
        SP3SatelliteId(GNSSSystem.GPS, 0)
    with pytest.raises(bh.BraheError):
        SP3SatelliteId(GNSSSystem.GPS, 100)
    with pytest.raises(bh.BraheError):
        SP3SatelliteId(GNSSSystem.GPS, -1)


@pytest.mark.parametrize("text", ["", "G5", "G005", "G0A", "G+5", "G00", "G  "])
def test_SP3SatelliteId_parse_invalid(text):
    with pytest.raises(bh.BraheError):
        SP3SatelliteId.parse(text)


def test_SP3SatelliteId_ordering():
    ids = [SP3SatelliteId.parse(s) for s in ["E01", "G10", "G02", "R01"]]
    assert [str(s) for s in sorted(ids)] == ["G02", "G10", "R01", "E01"]
    assert SP3SatelliteId.parse("G02") < SP3SatelliteId.parse("G10")
    assert SP3SatelliteId.parse("R01") > SP3SatelliteId.parse("G99")


def test_SP3SatelliteId_Display():
    ident = SP3SatelliteId(GNSSSystem.BEIDOU, 7)
    assert str(ident) == "C07"
    assert SP3SatelliteId.parse(str(ident)) == ident


def test_SP3SatelliteId_eq_hash_and_system_argument():
    ident = SP3SatelliteId(GNSSSystem.GPS, 5)
    assert SP3SatelliteId("G", 5) == ident
    assert SP3SatelliteId("GPS", 5) == ident
    assert SP3SatelliteId("galileo", 5) == SP3SatelliteId(GNSSSystem.GALILEO, 5)
    assert hash(SP3SatelliteId.parse("G05")) == hash(ident)
    assert len({ident, SP3SatelliteId.parse("G05"), SP3SatelliteId.parse("G06")}) == 2
    assert ident != SP3SatelliteId.parse("R05")
    with pytest.raises(bh.BraheError):
        SP3SatelliteId("X", 5)
    with pytest.raises(bh.BraheError):
        SP3SatelliteId("Nonsense", 5)
    with pytest.raises(TypeError):
        SP3SatelliteId(5, 5)


def test_SP3TimeSystem_to_time_system():
    assert SP3TimeSystem.parse("ccc") == SP3TimeSystem.GPS
    assert SP3TimeSystem.GAL.to_time_system() == bh.TimeSystem.GST
    assert SP3TimeSystem.BDT.to_time_system() == bh.TimeSystem.BDT
    for t in [SP3TimeSystem.GLO, SP3TimeSystem.QZS, SP3TimeSystem.IRN]:
        assert t.to_time_system() is None
    assert SP3TimeSystem.from_time_system(bh.TimeSystem.GST) == SP3TimeSystem.GAL
    assert SP3TimeSystem.from_time_system(bh.TimeSystem.TT) is None


def test_SP3TimeSystem_roundtrip():
    for t in [
        SP3TimeSystem.GPS,
        SP3TimeSystem.GLO,
        SP3TimeSystem.GAL,
        SP3TimeSystem.BDT,
        SP3TimeSystem.TAI,
        SP3TimeSystem.UTC,
        SP3TimeSystem.QZS,
        SP3TimeSystem.IRN,
    ]:
        assert SP3TimeSystem.parse(str(t)) == t
    assert SP3TimeSystem.parse(" GPS ") == SP3TimeSystem.GPS
    with pytest.raises(bh.BraheError):
        SP3TimeSystem.parse("XYZ")
    for ts in [
        bh.TimeSystem.GPS,
        bh.TimeSystem.TAI,
        bh.TimeSystem.UTC,
        bh.TimeSystem.BDT,
    ]:
        sp3_ts = SP3TimeSystem.from_time_system(ts)
        assert sp3_ts.to_time_system() == ts


def test_SP3OrbitType_parse():
    assert SP3OrbitType.parse("FIT") == SP3OrbitType.FIT
    assert SP3OrbitType.parse("BHN") == SP3OrbitType.parse("BHN")
    assert SP3OrbitType.parse("BHN") not in (
        SP3OrbitType.FIT,
        SP3OrbitType.EXT,
        SP3OrbitType.BCT,
        SP3OrbitType.HLM,
    )
    assert str(SP3OrbitType.parse("BHN")) == "BHN"
    for t in ["FIT", "EXT", "BCT", "HLM"]:
        assert str(SP3OrbitType.parse(t)) == t
    assert SP3OrbitType.parse(" EXT ") == SP3OrbitType.EXT
    assert hash(SP3OrbitType.parse("EXT")) == hash(SP3OrbitType.EXT)


@pytest.mark.parametrize(
    "c, v",
    [
        ("a", SP3Version.A),
        ("b", SP3Version.B),
        ("c", SP3Version.C),
        ("d", SP3Version.D),
    ],
)
def test_SP3Version_roundtrip(c, v):
    assert SP3Version.from_char(c) == v
    assert SP3Version.from_char(c.upper()) == v
    assert v.as_char() == c
    assert str(v) == c
    with pytest.raises(bh.BraheError):
        SP3Version.from_char("e")


GNSS_CASES = [
    (GNSSSystem.GPS, "G", "GPS"),
    (GNSSSystem.GLONASS, "R", "GLONASS"),
    (GNSSSystem.GALILEO, "E", "Galileo"),
    (GNSSSystem.BEIDOU, "C", "BeiDou"),
    (GNSSSystem.QZSS, "J", "QZSS"),
    (GNSSSystem.IRNSS, "I", "IRNSS"),
    (GNSSSystem.SBAS, "S", "SBAS"),
    (GNSSSystem.LEO, "L", "LEO"),
]


@pytest.mark.parametrize("system, code, name", GNSS_CASES)
def test_GNSSSystem_roundtrip(system, code, name):
    assert GNSSSystem.from_code(code) == system
    assert system.code == code
    assert str(system) == name
    assert hash(GNSSSystem.from_code(code)) == hash(system)


def test_GNSSSystem_invalid_codes():
    with pytest.raises(bh.BraheError):
        GNSSSystem.from_code("X")
    with pytest.raises(bh.BraheError):
        GNSSSystem.from_code("M")


FILE_TYPE_CASES = [
    (SP3FileType.GPS, "G", "GPS"),
    (SP3FileType.GLONASS, "R", "GLONASS"),
    (SP3FileType.GALILEO, "E", "Galileo"),
    (SP3FileType.BEIDOU, "C", "BeiDou"),
    (SP3FileType.QZSS, "J", "QZSS"),
    (SP3FileType.IRNSS, "I", "IRNSS"),
    (SP3FileType.SBAS, "S", "SBAS"),
    (SP3FileType.LEO, "L", "LEO"),
    (SP3FileType.MIXED, "M", "Mixed"),
]


@pytest.mark.parametrize("file_type, code, name", FILE_TYPE_CASES)
def test_SP3FileType_roundtrip(file_type, code, name):
    assert SP3FileType.from_code(code) == file_type
    assert file_type.code == code
    assert str(file_type) == name
    assert hash(SP3FileType.from_code(code)) == hash(file_type)


def test_SP3FileType_invalid_code():
    with pytest.raises(bh.BraheError):
        SP3FileType.from_code("X")


# ---------------------------------------------------------------------------
# types.rs: SP3 construction and access
# ---------------------------------------------------------------------------


def test_SP3_new_and_push_epoch():
    sp3 = SP3(["G01", "G02"], 900.0)
    t0 = gps_t0()
    assert sp3.push_epoch(t0) == 0
    assert sp3.push_epoch(t0 + 900.0) == 1
    with pytest.raises(bh.BraheError):
        sp3.push_epoch(t0 + 1000.0)
    with pytest.raises(bh.BraheError):
        sp3.push_epoch(t0)
    assert sp3.push_epoch(t0 + 2700.0) == 2
    g02 = sat("G02")
    sp3.set_record(g02, 1, SP3Record([1.0, 2.0, 3.0], clock=1e-6))
    assert sp3.record(g02, 1).clock == 1e-6
    assert sp3.record(g02, 0) is None
    with pytest.raises(bh.BraheError):
        sp3.record(sat("G03"), 0)
    with pytest.raises(bh.BraheError):
        sp3.set_record(g02, 9, SP3Record([0.0, 0.0, 0.0]))
    with pytest.raises(bh.BraheError):
        sp3.set_record(sat("G03"), 0, SP3Record([0.0, 0.0, 0.0]))
    assert sp3.header.agency == "BRHE"
    assert sp3.header.time_system == SP3TimeSystem.GPS
    assert sp3.header.has_velocity


def test_SP3_new_header_fields():
    sp3 = SP3(["G01", "G02", "G03"], 300.0)
    h = sp3.header
    assert h.version == SP3Version.D
    assert h.file_type == SP3FileType.GPS
    assert h.epoch_interval == 300.0
    assert h.number_of_epochs == 0
    assert h.accuracies == [None, None, None]
    assert h.position_base is None
    assert h.clock_base is None
    assert h.orbit_type == SP3OrbitType.EXT
    assert h.coordinate_system == "IGS20"
    assert h.data_used == "ORBIT"
    assert sp3.num_satellites == 3
    assert sp3.num_epochs == 0
    assert len(sp3) == 0


def test_SP3_empty_start_end_epoch():
    sp3 = SP3(["G01"], 900.0)
    zero = bh.Epoch.from_gps_date(0, 0.0)
    assert sp3.start_epoch == zero
    assert sp3.end_epoch == zero
    assert sp3.header.start_epoch == zero
    assert sp3.header.gps_week == 0
    assert sp3.header.seconds_of_week == 0.0
    assert sp3.header.mjd == 44244
    assert sp3.header.fractional_day == 0.0


def test_SP3_first_push_epoch_sets_header_fields():
    sp3 = SP3(["G01"], 900.0)
    t0 = bh.Epoch.from_datetime(2024, 11, 10, 6, 0, 0.0, 0.0, bh.TimeSystem.GPS)
    sp3.push_epoch(t0)
    sp3.push_epoch(t0 + 900.0)
    h = sp3.header
    assert h.start_epoch == t0
    assert h.number_of_epochs == 2
    assert h.gps_week == 2340
    assert h.seconds_of_week == 6.0 * 3600.0
    assert h.mjd == 60624
    assert h.fractional_day == 0.25
    assert sp3.start_epoch == t0
    assert sp3.end_epoch == t0 + 900.0
    assert sp3.epochs == [t0, t0 + 900.0]
    assert len(sp3) == 2


def test_SP3_set_record_zero_position_is_absent():
    sp3 = SP3(["G01"], 900.0)
    sp3.push_epoch(gps_t0())
    sp3.set_record("G01", 0, SP3Record([1.0, 2.0, 3.0]))
    assert sp3.record("G01", 0) is not None
    sp3.set_record("G01", 0, SP3Record([0.0, 0.0, 0.0]))
    assert sp3.record("G01", 0) is None


def test_SP3_contains_and_satellite_ids():
    sp3 = SP3(["G02", "G01"], 900.0)
    assert sat("G01") in sp3
    assert "G01" in sp3
    assert sat("G03") not in sp3
    assert "G03" not in sp3
    assert sp3.satellite_ids == [sat("G02"), sat("G01")]
    with pytest.raises(bh.BraheError):
        "X01" in sp3  # noqa: B015


def test_SP3_header_mut():
    sp3 = SP3(["G01"], 900.0)
    header = sp3.header
    header.agency = "XYZ"
    sp3.header = header
    assert sp3.header.agency == "XYZ"


def test_SP3_header_is_copy():
    sp3 = SP3.from_file(fixture("example-d-1.sp3"))
    h = sp3.header
    h.agency = "TEST"
    assert sp3.header.agency != "TEST"
    sp3.header = h
    assert sp3.header.agency == "TEST"


def test_SP3_record_is_copy():
    sp3 = SP3(["G01"], 900.0)
    sp3.push_epoch(gps_t0())
    sp3.set_record("G01", 0, SP3Record([1.0, 2.0, 3.0], clock=1e-6))
    record = sp3.record("G01", 0)
    record.clock = 2e-6
    record.maneuver = True
    assert sp3.record("G01", 0).clock == 1e-6
    assert not sp3.record("G01", 0).maneuver
    sp3.set_record("G01", 0, record)
    assert sp3.record("G01", 0).clock == 2e-6
    assert sp3.record("G01", 0).maneuver


def test_SP3_new_options_applied():
    sp3 = SP3(
        ["G01"],
        60.0,
        agency="ABC",
        time_system=SP3TimeSystem.UTC,
        coordinate_system="ITRF2",
        orbit_type=SP3OrbitType.FIT,
        data_used="DOPPL",
        include_velocity=False,
        comments=["note"],
    )
    h = sp3.header
    assert h.agency == "ABC"
    assert h.time_system == SP3TimeSystem.UTC
    assert h.coordinate_system == "ITRF2"
    assert h.orbit_type == SP3OrbitType.FIT
    assert h.data_used == "DOPPL"
    assert not h.has_velocity
    assert h.comments == ["note"]


def test_SP3_new_accepts_string_codes():
    sp3 = SP3(
        [sat("G01"), "G02"],
        60.0,
        time_system="UTC",
        orbit_type="FIT",
    )
    assert sp3.header.time_system == SP3TimeSystem.UTC
    assert sp3.header.orbit_type == SP3OrbitType.FIT
    assert sp3.satellite_ids == [sat("G01"), sat("G02")]
    custom = SP3(["G01"], 60.0, orbit_type="BHN")
    assert str(custom.header.orbit_type) == "BHN"
    with pytest.raises(bh.BraheError):
        SP3(["G01"], 60.0, time_system="XYZ")
    with pytest.raises(TypeError):
        SP3(["G01"], 60.0, time_system=5)
    with pytest.raises(TypeError):
        SP3([5], 60.0)


def test_SP3_header_setters_accept_string_codes():
    sp3 = SP3(["G01"], 900.0)
    header = sp3.header
    header.time_system = "UTC"
    header.orbit_type = "HLM"
    assert header.time_system == SP3TimeSystem.UTC
    assert header.orbit_type == SP3OrbitType.HLM


def test_SP3_new_errors():
    with pytest.raises(bh.BraheError):
        SP3(["G01", "G01"], 900.0)
    with pytest.raises(bh.BraheError):
        SP3(["G01"], 0.0)
    with pytest.raises(bh.BraheError):
        SP3(["G01"], -900.0)
    with pytest.raises(bh.BraheError):
        SP3(["G01"], math.nan)
    with pytest.raises(bh.BraheError):
        SP3(["G01"], 900.0, time_system=SP3TimeSystem.GLO)

    many = [
        SP3SatelliteId(s, n)
        for s in [
            GNSSSystem.GPS,
            GNSSSystem.GLONASS,
            GNSSSystem.GALILEO,
            GNSSSystem.BEIDOU,
            GNSSSystem.QZSS,
            GNSSSystem.IRNSS,
            GNSSSystem.SBAS,
            GNSSSystem.LEO,
        ]
        for n in range(1, 100)
    ]
    assert len(many) == 792
    assert SP3(many, 900.0).num_satellites == 792


def test_SP3_new_rejects_more_than_999_satellites():
    with pytest.raises(bh.BraheError) as excinfo:
        SP3(["G01"] * 1000, 900.0)
    assert "999" in str(excinfo.value)


@pytest.mark.parametrize(
    "ids, expected",
    [
        (["G01", "G02"], SP3FileType.GPS),
        (["R01"], SP3FileType.GLONASS),
        (["E01"], SP3FileType.GALILEO),
        (["C01"], SP3FileType.BEIDOU),
        (["J01"], SP3FileType.QZSS),
        (["I01"], SP3FileType.IRNSS),
        (["S01"], SP3FileType.SBAS),
        (["L01", "L02"], SP3FileType.LEO),
        (["G01", "E01"], SP3FileType.MIXED),
        (["G01", "L01"], SP3FileType.MIXED),
    ],
)
def test_SP3_new_file_type_from_systems(ids, expected):
    assert SP3(ids, 900.0).header.file_type == expected


def test_SP3_push_epoch_non_gps_time_system():
    sp3 = SP3(["G01"], 900.0, time_system=SP3TimeSystem.UTC)
    t0 = bh.Epoch.from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    sp3.push_epoch(t0)
    assert sp3.header.mjd == 60624
    assert sp3.header.fractional_day == 0.0


def test_SP3WriteOptions_defaults():
    h = SP3(["G01"], 900.0).header
    assert (h.agency, h.coordinate_system, h.data_used) == ("BRHE", "IGS20", "ORBIT")
    assert h.orbit_type == SP3OrbitType.EXT
    assert h.time_system == SP3TimeSystem.GPS
    assert h.has_velocity
    assert h.comments == []


def test_SP3Record_builders():
    r = SP3Record(
        [1.0, 2.0, 3.0],
        velocity=[4.0, 5.0, 6.0],
        clock=1e-6,
        clock_rate=1e-12,
    )
    assert r.position.tolist() == [1.0, 2.0, 3.0]
    assert r.velocity.tolist() == [4.0, 5.0, 6.0]
    assert r.clock == 1e-6
    assert r.clock_rate == 1e-12
    assert r.position_std == (None, None, None)
    assert not (r.clock_event or r.clock_predicted or r.maneuver or r.orbit_predicted)
    assert r.position_correlation is None and r.velocity_correlation is None
    assert SP3Record([0.0, 0.0, 0.0]).position.tolist() == [0.0, 0.0, 0.0]


def test_SP3Record_fields_and_vector_arguments():
    r = SP3Record(
        np.array([1.0, 2.0, 3.0]),
        velocity=np.array([4.0, 5.0, 6.0]),
        position_std=(0.01, None, 0.03),
        clock_std=1e-12,
        velocity_std=[1e-4, 2e-4, 3e-4],
        clock_rate_std=1e-16,
        clock_event=True,
        clock_predicted=True,
        maneuver=True,
        orbit_predicted=True,
        position_correlation=SP3Correlation(std=(0.1, 0.2, 0.3), xy=0.5),
    )
    assert isinstance(r.position, np.ndarray) and r.position.shape == (3,)
    assert isinstance(r.velocity, np.ndarray) and r.velocity.shape == (3,)
    assert r.position_std == (0.01, None, 0.03)
    assert r.velocity_std == (1e-4, 2e-4, 3e-4)
    assert r.clock_std == 1e-12
    assert r.clock_rate_std == 1e-16
    assert r.clock_event and r.clock_predicted and r.maneuver and r.orbit_predicted
    assert r.position_correlation.std == (0.1, 0.2, 0.3)
    assert r.position_correlation.xy == 0.5

    r.position = [7.0, 8.0, 9.0]
    r.velocity = None
    r.position_std = [None, None, None]
    r.velocity_correlation = SP3Correlation(clock_std=1e-17)
    assert r.position.tolist() == [7.0, 8.0, 9.0]
    assert r.velocity is None
    assert r.position_std == (None, None, None)
    assert r.velocity_correlation.clock_std == 1e-17

    with pytest.raises(ValueError):
        SP3Record([1.0, 2.0])
    with pytest.raises(ValueError):
        SP3Record([1.0, 2.0, 3.0], velocity=[1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        SP3Record([1.0, 2.0, 3.0], position_std=[1.0, 2.0])


def test_SP3Correlation_properties():
    corr = SP3Correlation()
    assert corr.std == (None, None, None)
    assert corr.clock_std is None
    assert (corr.xy, corr.xz, corr.xc, corr.yz, corr.yc, corr.zc) == (0.0,) * 6
    corr.std = (0.01, 0.01, 0.02)
    corr.clock_std = 1e-11
    corr.xy, corr.xz, corr.xc, corr.yz, corr.yc, corr.zc = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    assert corr.std[2] == 0.02
    assert corr.clock_std == 1e-11
    assert (corr.xy, corr.xz, corr.xc, corr.yz, corr.yc, corr.zc) == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    )


def test_SP3_accepts_pathlike(tmp_path):
    sp3 = SP3.from_file(pathlib.Path(fixture("example-d-2.sp3")))
    path = tmp_path / "out.sp3"
    sp3.to_file(path)
    assert SP3.from_file(path).num_epochs == sp3.num_epochs


def test_SP3_header_type():
    assert isinstance(SP3(["G01"], 900.0).header, SP3Header)


# ---------------------------------------------------------------------------
# parse.rs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "f, v, nsat, nep",
    [
        ("example-a-1.sp3", SP3Version.A, 25, 3),
        ("example-a-2.sp3", SP3Version.A, 25, 3),
        ("example-c-1.sp3", SP3Version.C, 26, 2),
        ("example-c-2.sp3", SP3Version.C, 26, 2),
        ("example-d-1.sp3", SP3Version.D, 140, 3),
        ("example-d-2.sp3", SP3Version.D, 26, 2),
        ("example-d-3.sp3", SP3Version.D, 26, 1),
        ("example-d-4.sp3", SP3Version.D, 26, 1),
        ("example-d-5.sp3", SP3Version.D, 26, 1),
    ],
)
def test_SP3_from_file_spec_examples(f, v, nsat, nep):
    sp3 = SP3.from_file(fixture(f))
    assert (sp3.header.version, sp3.num_satellites, sp3.num_epochs) == (v, nsat, nep)


def test_SP3_from_file_example_a_1():
    sp3 = SP3.from_file(fixture("example-a-1.sp3"))
    h = sp3.header
    assert h.version == SP3Version.A
    assert not h.has_velocity
    assert h.orbit_type == SP3OrbitType.FIT
    assert h.time_system == SP3TimeSystem.GPS
    assert h.file_type == SP3FileType.GPS
    assert h.agency == "NGS"
    assert h.coordinate_system == "ITR92"
    assert h.data_used == "d"
    assert h.number_of_epochs == 3
    assert h.gps_week == 779
    assert h.seconds_of_week == pytest.approx(518400.0, abs=1e-10)
    assert h.epoch_interval == pytest.approx(900.0, abs=1e-15)
    assert h.mjd == 49703
    assert h.fractional_day == pytest.approx(0.0, abs=1e-15)
    assert h.position_base is None
    assert h.clock_base is None
    assert len(h.comments) == 4
    assert len(h.accuracies) == 25
    assert h.accuracies[0] == pytest.approx(0.128, abs=1e-15)
    assert h.accuracies[2] == pytest.approx(0.032, abs=1e-15)

    t0 = bh.Epoch.from_datetime(1994, 12, 17, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)
    assert_epoch(h.start_epoch, t0)
    assert_epoch(sp3.start_epoch, t0)
    assert_epoch(sp3.epochs[1], t0 + 900.0)

    g01 = sat("G01")
    assert sp3.satellite_ids[0] == g01
    assert sp3.satellite_ids[24] == sat("G31")
    assert sat("G03") not in sp3
    assert sp3.record(sat("G04"), 2) is not None
    r = sp3.record(g01, 0)
    assert r.position[0] == pytest.approx(16258524.75, abs=1e-6)
    assert r.position[1] == pytest.approx(-3529015.75, abs=1e-6)
    assert r.position[2] == pytest.approx(-20611427.05, abs=1e-6)
    assert r.clock == pytest.approx(-62.5406e-6, abs=1e-15)
    assert r.velocity is None
    assert r.clock_rate is None
    assert r.position_std == (None, None, None)
    assert r.clock_std is None
    assert not (r.clock_event or r.clock_predicted or r.maneuver or r.orbit_predicted)


def test_SP3_from_file_example_a_2_velocity():
    sp3 = SP3.from_file(fixture("example-a-2.sp3"))
    assert sp3.header.has_velocity
    r = sp3.record(sat("G01"), 0)
    v = r.velocity
    assert v[0] == pytest.approx(-656.0373522, abs=1e-9)
    assert v[1] == pytest.approx(2560.5954994, abs=1e-9)
    assert v[2] == pytest.approx(-946.0427179, abs=1e-9)
    assert r.clock_rate == pytest.approx(-2.4236e-12, abs=1e-22)


def test_SP3_from_file_example_c_1_flags():
    sp3 = SP3.from_file(fixture("example-c-1.sp3"))
    h = sp3.header
    assert h.orbit_type == SP3OrbitType.HLM
    assert h.agency == "IGS"
    assert h.position_base == 1.25
    assert h.clock_base == 1.025

    r = sp3.record(sat("G01"), 0)
    assert r.position[0] == pytest.approx(-11044805.8, abs=1e-6)
    assert r.position[1] == pytest.approx(-10475672.35, abs=1e-6)
    assert r.position[2] == pytest.approx(21929418.2, abs=1e-6)
    assert r.clock == pytest.approx(189.1633e-6, abs=1e-15)

    g02 = sp3.record(sat("G02"), 0)
    assert g02.maneuver
    assert not (g02.clock_event or g02.clock_predicted or g02.orbit_predicted)

    g04 = sp3.record(sat("G04"), 1)
    assert g04.clock_event
    assert g04.clock_predicted
    assert not g04.maneuver
    assert g04.orbit_predicted

    g01 = sp3.record(sat("G01"), 1)
    assert not g01.clock_event
    assert g01.clock_predicted and g01.orbit_predicted

    assert sp3.record(sat("G07"), 0) is None
    assert sp3.record(sat("G05"), 1) is None


def test_SP3_from_file_example_c_2_correlations():
    sp3 = SP3.from_file(fixture("example-c-2.sp3"))
    r = sp3.record(sat("G01"), 0)
    v = r.velocity
    assert v[0] == pytest.approx(2029.8880364, abs=1e-9)
    assert v[1] == pytest.approx(-1846.2044804, abs=1e-9)
    assert v[2] == pytest.approx(138.1387685, abs=1e-9)
    assert r.clock_rate == pytest.approx(-4.534317e-10, abs=1e-22)

    ep = r.position_correlation
    for s in ep.std:
        assert s == pytest.approx(0.055, abs=1e-15)
    assert ep.clock_std == pytest.approx(222e-12, abs=1e-24)
    assert ep.xy == pytest.approx(0.1234567, abs=1e-15)
    assert ep.xz == pytest.approx(-0.1234567, abs=1e-15)
    assert ep.xc == pytest.approx(0.5999999, abs=1e-15)
    assert ep.yz == pytest.approx(-0.0000030, abs=1e-15)
    assert ep.yc == pytest.approx(0.0000021, abs=1e-15)
    assert ep.zc == pytest.approx(-0.1230000, abs=1e-15)

    ev = r.velocity_correlation
    for s in ev.std:
        assert s == pytest.approx(22e-7, abs=1e-19)
    assert ev.clock_std == pytest.approx(111e-16, abs=1e-28)
    for c in [ev.xy, ev.xz, ev.xc, ev.yz, ev.yc, ev.zc]:
        assert c == pytest.approx(0.1234567, abs=1e-15)

    g02 = sp3.record(sat("G02"), 0)
    assert g02.maneuver
    assert g02.position_correlation is not None


def test_SP3_from_file_example_d_1():
    sp3 = SP3.from_file(fixture("example-d-1.sp3"))
    h = sp3.header
    assert h.orbit_type == SP3OrbitType.BCT
    assert h.time_system == SP3TimeSystem.GPS
    assert h.file_type == SP3FileType.MIXED
    assert h.coordinate_system == "WGS84"
    assert h.agency == "MGEX"
    assert h.accuracies == [None] * 140
    assert h.comments == [
        "Note: This is a simulated file, meant to illustrate what an SP3-d header",
        "might look like with more than 85 satellites. Source for GPS and SBAS satel-",
        "lite positions: BRDM0930.13N. G=GPS,R=GLONASS,E=Galileo,C=BeiDou,J=QZSS,",
        "I=IRNSS,S=SBAS. For definitions of SBAS satellites, refer to the website:",
        "http://igs.org/mgex/status-SBAS",
    ]

    ids = sp3.satellite_ids
    assert ids[0] == sat("G01")
    assert ids[139] == sat("S38")
    assert sum(1 for s in ids if s.system == GNSSSystem.BEIDOU) == 35

    s37 = sat("S37")
    r = sp3.record(s37, 0)
    assert r.position[0] == pytest.approx(-34534904.566, abs=1e-6)
    assert r.position[1] == pytest.approx(24164610.955, abs=1e-6)
    assert r.position[2] == pytest.approx(29812.840, abs=1e-6)
    assert r.clock == pytest.approx(0.29942e-6, abs=1e-15)
    assert sp3.record(s37, 1) is None
    assert sp3.record(s37, 2) is not None
    assert_epoch(
        sp3.end_epoch,
        bh.Epoch.from_datetime(2013, 4, 3, 23, 45, 0.0, 0.0, bh.TimeSystem.GPS),
    )


def test_SP3_from_file_example_d_2():
    sp3 = SP3.from_file(fixture("example-d-2.sp3"))
    h = sp3.header
    assert h.time_system == SP3TimeSystem.GPS
    assert h.orbit_type == SP3OrbitType.HLM
    assert h.file_type == SP3FileType.GPS
    assert h.has_velocity

    g01 = sat("G01")
    r = sp3.record(g01, 0)
    assert r.position[0] == pytest.approx(-11044.805800e3, abs=1e-6)
    assert r.position[1] == pytest.approx(-10475.672350e3, abs=1e-6)
    assert r.position[2] == pytest.approx(21929.418200e3, abs=1e-6)
    assert r.clock == pytest.approx(189.163300e-6, abs=1e-15)
    for s in r.position_std:
        assert_rel(s, 1.25**18 * 1e-3, 1e-12)
    assert_rel(r.clock_std, 1.025**219 * 1e-12, 1e-12)
    for s in r.velocity_std:
        assert_rel(s, 1.25**14 * 1e-7, 1e-12)
    assert_rel(r.clock_rate_std, 1.025**191 * 1e-16, 1e-12)
    assert not (r.clock_event or r.clock_predicted or r.maneuver or r.orbit_predicted)

    later = sp3.record(g01, 1)
    assert not later.clock_event
    assert later.clock_predicted
    assert not later.maneuver
    assert later.orbit_predicted
    assert_epoch(
        sp3.epochs[1],
        bh.Epoch.from_datetime(2001, 8, 9, 23, 45, 0.0, 0.0, bh.TimeSystem.GPS),
    )

    assert sp3.record(sat("G06"), 0) is None
    assert sp3.record(sat("G03"), 1) is None


def test_SP3_from_str_bad_clock():
    content = (
        read_fixture("example-d-2.sp3")
        .replace(
            "PG01 -11044.805800 -10475.672350  21929.418200    189.163300 18 18 18 219\n",
            "PG01 -11044.805800 -10475.672350  21929.418200 999999.999999 18 18 18 219\n",
        )
        .replace(
            "VG01  20298.880364 -18462.044804   1381.387685     -4.534317 14 14 14 191\n",
            "VG01  20298.880364 -18462.044804   1381.387685 999999.999999 14 14 14 191\n",
        )
    )
    sp3 = SP3.from_str(content)
    r = sp3.record(sat("G01"), 0)
    assert r.clock is None
    assert r.clock_rate is None
    assert r.position[0] == pytest.approx(-11044805.8, abs=1e-6)
    assert r.velocity is not None
    g02 = sp3.record(sat("G02"), 0)
    assert g02.clock is not None


@pytest.mark.parametrize(
    "f, sp3_ts, ts, file_type",
    [
        ("example-d-3.sp3", SP3TimeSystem.UTC, bh.TimeSystem.UTC, SP3FileType.IRNSS),
        ("example-d-4.sp3", SP3TimeSystem.UTC, bh.TimeSystem.UTC, SP3FileType.SBAS),
        ("example-d-5.sp3", SP3TimeSystem.GAL, bh.TimeSystem.GST, SP3FileType.SBAS),
    ],
)
def test_SP3_from_file_time_systems(f, sp3_ts, ts, file_type):
    sp3 = SP3.from_file(fixture(f))
    assert sp3.header.time_system == sp3_ts
    assert sp3.header.file_type == file_type
    t0 = bh.Epoch.from_datetime(2001, 8, 8, 0, 0, 0.0, 0.0, ts)
    assert_epoch(sp3.header.start_epoch, t0)
    assert_epoch(sp3.start_epoch, t0)


def test_SP3_from_file_unsupported_time_system():
    assert (
        file_error("example-d-2-glo.sp3")
        == "SP3: line 13: time system 'GLO' is not supported"
    )


def test_SP3_from_file_missing_file():
    msg = file_error("does-not-exist.sp3")
    assert msg.startswith("SP3: failed to read ")


def test_SP3_from_str_unknown_line():
    content = read_fixture("example-c-1.sp3").replace("PG03", "XG03", 1)
    assert parse_error(content).startswith("SP3: line 26: ")


def test_SP3_from_str_undeclared_satellite():
    content = read_fixture("example-c-1.sp3").replace("PG03", "PG32", 1)
    msg = parse_error(content)
    assert msg.startswith("SP3: line 26: "), msg
    assert "G32" in msg


def test_SP3_from_str_too_large_standard_deviations():
    content = (
        read_fixture("example-d-2.sp3")
        .replace(
            "    189.163300 18 18 18 219\n",
            "    189.163300 99 18    999\n",
            1,
        )
        .replace(
            "EP    55   55   55     222",
            "EP  9999        55 9999999",
            1,
        )
    )
    sp3 = SP3.from_str(content)
    r = sp3.record(sat("G01"), 0)
    assert r.position_std[0] == math.inf
    assert_rel(r.position_std[1], 1.25**18 * 1e-3, 1e-12)
    assert r.position_std[2] is None
    assert r.clock_std == math.inf
    ep = r.position_correlation
    assert ep.std[0] == math.inf
    assert ep.std[1] is None
    assert ep.std[2] == pytest.approx(0.055, abs=1e-15)
    assert ep.clock_std == math.inf


def test_SP3_from_str_exponent_without_base():
    content = read_fixture("example-d-2.sp3").replace(
        "%f  1.2500000  1.025000000",
        "%f  1.2500000  0.000000000",
        1,
    )
    msg = parse_error(content)
    assert msg.startswith("SP3: line 24: "), msg
    assert "no clock base" in msg


def test_SP3_from_str_zero_velocity():
    content = read_fixture("example-d-2.sp3").replace(
        "VG01  20298.880364 -18462.044804   1381.387685",
        "VG01      0.000000      0.000000      0.000000",
        1,
    )
    sp3 = SP3.from_str(content)
    r = sp3.record(sat("G01"), 0)
    assert r.velocity is None
    assert r.clock_rate == pytest.approx(-4.534317e-10, abs=1e-22)


def test_SP3_from_str_record_errors():
    c1 = read_fixture("example-c-1.sp3")
    c2 = read_fixture("example-c-2.sp3")

    msg = parse_error(c1.replace("PG02", "PG01", 1))
    assert msg.startswith("SP3: line 25: "), msg
    assert "second P record for G01" in msg

    msg = parse_error(c2.replace("VG01", "VG02", 1))
    assert msg.startswith("SP3: line 26: "), msg
    assert "V record for G02" in msg

    msg = parse_error(c1.replace("    189.163300 18 18 18 219", "    189.16", 1))
    assert msg == "SP3: line 24: record truncated in columns 47-60"

    msg = parse_error(c1.replace("PG01 -11044.805800", "PG01             ", 1))
    assert msg == "SP3: line 24: missing x component"

    msg = parse_error(c1.replace("PG01 -11044.805800", "PG01 -11044.8058µ", 1))
    assert msg.startswith("SP3: line 24: non-ASCII"), msg

    msg = parse_error(c1.replace("*  2001  8  8", "*  2001  2 30", 1))
    assert msg.startswith("SP3: line 23: day out of range"), msg

    msg = parse_error(c1.replace("*  2001", "EP", 1))
    assert msg.startswith("SP3: line 23: unexpected EP line after a /* line"), msg


def test_SP3_from_str_header_errors():
    c1 = read_fixture("example-c-1.sp3")

    assert parse_error("") == "SP3: empty file"
    assert (
        parse_error("## 1126")
        == "SP3: line 1: the first line is not an SP3 version line"
    )
    header_only = "".join(f"{line}\n" for line in c1.splitlines()[:14])
    assert (
        parse_error(header_only[: header_only.find("%c")])
        == "SP3: file ends before the %c time system line"
    )
    assert parse_error(c1.replace("#cP", "#cX", 1)).startswith(
        "SP3: line 1: invalid position/velocity flag 'X'"
    )
    assert parse_error(c1.replace("+   26", "+   27", 1)).startswith(
        "SP3: line 8: header declares 27 satellites but lists 26"
    )
    assert parse_error(c1.replace("G02G03", "G02G02", 1)).startswith(
        "SP3: line 3: satellite G02 is listed twice"
    )
    assert parse_error(c1.replace("%c G ", "%c cc", 1)).startswith(
        "SP3: line 13: unknown file type 'cc'"
    )
    assert parse_error(c1.replace("%c G  cc GPS", "%c G  cc XYZ", 1)).startswith(
        "SP3: line 13: unknown time system 'XYZ'"
    )


def test_SP3_from_str_version_a_file_type_placeholder():
    content = (
        read_fixture("example-c-1.sp3")
        .replace("#cP", "#bP", 1)
        .replace("PG03", "PR03")
        .replace("G03G04", "R03G04", 1)
        .replace("%c G ", "%c cc", 1)
    )
    sp3 = SP3.from_str(content)
    assert sp3.header.version == SP3Version.B
    assert sp3.header.file_type == SP3FileType.MIXED
    sp3 = SP3.from_file(fixture("example-a-2.sp3"))
    assert sp3.header.file_type == SP3FileType.GPS


@pytest.mark.parametrize(
    "f, nsat, nep",
    [
        ("issue895-hours-increment.sp3", 1, 61),
        ("issue895-minutes-increment.sp3", 1, 91),
        ("issue1014-days-increment.sp3", 1, 62),
        ("issue895-second-digits.sp3", 1, 1),
        ("issue895-no-eof.sp3", 1, 1),
        ("issue895-clock-record.sp3", 1, 1),
        ("issue895-header-comment.sp3", 1, 1),
        ("three-hours.sp3", 2, 36),
        ("missing-agency.sp3", 26, 1),
        ("issue1327-136-sats.sp3", 136, 4),
        ("missing-standard-deviation.sp3", 32, 21),
        ("truncated-nsgf.orb.lageos2.160305.v35.sp3", 1, 1),
        ("esaBHN.sp3", 53, 2),
        ("esaPRO.sp3", 53, 2),
        ("unknownType.sp3", 53, 2),
        ("nsgf.orb.stella.v00.sp3", 1, 100),
        ("too-many-comments-a.sp3", 25, 3),
        ("too-long-comment-a.sp3", 25, 3),
        ("too-long-comment-d.sp3", 26, 1),
        ("exceeded-sat-count.sp3", 140, 3),
    ],
)
def test_SP3_from_file_quirks(f, nsat, nep):
    sp3 = SP3.from_file(fixture(f))
    assert (sp3.num_satellites, sp3.num_epochs) == (nsat, nep)
    assert sp3.header.number_of_epochs == nep


def test_SP3_from_file_minutes_rollover():
    sp3 = SP3.from_file(fixture("issue895-minutes-increment.sp3"))

    def at(h, m):
        return bh.Epoch.from_datetime(2016, 7, 6, h, m, 0.0, 0.0, bh.TimeSystem.UTC)

    assert_epoch(sp3.epochs[29], at(16, 58))
    assert_epoch(sp3.epochs[30], at(17, 0))
    assert_epoch(sp3.epochs[31], at(17, 2))
    assert_epoch(sp3.epochs[60], at(18, 0))
    r = sp3.record(sat("L51"), 30)
    assert r.position[0] == pytest.approx(11948228.978, abs=1e-6)
    assert r.velocity[2] == pytest.approx(-5331.6820671, abs=1e-9)


def test_SP3_from_file_hours_rollover():
    sp3 = SP3.from_file(fixture("issue895-hours-increment.sp3"))
    assert_epoch(
        sp3.epochs[29],
        bh.Epoch.from_datetime(2016, 7, 6, 23, 58, 0.0, 0.0, bh.TimeSystem.UTC),
    )
    assert_epoch(
        sp3.epochs[30],
        bh.Epoch.from_datetime(2016, 7, 7, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC),
    )
    assert_epoch(
        sp3.epochs[31],
        bh.Epoch.from_datetime(2016, 7, 7, 0, 2, 0.0, 0.0, bh.TimeSystem.UTC),
    )
    r = sp3.record(sat("L51"), 30)
    assert r.position[0] == pytest.approx(2989229.334, abs=1e-6)


def test_SP3_from_file_days_rollover():
    sp3 = SP3.from_file(fixture("issue1014-days-increment.sp3"))
    h = sp3.header
    assert h.coordinate_system == "ITRF14"
    assert h.agency == "JCET"
    assert h.time_system == SP3TimeSystem.UTC

    def at(d, hour, m):
        return bh.Epoch.from_datetime(2021, 12, d, hour, m, 0.0, 0.0, bh.TimeSystem.UTC)

    assert_epoch(sp3.epochs[30], at(31, 0, 0))
    assert_epoch(sp3.epochs[59], at(31, 0, 58))
    assert_epoch(sp3.epochs[60], at(31, 1, 0))
    assert_epoch(sp3.epochs[61], at(31, 1, 2))
    r = sp3.record(sat("L51"), 61)
    assert r.position[0] == pytest.approx(-5691093.473, abs=1e-6)
    assert r.velocity[0] == pytest.approx(-3918.8598237, abs=1e-9)


def test_SP3_from_file_second_digits():
    sp3 = SP3.from_file(fixture("issue895-second-digits.sp3"))
    assert_epoch(
        sp3.start_epoch,
        bh.Epoch.from_datetime(2016, 7, 3, 0, 0, 0.1234, 0.0, bh.TimeSystem.UTC),
    )


def test_SP3_from_file_136_satellites():
    sp3 = SP3.from_file(fixture("issue1327-136-sats.sp3"))
    assert sp3.num_satellites == 136
    assert len(sp3.header.accuracies) == 136
    assert sp3.header.gps_week == 0
    assert sp3.header.mjd == 0
    assert_epoch(
        sp3.start_epoch,
        bh.Epoch.from_datetime(2024, 2, 20, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS),
    )


def test_SP3_from_file_missing_agency():
    sp3 = SP3.from_file(fixture("missing-agency.sp3"))
    assert sp3.header.version == SP3Version.C
    assert sp3.header.agency == ""


def test_SP3_from_file_truncated_clock_columns():
    sp3 = SP3.from_file(fixture("truncated-nsgf.orb.lageos2.160305.v35.sp3"))
    assert sp3.header.file_type == SP3FileType.LEO
    r = sp3.record(sat("L52"), 0)
    assert r.position[0] == pytest.approx(2228470.946, abs=1e-6)
    assert r.velocity[2] == pytest.approx(-711.62228, abs=1e-9)
    assert r.clock is None
    assert r.clock_rate is None


@pytest.mark.parametrize(
    "f, label",
    [
        ("esaBHN.sp3", "BHN"),
        ("esaPRO.sp3", "PRO"),
        ("unknownType.sp3", "UKN"),
    ],
)
def test_SP3_from_file_other_orbit_types(f, label):
    sp3 = SP3.from_file(fixture(f))
    assert sp3.header.orbit_type == SP3OrbitType.parse(label)
    assert str(sp3.header.orbit_type) == label
    assert sp3.record(sat("G18"), 0) is not None


def test_SP3_from_file_missing_standard_deviation():
    sp3 = SP3.from_file(fixture("missing-standard-deviation.sp3"))
    g06 = sat("G06")
    for i in [7, 8]:
        r = sp3.record(g06, i)
        assert r.position_std[0] is not None
        assert r.position_std[1] is None
        assert r.position_std[2] is not None
        assert r.clock_std is not None
    r = sp3.record(g06, 6)
    assert all(s is not None for s in r.position_std)


def test_SP3_from_file_zero_padded_satellite_ids():
    sp3 = SP3.from_file(fixture("three-hours.sp3"))
    assert sp3.satellite_ids == [sat("C01"), sat("C02")]
    assert len(sp3.header.accuracies) == 2


def test_SP3_from_file_ilrs_comments_and_no_eof():
    sp3 = SP3.from_file(fixture("issue895-no-eof.sp3"))
    h = sp3.header
    assert h.file_type == SP3FileType.LEO
    assert h.time_system == SP3TimeSystem.UTC
    assert h.comments[0] == "ilrsa.orb.lageos1.220101.v70.sp3 Reference TRF: SLRF2008"
    assert len(h.comments) == 4
    r = sp3.record(sat("L51"), 0)
    assert r.position[2] == pytest.approx(-11075527.276, abs=1e-6)
    assert r.velocity[0] == pytest.approx(-1772.0521773, abs=1e-9)
    assert r.clock is None


def test_SP3_from_str_eof_case_and_blank_lines():
    c1 = read_fixture("example-c-1.sp3")
    sp3 = SP3.from_str(c1.replace("\nEOF", "\neof  "))
    assert sp3.num_epochs == 2
    sp3 = SP3.from_str(c1.replace("\nEOF", "\n").replace("\n/*", "\n\n/*"))
    assert sp3.num_epochs == 2
    assert len(sp3.header.comments) == 4


def test_SP3_from_str_blank_partial_fields():
    content = (
        read_fixture("example-d-2.sp3")
        .replace("    189.163300 18 18 18 219\n", "    189.163300  \n", 1)
        .replace(
            "PG02 -12593.593500  10170.327650 -20354.534400    -55.976000 18 18 18 219     M\n",
            "PG02 -12593.593500  10170.327650 -20354.534400    \n",
            1,
        )
    )
    sp3 = SP3.from_str(content)
    r = sp3.record(sat("G01"), 0)
    assert r.clock == pytest.approx(189.1633e-6, abs=1e-15)
    assert r.position_std == (None, None, None)
    assert r.clock_std is None
    r = sp3.record(sat("G02"), 0)
    assert r.position[2] == pytest.approx(-20354534.4, abs=1e-6)
    assert r.clock is None


@pytest.mark.parametrize(
    "f, prefix, detail",
    [
        ("truncated-line.sp3", "SP3: line 27: ", "record truncated in columns 47-60"),
        (
            "missing-eof.sp3",
            "SP3: ",
            "header declares 192 epochs but the file has 1",
        ),
        (
            "wrong-number-of-epochs.sp3",
            "SP3: ",
            "header declares 192 epochs but the file has 2",
        ),
        (
            "wrong-line-identifier.sp3",
            "SP3: line 13: ",
            "unrecognized line identifier",
        ),
        ("unsupported-version.sp3", "SP3: line 1: ", "unsupported version 'z'"),
        ("wrong-clock-base-a.sp3", "SP3: line 15: ", "clock base"),
        ("wrong-pos-vel-base-a.sp3", "SP3: line 15: ", "position/velocity base"),
        ("wrong-clock-base-d.sp3", "SP3: line 24: ", "clock base"),
        ("wrong-pos-vel-base-d.sp3", "SP3: line 24: ", "position/velocity base"),
        (
            "inconsistent-sampling-dates.sp3",
            "SP3: line 37: ",
            "is not on the 900 s grid",
        ),
        (
            "example-d-2-glo.sp3",
            "SP3: line 13: ",
            "time system 'GLO' is not supported",
        ),
    ],
)
def test_SP3_from_file_errors(f, prefix, detail):
    msg = file_error(f)
    assert msg.startswith("SP3"), msg
    assert msg.startswith(prefix), msg
    assert detail in msg, msg


def test_SP3_from_str_epoch_order_and_rollover_errors():
    c1 = read_fixture("example-c-1.sp3")
    msg = parse_error(c1.replace("*  2001  8  9 23 45", "*  2001  8  7 23 45", 1))
    assert msg.startswith("SP3: line 50: "), msg
    assert "is not after the last epoch" in msg

    msg = parse_error(c1.replace("*  2001  8  9 23 45", "*  2001  8  9 23 61", 1))
    assert msg.startswith("SP3: line 50: minute out of range"), msg

    msg = parse_error(c1.replace("*  2001  8  9 23 45", "*  2001  8  9 25 45", 1))
    assert msg.startswith("SP3: line 50: hour out of range"), msg

    msg = parse_error(c1.replace("*  2001  8  9 23 45", "*  2001  0  0 23 45", 1))
    assert msg.startswith("SP3: line 50: month out of range"), msg

    msg = parse_error(c1.replace("*  2001  8  9 23 45", "*  2001  8  9 255 60", 1))
    assert msg.startswith("SP3: line 50: hour out of range"), msg

    msg = parse_error(c1.replace("*  2001  8  9 23 45", "*  4294967295 12 31 24 45", 1))
    assert msg.startswith("SP3: line 50: "), msg


def test_SP3_from_str_first_epoch_rollover_with_distant_header_start():
    content = read_fixture("issue895-minutes-increment.sp3").replace(
        "*  2016  7  6 16  0  0.00000000",
        "*  2016  7  6 15 60  0.00000000",
        1,
    )
    sp3 = SP3.from_str(content)
    assert_epoch(
        sp3.start_epoch,
        bh.Epoch.from_datetime(2016, 7, 6, 16, 0, 0.0, 0.0, bh.TimeSystem.UTC),
    )

    content = (
        read_fixture("example-c-1.sp3")
        .replace("#cP2001  8  8", "#cP2001  8  5", 1)
        .replace("*  2001  8  8  0  0", "*  2001  8  7 24  0", 1)
    )
    sp3 = SP3.from_str(content)
    assert_epoch(
        sp3.start_epoch,
        bh.Epoch.from_datetime(2001, 8, 8, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS),
    )


def test_SP3_from_str_rollover_after_gap():
    content = read_fixture("example-c-1.sp3").replace(
        "*  2001  8  9 23 45",
        "*  2001  8  9 22 60",
        1,
    )
    sp3 = SP3.from_str(content)
    assert_epoch(
        sp3.epochs[1],
        bh.Epoch.from_datetime(2001, 8, 9, 23, 0, 0.0, 0.0, bh.TimeSystem.GPS),
    )


def test_SP3_from_str_one_second_day():
    lines = [
        "#dP2024  1  1  0  0  1.00000000   86400 ORBIT IGS20 FIT  TST",
        "## 2295  86401.00000000     1.00000000 60310 0.0000115740741",
        "+    1   G01",
        "++         0",
        "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* ONE-SECOND GRID",
    ]
    for i in range(1, 86_401):
        if i == 86_400:
            lines.append("*  2024  1  1 23 59 59.99999999")
        else:
            lines.append(
                f"*  2024  1  1 {i // 3600:2d} {(i % 3600) // 60:2d} {float(i % 60):11.8f}"
            )
        lines.append("PG01  15000.000000  20000.000000   5000.000000    100.000000")
    lines.append("EOF")

    sp3 = SP3.from_str("\n".join(lines) + "\n")
    assert sp3.num_epochs == 86_400
    t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 1.0, 0.0, bh.TimeSystem.GPS)
    assert_epoch(sp3.start_epoch, t0)
    assert sp3.end_epoch - t0 == pytest.approx(86_399.0, abs=1e-7)


# ---------------------------------------------------------------------------
# write.rs
# ---------------------------------------------------------------------------


def two_satellite_file(**options):
    sp3 = SP3(["G01", "G02"], 900.0, **options)
    for k in range(2):
        sp3.push_epoch(gps_t0() + 900.0 * k)
        sp3.set_record(
            "G01",
            k,
            SP3Record(
                [15_000_000.0, 20_000_000.0, 5_000_000.0],
                velocity=[1000.0, -2000.0, 500.0],
                clock=2.5e-5,
                clock_rate=1e-12,
            ),
        )
    return sp3


def assert_option_close(read, written, tol, what):
    if read is None and written is None:
        return
    assert read is not None and written is not None, f"{what}: {read} != {written}"
    assert abs(read - written) <= tol, f"{what}: {read} != {written}"


def assert_quantized(read, written, base, what):
    """A written standard deviation is the smallest power of the base not
    below the read value, so it lies in [read, read * base)."""
    if read is None and written is None:
        return
    assert read is not None and written is not None, f"{what}: {read} != {written}"
    if math.isinf(read):
        assert math.isinf(written), f"{what}: {written}"
        return
    assert written >= read * (1.0 - 1e-12) and written < read * base, (
        f"{what}: written {written} outside [{read}, {read} * {base})"
    )


def assert_correlation_close(read, written, what):
    if read is None and written is None:
        return
    assert read is not None and written is not None, what
    for k in range(3):
        assert_option_close(read.std[k], written.std[k], 1e-15, what)
    assert_option_close(read.clock_std, written.clock_std, 1e-22, what)
    for name in ["xy", "xz", "xc", "yz", "yc", "zc"]:
        assert getattr(read, name) == pytest.approx(getattr(written, name), abs=1e-12)


def assert_record_close(read, written, pb, cb):
    for k in range(3):
        assert read.position[k] == pytest.approx(written.position[k], abs=1e-6)
        assert_quantized(
            read.position_std[k], written.position_std[k], pb, "position std"
        )
        assert_quantized(
            read.velocity_std[k], written.velocity_std[k], pb, "velocity std"
        )
    assert_option_close(read.clock, written.clock, 1e-12, "clock")
    assert_quantized(read.clock_std, written.clock_std, cb, "clock std")
    if read.velocity is None:
        assert written.velocity is None
    else:
        assert written.velocity is not None
        for k in range(3):
            assert read.velocity[k] == pytest.approx(written.velocity[k], abs=1e-7)
    assert_option_close(read.clock_rate, written.clock_rate, 1e-16, "clock rate")
    assert_quantized(read.clock_rate_std, written.clock_rate_std, cb, "clock rate std")
    assert (
        read.clock_event,
        read.clock_predicted,
        read.maneuver,
        read.orbit_predicted,
    ) == (
        written.clock_event,
        written.clock_predicted,
        written.maneuver,
        written.orbit_predicted,
    )
    assert_correlation_close(
        read.position_correlation, written.position_correlation, "EP"
    )
    assert_correlation_close(
        read.velocity_correlation, written.velocity_correlation, "EV"
    )


def header_time_fields(epoch, time_system):
    year, month, day, hour, minute, second, nanosecond = (
        epoch.to_datetime_as_time_system(time_system)
    )
    mjd = round(bh.datetime_to_mjd(year, month, day, 0, 0, 0.0, 0.0))
    seconds_of_day = hour * 3600.0 + minute * 60.0 + second + nanosecond * 1.0e-9
    days_since_gps_zero = mjd - GPS_ZERO_MJD
    week = days_since_gps_zero // 7
    day_of_week = days_since_gps_zero % 7
    return (
        week,
        day_of_week * SECONDS_PER_DAY + seconds_of_day,
        mjd,
        seconds_of_day / SECONDS_PER_DAY,
    )


def header_fields(h):
    return (
        h.version,
        h.has_velocity,
        h.start_epoch,
        h.number_of_epochs,
        h.data_used,
        h.coordinate_system,
        h.orbit_type,
        h.agency,
        h.gps_week,
        h.seconds_of_week,
        h.epoch_interval,
        h.mjd,
        h.fractional_day,
        h.file_type,
        h.time_system,
        h.position_base,
        h.clock_base,
        h.accuracies,
        h.comments,
    )


ROUND_TRIP_FIXTURES = [
    "example-a-1.sp3",
    "example-a-2.sp3",
    "example-c-1.sp3",
    "example-c-2.sp3",
    "example-d-1.sp3",
    "example-d-2.sp3",
    "example-d-3.sp3",
    "example-d-4.sp3",
    "example-d-5.sp3",
    "issue895-hours-increment.sp3",
    "issue895-minutes-increment.sp3",
    "issue1014-days-increment.sp3",
    "issue895-second-digits.sp3",
    "issue895-no-eof.sp3",
    "issue895-clock-record.sp3",
    "issue895-header-comment.sp3",
    "three-hours.sp3",
    "missing-agency.sp3",
    "issue1327-136-sats.sp3",
    "missing-standard-deviation.sp3",
    "truncated-nsgf.orb.lageos2.160305.v35.sp3",
    "esaBHN.sp3",
    "esaPRO.sp3",
    "unknownType.sp3",
    "nsgf.orb.stella.v00.sp3",
    "too-many-comments-a.sp3",
    "too-long-comment-a.sp3",
    "exceeded-sat-count.sp3",
    "gbm19500_after_drop.sp3",
    "gbm19500_after_no_drop.sp3",
    "gbm19500_large_gap.sp3",
    "gbm19500_truncated.sp3",
    "gbm19500_wrong_agency.sp3",
    "gbm19500_wrong_coordinate_system.sp3",
    "gbm19500_wrong_data_used.sp3",
    "gbm19500_wrong_derivatives.sp3",
    "gbm19500_wrong_frame.sp3",
    "gbm19500_wrong_mu.sp3",
    "gbm19500_wrong_orbit_type.sp3",
    "gbm19500_wrong_satellite_count.sp3",
    "gbm19500_wrong_satellite_list.sp3",
    "gbm19500_wrong_time_system.sp3",
    "gbm19500_wrong_type.sp3",
]


@pytest.mark.parametrize("f", ROUND_TRIP_FIXTURES)
def test_SP3_round_trip(f):
    read = SP3.from_file(fixture(f))
    text = read.to_string()
    written = SP3.from_str(text)

    assert written.epochs == read.epochs
    assert written.satellite_ids == read.satellite_ids

    rh, wh = read.header, written.header
    for k in range(read.num_epochs):
        for sat_id in read.satellite_ids:
            r = read.record(sat_id, k)
            w = written.record(sat_id, k)
            if r is None and w is None:
                continue
            assert r is not None and w is not None, f"epoch {k} satellite {sat_id}"
            assert_record_close(r, w, rh.position_base, rh.clock_base)

    ts = rh.time_system.to_time_system()
    week, sow, mjd, fd = header_time_fields(read.start_epoch, ts)
    assert wh.version == SP3Version.D
    assert wh.start_epoch == read.start_epoch
    assert wh.number_of_epochs == read.num_epochs
    assert (wh.gps_week, wh.mjd) == (week, mjd)
    assert wh.seconds_of_week == pytest.approx(sow, abs=1e-8)
    assert wh.fractional_day == pytest.approx(fd, abs=1e-13)
    for r, w in zip(rh.accuracies, wh.accuracies):
        assert_quantized(r, w, 2.0, "accuracy")

    expected = rh
    expected.version = SP3Version.D
    expected.start_epoch = wh.start_epoch
    expected.number_of_epochs = wh.number_of_epochs
    expected.gps_week = wh.gps_week
    expected.seconds_of_week = wh.seconds_of_week
    expected.mjd = wh.mjd
    expected.fractional_day = wh.fractional_day
    expected.accuracies = wh.accuracies
    if len(expected.comments) < 4:
        expected.comments = expected.comments + [""] * (4 - len(expected.comments))
    assert header_fields(wh) == header_fields(expected)


def test_SP3_to_string_too_long_comment_fixture():
    msg = write_error(SP3.from_file(fixture("too-long-comment-d.sp3")))
    assert "comment 3 is longer than 77 characters" in msg


def test_SP3_to_string_layout():
    text = SP3.from_file(fixture("example-d-1.sp3")).to_string()
    lines = text.splitlines()
    assert lines[0].startswith("#dP"), lines[0]
    assert sum(1 for l in lines if l.startswith("+") and not l.startswith("++")) >= 5
    assert sum(1 for l in lines if l.startswith("++")) >= 5
    assert sum(1 for l in lines if l.startswith("/*")) >= 4
    assert lines[-1] == "EOF"
    assert text.endswith("EOF\n")
    for line in lines:
        assert len(line) <= 80, f"line too long: {line}"
        assert line.rstrip() == line, f"trailing whitespace: {line!r}"


def test_SP3_to_string_header_lines():
    lines = SP3.from_file(fixture("example-d-2.sp3")).to_string().splitlines()
    assert lines[0] == "#dV2001  8  8  0  0  0.00000000       2 ORBIT IGS97 HLM MGEX"
    assert lines[1] == "## 1126 259200.00000000   900.00000000 52129 0.0000000000000"
    assert lines[2] == "+   26   G01G02G03G04G05G06G07G08G09G10G11G13G14G17G18G20G21"
    assert lines[3] == "+        G23G24G25G26G27G28G29G30G31  0  0  0  0  0  0  0  0"
    assert lines[6] == "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0"
    assert lines[7] == "++         7  8  7  8  6  7  7  7  7  7  7  7  7  8  8  7  9"
    assert lines[12] == "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
    assert lines[13] == "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
    assert lines[14] == "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000"
    assert lines[15] == "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000"
    assert lines[16] == "%i    0    0    0    0      0      0      0      0         0"
    assert lines[22] == "*  2001  8  8  0  0  0.00000000"
    assert (
        lines[23]
        == "PG01 -11044.805800 -10475.672350  21929.418200    189.163300 18 18 18 219"
    )
    assert (
        lines[24]
        == "EP    55   55   55     222  1234567 -1234567  5999999      -30       21 -1230000"
    )
    assert (
        lines[25]
        == "VG01  20298.880364 -18462.044804   1381.387685     -4.534317 14 14 14 191"
    )
    assert (
        lines[26]
        == "EV    22   22   22     111  1234567  1234567  1234567  1234567  1234567  1234567"
    )
    assert (
        lines[27]
        == "PG02 -12593.593500  10170.327650 -20354.534400    -55.976000 18 18 18 219     M"
    )


def test_SP3_to_string_long_coordinate_system():
    read = SP3.from_file(fixture("issue1014-days-increment.sp3"))
    assert read.header.coordinate_system == "ITRF14"
    text = read.to_string()
    first = text.splitlines()[0]
    assert first == "#dV2021 12 30 23  0  0.00000000      62   SLR ITRF14 FIT JCET"
    assert SP3.from_str(text).header.coordinate_system == "ITRF14"


def test_SP3_to_string_recomputes_derived_header_fields():
    read = SP3.from_file(fixture("issue1327-136-sats.sp3"))
    assert (read.header.gps_week, read.header.mjd) == (0, 0)
    text = read.to_string()
    assert (
        text.splitlines()[1]
        == "## 2302 172800.00000000   300.00000000 60360 0.0000000000000"
    )
    assert (read.header.gps_week, read.header.mjd) == (0, 0)

    sp3 = two_satellite_file()
    header = sp3.header
    header.number_of_epochs = 7
    header.start_epoch = gps_t0() + 86400.0
    sp3.header = header
    text = sp3.to_string()
    assert (
        text.splitlines()[0]
        == "#dV2024 11 10  0  0  0.00000000       2 ORBIT IGS20 EXT BRHE"
    )
    assert sp3.header.number_of_epochs == 7


def test_SP3_to_string_absent_record():
    sp3 = two_satellite_file()
    sp3.set_record("G01", 1, SP3Record([1.0e7, 2.0e7, 3.0e7]))
    text = sp3.to_string()
    lines = text.splitlines()
    g02 = [l for l in lines if l[1:4] == "G02"]
    assert g02 == [
        "PG02      0.000000      0.000000      0.000000 999999.999999",
        "VG02      0.000000      0.000000      0.000000 999999.999999",
        "PG02      0.000000      0.000000      0.000000 999999.999999",
        "VG02      0.000000      0.000000      0.000000 999999.999999",
    ]
    assert "PG01  10000.000000  20000.000000  30000.000000 999999.999999" in lines
    assert "VG01      0.000000      0.000000      0.000000 999999.999999" in lines

    back = SP3.from_str(text)
    assert back.record("G02", 0) is None
    g01 = back.record("G01", 1)
    assert (g01.clock, g01.velocity, g01.clock_rate) == (None, None, None)


def test_SP3_to_string_standard_deviations():
    sp3 = two_satellite_file(include_velocity=False)
    header = sp3.header
    header.position_base = 1.25
    header.clock_base = 1.025
    sp3.header = header
    record = SP3Record(
        [1.0e7, 2.0e7, 3.0e7],
        clock=1e-4,
        position_std=(0.010, 1.25**18 * 1e-3, math.inf),
        clock_std=1.025**219 * 1e-12,
        clock_event=True,
        orbit_predicted=True,
    )
    sp3.set_record("G01", 0, record)
    text = sp3.to_string()
    # 10 mm lies between 1.25^10 = 9.31 mm and 1.25^11 = 11.64 mm.
    assert (
        "PG01  10000.000000  20000.000000  30000.000000    100.000000 11 18 99 219 E    P"
        in text.splitlines()
    ), text


def test_SP3_to_file(tmp_path):
    sp3 = two_satellite_file()
    path = tmp_path / "out.sp3"
    sp3.to_file(str(path))
    assert path.read_text() == sp3.to_string()
    reread = SP3.from_file(str(path))
    assert reread.to_string() == SP3.from_str(sp3.to_string()).to_string()
    assert header_fields(reread.header) == header_fields(
        SP3.from_str(sp3.to_string()).header
    )

    with pytest.raises(bh.BraheError) as excinfo:
        sp3.to_file(str(tmp_path / "missing" / "out.sp3"))
    assert str(excinfo.value).startswith("SP3: ")


def test_SP3_to_string_comment_length():
    text = two_satellite_file(comments=["C" * 77]).to_string()
    back = SP3.from_str(text)
    assert back.header.comments[0] == "C" * 77

    msg = write_error(two_satellite_file(comments=["C" * 78]))
    assert "comment 1 is longer than 77 characters" in msg


def test_SP3_to_string_standard_deviation_without_base():
    sp3 = two_satellite_file()
    sp3.set_record(
        "G02", 1, SP3Record([1.0e7, 2.0e7, 3.0e7], position_std=[None, 0.01, None])
    )
    msg = write_error(sp3)
    assert "G02 at epoch 1" in msg
    assert "no position/velocity base" in msg

    sp3 = two_satellite_file()
    header = sp3.header
    header.position_base = 1.25
    sp3.header = header
    sp3.set_record(
        "G02",
        0,
        SP3Record(
            [1.0e7, 2.0e7, 3.0e7],
            position_std=[0.01] * 3,
            clock_rate_std=1e-16,
        ),
    )
    assert "no clock base" in write_error(sp3)


def test_SP3_to_string_unmapped_time_system():
    sp3 = two_satellite_file()
    header = sp3.header
    header.time_system = SP3TimeSystem.GLO
    sp3.header = header
    assert "time system GLO" in write_error(sp3)


def test_SP3_to_string_invalid_tables():
    off_grid = two_satellite_file()
    header = off_grid.header
    header.epoch_interval = 950.0
    off_grid.header = header
    assert "is not on the 950 s grid" in write_error(off_grid)

    bad_interval = two_satellite_file()
    header = bad_interval.header
    header.epoch_interval = 0.0
    bad_interval.header = header
    assert "epoch interval" in write_error(bad_interval)

    bad_accuracies = two_satellite_file()
    header = bad_accuracies.header
    header.accuracies = header.accuracies + [None]
    bad_accuracies.header = header
    assert "3 accuracies for 2 satellites" in write_error(bad_accuracies)


def test_SP3_to_string_header_label_errors():
    sp3 = two_satellite_file()
    header = sp3.header
    header.coordinate_system = ""
    sp3.header = header
    assert "coordinate system" in write_error(sp3)

    sp3 = two_satellite_file()
    header = sp3.header
    header.data_used = "u U"
    sp3.header = header
    assert "data used" in write_error(sp3)

    sp3 = two_satellite_file()
    header = sp3.header
    header.orbit_type = "X" * 30
    sp3.header = header
    assert "longer than 80 characters" in write_error(sp3)

    sp3 = two_satellite_file()
    header = sp3.header
    header.comments = ["first\nsecond"]
    sp3.header = header
    assert "line break" in write_error(sp3)


def test_SP3_to_string_value_errors():
    sp3 = two_satellite_file()
    sp3.set_record("G02", 0, SP3Record([1.0e13, 0.0, 0.0]))
    assert "does not fit" in write_error(sp3)

    sp3 = two_satellite_file()
    sp3.set_record("G02", 0, SP3Record([math.nan, 0.0, 0.0]))
    assert "does not fit" in write_error(sp3)

    sp3 = two_satellite_file()
    header = sp3.header
    header.accuracies = [math.inf, None]
    sp3.header = header
    assert "accuracy" in write_error(sp3)

    sp3 = two_satellite_file()
    header = sp3.header
    header.position_base = 1.25
    sp3.header = header
    sp3.set_record(
        "G02", 0, SP3Record([1.0e7, 2.0e7, 3.0e7], position_std=[-1.0, None, None])
    )
    assert "standard deviation" in write_error(sp3)


def test_SP3_to_string_epoch_second_rounding():
    sp3 = SP3(["G01"], 30.0)
    t0 = gps_t0() + 59.999_999_999_9
    sp3.push_epoch(t0)
    text = sp3.to_string()
    assert "\n*  2024 11 10  0  1  0.00000000\n" in text
    assert text.startswith("#dV2024 11 10  0  1  0.00000000")
