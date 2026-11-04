"""Tests mirroring src/clients/igs/file_name.rs."""

import pytest

import brahe as bh


def gps(y, mo, d, h):
    return bh.Epoch(y, mo, d, h, 0, 0.0, 0.0, time_system=bh.GPS)


LONG = bh.IGSNameStyle.LONG
SHORT = bh.IGSNameStyle.SHORT
FIN = bh.IGSSolutionType.FINAL
RAP = bh.IGSSolutionType.RAPID
ULT = bh.IGSSolutionType.ULTRA_RAPID
ORB = bh.IGSContentType.ORBIT
CLK_C = bh.IGSContentType.CLOCK
ERP_C = bh.IGSContentType.EARTH_ROTATION
NONE = bh.IGSCompression.NONE
GZIP = bh.IGSCompression.GZIP
UNIX = bh.IGSCompression.UNIX_COMPRESS


IDENTITY_CASES = [
    (
        "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz",
        LONG,
        "IGS",
        FIN,
        gps(2024, 11, 10, 0),
        2340,
    ),
    (
        "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
        LONG,
        "COD",
        FIN,
        gps(2024, 11, 10, 0),
        2340,
    ),
    (
        "IGS0OPSULT_20243150600_02D_15M_ORB.SP3.gz",
        LONG,
        "IGS",
        ULT,
        gps(2024, 11, 10, 6),
        2340,
    ),
    (
        "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
        LONG,
        "COD",
        FIN,
        gps(2024, 11, 10, 0),
        2340,
    ),
    (
        "IGS0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        LONG,
        "IGS",
        FIN,
        gps(2024, 11, 10, 0),
        2340,
    ),
    ("igs21000.sp3.Z", SHORT, "IGS", FIN, gps(2020, 4, 5, 0), 2100),
    ("igr21003.clk.Z", SHORT, "IGS", RAP, gps(2020, 4, 8, 0), 2100),
    ("igu21000_06.sp3.Z", SHORT, "IGS", ULT, gps(2020, 4, 5, 6), 2100),
    ("igs21007.erp.Z", SHORT, "IGS", FIN, gps(2020, 4, 5, 0), 2100),
    ("cod21000.sp3", SHORT, "COD", FIN, gps(2020, 4, 5, 0), 2100),
]


@pytest.mark.parametrize("name_str,style,center,solution,start,week", IDENTITY_CASES)
def test_igs_product_file_name_parse_identity(
    name_str, style, center, solution, start, week
):
    name = bh.IGSProductFileName.parse(name_str)
    assert name.name_style == style
    assert name.center == center
    assert name.solution == solution
    assert name.start_epoch == start
    assert name.gps_week == week
    assert name.start_epoch.gps_date()[0] == week
    assert str(name) == name_str
    if style == LONG:
        assert name.version == 0
        assert name.campaign == "OPS"
    else:
        assert name.version is None
        assert name.campaign is None


PRODUCT_CASES = [
    (
        "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz",
        86400.0,
        900.0,
        ORB,
        bh.IGSFileFormat.SP3,
        GZIP,
    ),
    (
        "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
        86400.0,
        300.0,
        ORB,
        bh.IGSFileFormat.SP3,
        GZIP,
    ),
    (
        "IGS0OPSULT_20243150600_02D_15M_ORB.SP3.gz",
        172800.0,
        900.0,
        ORB,
        bh.IGSFileFormat.SP3,
        GZIP,
    ),
    (
        "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
        86400.0,
        30.0,
        CLK_C,
        bh.IGSFileFormat.CLK,
        GZIP,
    ),
    (
        "IGS0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        604800.0,
        86400.0,
        ERP_C,
        bh.IGSFileFormat.ERP,
        GZIP,
    ),
    ("igs21000.sp3.Z", 86400.0, None, ORB, bh.IGSFileFormat.SP3, UNIX),
    ("igr21003.clk.Z", 86400.0, None, CLK_C, bh.IGSFileFormat.CLK, UNIX),
    ("igu21000_06.sp3.Z", 172800.0, None, ORB, bh.IGSFileFormat.SP3, UNIX),
    ("igs21007.erp.Z", None, None, ERP_C, bh.IGSFileFormat.ERP, UNIX),
    ("cod21000.sp3", 86400.0, None, ORB, bh.IGSFileFormat.SP3, NONE),
]


@pytest.mark.parametrize(
    "name_str,span,sampling,content,fmt,compression", PRODUCT_CASES
)
def test_igs_product_file_name_parse_product(
    name_str, span, sampling, content, fmt, compression
):
    name = bh.IGSProductFileName.parse(name_str)
    assert name.span == span
    assert name.sampling == sampling
    assert name.content == content
    assert name.format == fmt
    assert name.compression == compression
    assert name.is_sp3 == (fmt == bh.IGSFileFormat.SP3)


@pytest.mark.parametrize(
    "name_str",
    [
        "readme.txt",
        "IGS0OPSFIN_2024315_01D_15M_ORB.SP3",
        "igs2100.sp3",
        "igs21008.sp3",
        "igs21000",
        "igs21000.",
        "IGS0OPSFIN_20243670000_01D_15M_ORB.SP3",
        "IGS0OPSFIN_20253660000_01D_15M_ORB.SP3",
        "IGS0OPSFIN_20243152400_01D_15M_ORB.SP3",
        "IGS0OPSFIN_20243150060_01D_15M_ORB.SP3.gz",
        "IGS0OPSFIN_19791000000_01D_15M_ORB.SP3",
        "IGSXOPSFIN_20243150000_01D_15M_ORB.SP3",
        "IGS0OPSFIN_20243150000_D_15M_ORB.SP3",
        "igs21000_06.sp3",
        "igu21000_24.sp3",
        "igu21000_6.sp3",
    ],
)
def test_igs_product_file_name_parse_errors(name_str):
    with pytest.raises(bh.BraheError):
        bh.IGSProductFileName.parse(name_str)


@pytest.mark.parametrize(
    "name_str",
    [
        "IGS0OPSFIN_2024A150000_01D_15M_ORB.SP3",
        "IGS0OPSFIN_20243150000__15M_ORB.SP3",
        "IGS0OPSFIN_20243150000_01D__ORB.SP3",
        "IGS-OPSFIN_20243150000_01D_15M_ORB.SP3",
        "IGS0OPSFIN_20243150000_01D_15M_OR.SP3",
        "IGS0OPSFIN_20243150000_01D_15M_OR-.SP3",
        "ig-21000.sp3",
    ],
)
def test_igs_product_file_name_malformed_fields(name_str):
    with pytest.raises(bh.BraheError):
        bh.IGSProductFileName.parse(name_str)


@pytest.mark.parametrize(
    "name_str,center,solution",
    [
        ("cor21000.sp3.Z", "COD", RAP),
        ("cou21000_12.sp3.Z", "COD", ULT),
        ("esr21000.sp3.Z", "ESA", RAP),
        ("esu21000_18.sp3.Z", "ESA", ULT),
        ("emr21000.sp3.Z", "EMR", FIN),
        ("emp21000.sp3.Z", "EMR", RAP),
        ("gfr21000.sp3.Z", "GFZ", RAP),
        ("gfu21000_00.sp3.Z", "GFZ", ULT),
        ("jpl21000.sp3.Z", "JPL", FIN),
        ("igl21000.sp3.Z", "IGL", FIN),
        ("igv21000_06.sp3.Z", "IGV", ULT),
        ("siu21000_02.sp3.Z", "SIO", ULT),
        ("sir21003.erp.Z", "SIO", RAP),
        ("cof21000.eph.Z", "COF", FIN),
    ],
)
def test_igs_product_file_name_short_center_codes(name_str, center, solution):
    name = bh.IGSProductFileName.parse(name_str)
    assert name.center == center
    assert name.solution == solution


@pytest.mark.parametrize(
    "code,expected",
    [
        ("01S", 1.0),
        ("15M", 900.0),
        ("06H", 21600.0),
        ("01D", 86400.0),
        ("02W", 1209600.0),
        ("01Y", 31536000.0),
        ("00U", None),
    ],
)
def test_igs_product_file_name_duration_units(code, expected):
    name = bh.IGSProductFileName.parse(f"IGS0OPSFIN_20243150000_{code}_{code}_ORB.SP3")
    assert name.span == expected
    assert name.sampling == expected


@pytest.mark.parametrize(
    "name_str,content,fmt",
    [
        ("igs21000.clk_30s.Z", bh.IGSContentType.CLOCK, bh.IGSFileFormat.CLK),
        ("igs21000.clk_05s.Z", bh.IGSContentType.CLOCK, bh.IGSFileFormat.CLK),
        ("igs21000.snx.Z", bh.IGSContentType.SOLUTION, bh.IGSFileFormat.SNX),
        ("igs21000.sum", bh.IGSContentType.SUMMARY, bh.IGSFileFormat.SUM),
        ("cof21000.tro.Z", bh.IGSContentType.TROPOSPHERE, bh.IGSFileFormat.TRO),
        (
            "igs21000.xyz",
            bh.IGSContentType.parse("XYZ"),
            bh.IGSFileFormat.parse("XYZ"),
        ),
    ],
)
def test_igs_product_file_name_short_extensions(name_str, content, fmt):
    name = bh.IGSProductFileName.parse(name_str)
    assert name.content == content
    assert name.format == fmt
    assert not name.is_sp3
    assert str(name) == name_str


def test_igs_product_file_name_long_other_codes():
    name = bh.IGSProductFileName.parse("WUM0MGXFIN_20243150000_01D_05M_ABC.XYZ")
    assert name.campaign == "MGX"
    assert name.content == bh.IGSContentType.parse("ABC")
    assert name.content.code == "ABC"
    assert name.format == bh.IGSFileFormat.parse("XYZ")
    assert name.compression == NONE


def test_igs_product_file_name_long_week_boundaries():
    sunday = bh.IGSProductFileName.parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3")
    saturday = bh.IGSProductFileName.parse("IGS0OPSFIN_20243142359_01D_15M_ORB.SP3")
    assert sunday.gps_week == 2340
    assert saturday.gps_week == 2339
    assert saturday.start_epoch.gps_date()[0] == 2339


def test_igs_product_file_name_long_leap_day():
    name = bh.IGSProductFileName.parse("IGS0OPSFIN_20243660000_01D_15M_ORB.SP3")
    assert name.start_epoch == gps(2024, 12, 31, 0)


def test_igs_product_file_name_igv_ultra_rapid_span():
    name = bh.IGSProductFileName.parse("igv21000_06.sp3")
    assert name.start_epoch == gps(2020, 4, 5, 6)
    assert name.span == 172800.0


def test_igs_product_file_name_siu_ultra_rapid():
    name = bh.IGSProductFileName.parse("siu21000_02.sp3.Z")
    assert name.start_epoch == gps(2020, 4, 5, 2)
    assert name.span == 172800.0
    assert name.compression == UNIX


def test_igs_product_file_name_sir_earth_rotation():
    name = bh.IGSProductFileName.parse("sir21003.erp.Z")
    assert name.content == bh.IGSContentType.EARTH_ROTATION
    assert name.format == bh.IGSFileFormat.ERP
    assert name.start_epoch == gps(2020, 4, 8, 0)


def test_igs_product_file_name_eph_extension_is_sp3():
    name = bh.IGSProductFileName.parse("cof21000.eph.Z")
    assert name.content == bh.IGSContentType.ORBIT
    assert name.format == bh.IGSFileFormat.SP3
    assert name.is_sp3
    assert str(name) == "cof21000.eph.Z"


def test_igs_product_file_name_compression_suffix_by_style():
    long = bh.IGSProductFileName.parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.Z")
    assert long.compression == UNIX
    assert long.format == bh.IGSFileFormat.SP3
    assert str(long) == "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.Z"

    short = bh.IGSProductFileName.parse("igs21000.sp3.gz")
    assert short.compression == GZIP
    assert short.format == bh.IGSFileFormat.SP3
    assert str(short) == "igs21000.sp3.gz"


@pytest.mark.parametrize(
    "name_str",
    [
        "igs21000é.sp3",
        "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gzé",
        "IGé0OPSFIN_20243150000_01D_15M_ORB.SP3",
    ],
)
def test_igs_product_file_name_non_ascii(name_str):
    with pytest.raises(bh.BraheError):
        bh.IGSProductFileName.parse(name_str)


def test_igs_solution_content_format_codes():
    for code in ["FIN", "RAP", "ULT", "PRD", "XYZ"]:
        assert bh.IGSSolutionType.parse(code).code == code
    for code in ["ORB", "CLK", "ERP", "SOL", "ATT", "BIA", "GIM", "TRO", "SUM", "XYZ"]:
        assert bh.IGSContentType.parse(code).code == code
    for code in ["SP3", "CLK", "ERP", "SNX", "OBX", "BIA", "INX", "TRO", "SUM", "XYZ"]:
        assert bh.IGSFileFormat.parse(code).code == code
    assert bh.IGSSolutionType.parse("fin") == bh.IGSSolutionType.FINAL
    assert bh.IGSSolutionType.parse("prd") == bh.IGSSolutionType.PREDICTED


def test_igs_enum_classattrs_and_codes():
    assert bh.IGSSolutionType.FINAL.code == "FIN"
    assert bh.IGSSolutionType.RAPID.code == "RAP"
    assert bh.IGSSolutionType.ULTRA_RAPID.code == "ULT"
    assert bh.IGSSolutionType.PREDICTED.code == "PRD"
    content_codes = {
        bh.IGSContentType.ORBIT: "ORB",
        bh.IGSContentType.CLOCK: "CLK",
        bh.IGSContentType.EARTH_ROTATION: "ERP",
        bh.IGSContentType.SOLUTION: "SOL",
        bh.IGSContentType.ATTITUDE: "ATT",
        bh.IGSContentType.BIAS: "BIA",
        bh.IGSContentType.IONOSPHERE: "GIM",
        bh.IGSContentType.TROPOSPHERE: "TRO",
        bh.IGSContentType.SUMMARY: "SUM",
    }
    for content, code in content_codes.items():
        assert content.code == code
        assert bh.IGSContentType.parse(code) == content
    format_codes = {
        bh.IGSFileFormat.SP3: "SP3",
        bh.IGSFileFormat.CLK: "CLK",
        bh.IGSFileFormat.ERP: "ERP",
        bh.IGSFileFormat.SNX: "SNX",
        bh.IGSFileFormat.OBX: "OBX",
        bh.IGSFileFormat.BIA: "BIA",
        bh.IGSFileFormat.INX: "INX",
        bh.IGSFileFormat.TRO: "TRO",
        bh.IGSFileFormat.SUM: "SUM",
    }
    for fmt, code in format_codes.items():
        assert fmt.code == code
        assert bh.IGSFileFormat.parse(code) == fmt


def test_igs_enum_str_repr_eq_hash():
    assert str(bh.IGSSolutionType.ULTRA_RAPID) == "ULT"
    assert repr(bh.IGSSolutionType.ULTRA_RAPID) == "IGSSolutionType.ULTRA_RAPID"
    assert repr(bh.IGSSolutionType.parse("abc")) == "IGSSolutionType.parse('ABC')"
    assert repr(bh.IGSContentType.EARTH_ROTATION) == "IGSContentType.EARTH_ROTATION"
    assert repr(bh.IGSContentType.parse("abc")) == "IGSContentType.parse('ABC')"
    assert repr(bh.IGSFileFormat.SP3) == "IGSFileFormat.SP3"
    assert repr(bh.IGSFileFormat.parse("abc")) == "IGSFileFormat.parse('ABC')"
    assert str(bh.IGSCompression.UNIX_COMPRESS) == "UNIX_COMPRESS"
    assert repr(bh.IGSCompression.GZIP) == "IGSCompression.GZIP"
    assert str(bh.IGSNameStyle.LONG) == "LONG"
    assert repr(bh.IGSNameStyle.SHORT) == "IGSNameStyle.SHORT"

    assert bh.IGSSolutionType.FINAL != bh.IGSSolutionType.RAPID
    assert bh.IGSCompression.NONE != bh.IGSCompression.GZIP
    assert bh.IGSNameStyle.LONG != bh.IGSNameStyle.SHORT
    assert hash(bh.IGSSolutionType.parse("fin")) == hash(bh.IGSSolutionType.FINAL)
    assert hash(bh.IGSContentType.parse("orb")) == hash(bh.IGSContentType.ORBIT)
    assert hash(bh.IGSFileFormat.parse("sp3")) == hash(bh.IGSFileFormat.SP3)
    assert (
        len({bh.IGSCompression.NONE, bh.IGSCompression.NONE, bh.IGSCompression.GZIP})
        == 2
    )
    assert len({bh.IGSNameStyle.LONG, bh.IGSNameStyle.SHORT}) == 2


def test_igs_product_file_name_str_repr_eq_hash():
    text = "igs21000.sp3.Z"
    name = bh.IGSProductFileName.parse(text)
    same = bh.IGSProductFileName.parse(text)
    other = bh.IGSProductFileName.parse("igr21003.clk.Z")
    assert repr(name) == "IGSProductFileName('igs21000.sp3.Z')"
    assert name == same
    assert name != other
    assert hash(name) == hash(same)
