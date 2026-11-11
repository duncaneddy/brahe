"""Tests mirroring src/clients/igs/centers.rs."""

import pytest

import brahe as bh

CODES = [
    "COD",
    "EMR",
    "ESA",
    "GFZ",
    "GOP",
    "GRG",
    "JGX",
    "JPL",
    "MIT",
    "NGS",
    "SIO",
    "USN",
    "WHU",
    "IGS",
]


@pytest.mark.parametrize("code", CODES)
def test_igs_analysis_center_code_round_trip(code):
    center = getattr(bh.IGSAnalysisCenter, code)
    assert bh.IGSAnalysisCenter.parse(code) == center
    assert bh.IGSAnalysisCenter.parse(code.lower()) == center
    assert center.code == code
    assert center.name


def test_igs_analysis_center_parse_case_insensitive():
    assert bh.IGSAnalysisCenter.parse("Cod") == bh.IGSAnalysisCenter.COD
    assert bh.IGSAnalysisCenter.COD.name == "Center for Orbit Determination in Europe"


def test_igs_analysis_center_parse_unknown():
    center = bh.IGSAnalysisCenter.parse("xyz")
    assert center.code == "XYZ"
    assert center.name is None
    assert center == bh.IGSAnalysisCenter.parse("XYZ")
    assert bh.IGSAnalysisCenter.parse("CODE").code == "CODE"


def test_igs_analysis_center_str_repr_eq_hash():
    center = bh.IGSAnalysisCenter.JPL
    assert str(center) == "JPL"
    assert repr(center) == "IGSAnalysisCenter.JPL"
    assert repr(bh.IGSAnalysisCenter.parse("igl")) == "IGSAnalysisCenter.parse('IGL')"
    assert center == bh.IGSAnalysisCenter.parse("jpl")
    assert center != bh.IGSAnalysisCenter.MIT
    assert hash(center) == hash(bh.IGSAnalysisCenter.parse("jpl"))
    assert len({center, bh.IGSAnalysisCenter.parse("jpl")}) == 1
