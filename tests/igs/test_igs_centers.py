"""Tests mirroring src/clients/igs/centers.rs."""

import brahe as bh


def test_igs_analysis_center_from_code_case_insensitive():
    center = bh.IGSAnalysisCenter.from_code("cod")
    assert center.code == "COD"
    assert center.name == "Center for Orbit Determination in Europe"
    assert bh.IGSAnalysisCenter.from_code("Cod") == center


def test_igs_analysis_center_from_code_unknown():
    assert bh.IGSAnalysisCenter.from_code("XYZ") is None
    assert bh.IGSAnalysisCenter.from_code("") is None
    assert bh.IGSAnalysisCenter.from_code("CODE") is None


def test_igs_list_analysis_centers_contents():
    codes = [center.code for center in bh.list_analysis_centers()]
    for expected in [
        "COD",
        "EMR",
        "ESA",
        "GFZ",
        "GRG",
        "JPL",
        "MIT",
        "NGS",
        "SIO",
        "USN",
        "WHU",
        "IGS",
    ]:
        assert expected in codes


def test_igs_list_analysis_centers_well_formed():
    centers = bh.list_analysis_centers()
    codes = [center.code for center in centers]
    assert len(set(codes)) == len(codes)
    for center in centers:
        assert len(center.code) == 3
        assert center.code.isupper()
        assert center.name


def test_igs_analysis_center_str_repr_eq_hash():
    center = bh.IGSAnalysisCenter.from_code("JPL")
    assert str(center) == "JPL"
    assert repr(center) == (
        "IGSAnalysisCenter(code='JPL', name='Jet Propulsion Laboratory')"
    )
    assert center == bh.IGSAnalysisCenter.from_code("jpl")
    assert center != bh.IGSAnalysisCenter.from_code("MIT")
    assert hash(center) == hash(bh.IGSAnalysisCenter.from_code("jpl"))
    assert len({center, bh.IGSAnalysisCenter.from_code("jpl")}) == 1
