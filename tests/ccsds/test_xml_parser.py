"""Tests for CCSDS XML parsing — parity with Rust tests in src/ccsds/xml/parser.rs."""

import pytest

import brahe  # noqa: F401
from brahe.ccsds import OEM, OMM, OPM


def _asset(path):
    """Read a CCSDS fixture; the same files back the Rust tests."""
    with open(path) as handle:
        return handle.read()


def test_parse_oem_xml_multiple_comments_per_block(eop):
    """Mirror of test_parse_oem_xml_multiple_comments_per_block in Rust."""
    oem = OEM.from_str(_asset("test_assets/ccsds/oem/OEM-multiple-comments.xml"))
    d = oem.to_dict()

    assert d["header"]["comments"] == ["first header comment", "second header comment"]
    assert d["segments"][0]["metadata"]["comments"] == [
        "first metadata comment",
        "second metadata comment",
    ]
    assert d["segments"][0]["comments"] == ["first data comment", "second data comment"]
    assert d["segments"][0]["num_covariances"] == 1


def test_parse_omm_xml_multiple_comments_per_block(eop):
    """Mirror of test_parse_omm_xml_multiple_comments_per_block in Rust."""
    omm = OMM.from_str(_asset("test_assets/ccsds/omm/OMM-multiple-comments.xml"))
    d = omm.to_dict()

    assert d["metadata"]["object_name"] == "GOES-9"
    assert "tle_parameters" in d
    assert "spacecraft_parameters" in d


def test_parse_opm_xml_multiple_comments_per_block(eop):
    """Mirror of test_parse_opm_xml_multiple_comments_per_block in Rust."""
    opm = OPM.from_str(_asset("test_assets/ccsds/opm/OPM-multiple-comments.xml"))

    assert opm.to_dict()["header"]["comments"] == [
        "first header comment",
        "second header comment",
    ]
    assert opm.maneuvers[0].comments == [
        "first maneuver comment",
        "second maneuver comment",
    ]


def test_parse_oem_xml_multiple_segments(eop):
    """Mirror of test_parse_oem_xml_multiple_segments in Rust."""
    oem = OEM.from_str(_asset("test_assets/ccsds/oem/OEM-two-segments.xml"))

    assert len(oem.segments) == 2
    assert oem.segments[0].num_states == 1
    assert oem.segments[1].num_states == 1
    assert oem.segments[1].states[0].position[0] == pytest.approx(-2432200.0, abs=1.0)


def test_parse_opm_xml_multiple_maneuvers(eop):
    """Mirror of test_parse_opm_xml_multiple_maneuvers in Rust."""
    opm = OPM.from_str(_asset("test_assets/ccsds/opm/OPM-two-maneuvers.xml"))

    assert len(opm.maneuvers) == 2
    assert opm.maneuvers[0].duration == pytest.approx(300.0)
    assert opm.maneuvers[1].duration == pytest.approx(150.0)
    assert opm.maneuvers[1].dv[1] == pytest.approx(2.0)


def test_parse_cdm_xml_reads_cdata_sections(eop):
    """Mirror of test_parse_cdm_xml_reads_cdata_sections in Rust."""
    from brahe.ccsds import CDM

    # A CDATA section is character data carrying markup characters unescaped.
    with open("test_assets/ccsds/cdm/CDMExample1.xml") as f:
        content = f.read()
    content = content.replace(
        "<ORIGINATOR>JSPOC</ORIGINATOR>",
        "<ORIGINATOR><![CDATA[R&D <ops>]]></ORIGINATOR>",
    )

    assert CDM.from_str(content).originator == "R&D <ops>"
