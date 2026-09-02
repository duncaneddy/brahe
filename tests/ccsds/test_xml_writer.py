"""Tests for CCSDS XML writing — parity with Rust tests in src/ccsds/xml/writer.rs."""

import xml.etree.ElementTree as ET

import pytest

import brahe
from brahe.ccsds import CDM, OEM, OMM, OPM

# Free text exercising every character that terminates XML markup.
MARKUP = 'R&D <ops> "quoted"'

# The same text once escaped for element content.
MARKUP_ESCAPED = 'R&amp;D &lt;ops&gt; "quoted"'

# Free text carrying U+0001, which XML 1.0 forbids in any document.
FORBIDDEN = "GSOC\u0001LAB"


def _with_originator(path, value):
    """Read a KVN fixture and replace its ORIGINATOR with the given text."""
    lines = []
    with open(path) as f:
        source = f.read()
    for line in source.splitlines():
        if line.split("=")[0].strip() == "ORIGINATOR":
            lines.append(f"ORIGINATOR = {value}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _with_markup_originator(path):
    """Read a KVN fixture and replace its ORIGINATOR with XML-hostile text."""
    return _with_originator(path, MARKUP)


def test_write_oem_xml_escapes_free_text(eop):
    """Mirror of test_write_oem_xml_escapes_free_text in Rust."""
    oem = OEM.from_str(_with_markup_originator("test_assets/ccsds/oem/OEMExample1.txt"))

    xml = oem.to_string("xml")
    assert f"<ORIGINATOR>{MARKUP_ESCAPED}</ORIGINATOR>" in xml
    assert OEM.from_str(xml).originator == MARKUP


def test_write_omm_xml_escapes_free_text(eop):
    """Mirror of test_write_omm_xml_escapes_free_text_and_user_defined in Rust."""
    omm = OMM.from_str(_with_markup_originator("test_assets/ccsds/omm/OMMExample2.txt"))

    xml = omm.to_string("xml")
    assert f"<ORIGINATOR>{MARKUP_ESCAPED}</ORIGINATOR>" in xml
    assert OMM.from_str(xml).originator == MARKUP


def test_write_opm_xml_escapes_free_text(eop):
    """Mirror of test_write_opm_xml_escapes_free_text in Rust."""
    opm = OPM.from_str(_with_markup_originator("test_assets/ccsds/opm/OPMExample1.txt"))

    xml = opm.to_string("xml")
    assert f"<ORIGINATOR>{MARKUP_ESCAPED}</ORIGINATOR>" in xml
    assert OPM.from_str(xml).originator == MARKUP


def test_write_cdm_xml_escapes_free_text(eop):
    """Mirror of test_write_cdm_xml_escapes_free_text in Rust."""
    cdm = CDM.from_str(_with_markup_originator("test_assets/ccsds/cdm/CDMExample1.txt"))

    xml = cdm.to_string("xml")
    assert f"<ORIGINATOR>{MARKUP_ESCAPED}</ORIGINATOR>" in xml
    assert CDM.from_str(xml).originator == MARKUP


@pytest.mark.parametrize(
    "cls,path,message_type",
    [
        (OEM, "test_assets/ccsds/oem/OEMExample1.txt", "OEM"),
        (OMM, "test_assets/ccsds/omm/OMMExample2.txt", "OMM"),
        (OPM, "test_assets/ccsds/opm/OPMExample1.txt", "OPM"),
        (CDM, "test_assets/ccsds/cdm/CDMExample1.txt", "CDM"),
    ],
)
def test_write_xml_rejects_forbidden_characters(eop, cls, path, message_type):
    """Mirror of test_write_<msg>_xml_rejects_forbidden_characters in Rust."""
    message = cls.from_str(_with_originator(path, FORBIDDEN))

    with pytest.raises(brahe.BraheError) as excinfo:
        message.to_string("xml")

    assert message_type in str(excinfo.value)
    assert "ORIGINATOR" in str(excinfo.value)
    assert "U+0001" in str(excinfo.value)


def test_write_omm_xml_rejects_forbidden_user_defined_value(eop):
    """Mirror of the user-defined half of the Rust OMM rejection test."""
    source = _with_originator("test_assets/ccsds/omm/OMMExample2.txt", "NOAA/USA")
    source += f"USER_DEFINED_EARTH_MODEL = {FORBIDDEN}\n"

    with pytest.raises(brahe.BraheError) as excinfo:
        OMM.from_str(source).to_string("xml")

    assert "USER_DEFINED_EARTH_MODEL" in str(excinfo.value)
    assert "U+0001" in str(excinfo.value)


def test_write_omm_xml_rejects_forbidden_user_defined_key(eop):
    """Mirror of the user-defined key half of the Rust OMM rejection test."""
    source = _with_originator("test_assets/ccsds/omm/OMMExample2.txt", "NOAA/USA")
    source += f"USER_DEFINED_{FORBIDDEN} = value\n"

    with pytest.raises(brahe.BraheError) as excinfo:
        OMM.from_str(source).to_string("xml")

    assert "USER_DEFINED_GSOC" in str(excinfo.value)
    assert "U+0001" in str(excinfo.value)


def test_write_xml_follows_the_xml_char_production(eop):
    """Mirror of test_write_xml_follows_the_xml_char_production in Rust."""
    path = "test_assets/ccsds/opm/OPMExample1.txt"

    # A line break cannot travel through a KVN value, so the Rust test covers
    # the #xA and #xD bounds.
    for c in [
        "\t",
        " ",
        "\ud7ff",
        "\ue000",
        "\ufffd",
        "\U00010000",
        "\U0001fffe",
        "\U0010ffff",
    ]:
        opm = OPM.from_str(_with_originator(path, "GSOC" + c + "LAB"))
        xml = opm.to_string("xml")
        assert f"<ORIGINATOR>GSOC{c}LAB</ORIGINATOR>" in xml
        ET.fromstring(xml)

    for c in ["\x00", "\x08", "\x0b", "\x0c", "\x0e", "\x1f", "\ufffe", "\uffff"]:
        opm = OPM.from_str(_with_originator(path, "GSOC" + c + "LAB"))
        with pytest.raises(brahe.BraheError) as excinfo:
            opm.to_string("xml")
        assert f"U+{ord(c):04X}" in str(excinfo.value)
