"""Tests for CCSDS XML writing — parity with Rust tests in src/ccsds/xml/writer.rs."""

import brahe  # noqa: F401
from brahe.ccsds import CDM, OEM, OMM, OPM

# Free text exercising every character that terminates XML markup.
MARKUP = 'R&D <ops> "quoted"'

# The same text once escaped for element content.
MARKUP_ESCAPED = 'R&amp;D &lt;ops&gt; "quoted"'


def _with_markup_originator(path):
    """Read a KVN fixture and replace its ORIGINATOR with XML-hostile text."""
    lines = []
    with open(path) as f:
        source = f.read()
    for line in source.splitlines():
        if line.split("=")[0].strip() == "ORIGINATOR":
            lines.append(f"ORIGINATOR = {MARKUP}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


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
