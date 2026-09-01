"""Tests for CCSDS write stability and CDM data-block fidelity — parity with Rust tests."""

import brahe  # noqa: F401
from brahe.ccsds import CDM, OEM, OMM, OPM

FORMATS = ["kvn", "xml", "json"]

CASES = [
    (OEM, "test_assets/ccsds/oem/OEMExample1.txt"),
    (OMM, "test_assets/ccsds/omm/OMMExample2.txt"),
    (OPM, "test_assets/ccsds/opm/OPMExample3.txt"),
    (CDM, "test_assets/ccsds/cdm/CDMExample2.txt"),
]


def test_message_writes_are_stable_in_every_encoding(eop):
    """Mirror of test_message_writes_are_stable_in_every_encoding in Rust."""
    for cls, path in CASES:
        message = cls.from_file(path)
        for fmt in FORMATS:
            written = message.to_string(fmt)
            rewritten = cls.from_str(written).to_string(fmt)
            assert rewritten == written, f"{path} {fmt} is not stable across a reparse"


def test_cdm_data_blocks_survive_every_encoding(eop):
    """Mirror of test_cdm_data_blocks_survive_every_encoding in Rust."""
    cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample2.txt")

    # The OD and additional parameter blocks were absent from JSON output
    # entirely, so a JSON round trip dropped them.
    for fmt in FORMATS:
        kvn = CDM.from_str(cdm.to_string(fmt)).to_string("kvn")
        assert "TIME_LASTOB_START" in kvn, f"{fmt} dropped the OD parameters block"
        assert "AREA_PC" in kvn, f"{fmt} dropped the additional parameters block"
