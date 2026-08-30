"""Tests for CCSDS covariance EPOCH conformance — parity with Rust tests in src/ccsds/kvn/writer.rs."""

import brahe  # noqa: F401
from brahe.ccsds import OEM, OPM


def test_opm_covariance_block_omits_epoch(eop):
    """Mirror of test_opm_covariance_block_omits_epoch in Rust."""
    # CCSDS 502.0-B-3 table 3-3 gives the OPM covariance block only COMMENT,
    # COV_REF_FRAME, and the matrix entries. The KVN parser matches EPOCH
    # positionally, so a second assignment would land on the state vector.
    opm = OPM.from_file("test_assets/ccsds/opm/OPMExample3.txt")

    epoch_lines = [
        line
        for line in opm.to_string("kvn").splitlines()
        if line.strip().startswith("EPOCH ")
    ]
    assert len(epoch_lines) == 1


def test_oem_covariance_blocks_keep_their_epoch(eop):
    """Mirror of test_oem_covariance_blocks_keep_their_epoch in Rust."""
    # The OEM is the one message whose covariance block defines EPOCH.
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    for fmt in ["kvn", "xml"]:
        assert "EPOCH" in oem.to_string(fmt)
        assert OEM.from_str(oem.to_string(fmt)).to_string("kvn") == oem.to_string("kvn")
