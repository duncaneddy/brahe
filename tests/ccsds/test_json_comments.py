"""Tests for CCSDS JSON comment fidelity — parity with Rust tests in src/ccsds/json.rs."""

import brahe  # noqa: F401
from brahe.ccsds import CDM, OEM, OMM, OPM


def _asset(path):
    """Read a CCSDS fixture; the same files back the Rust tests."""
    with open(path) as handle:
        return handle.read()


def test_oem_json_round_trip_preserves_comments(eop):
    """Mirror of test_oem_json_round_trip_preserves_comments in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExampleWithHeaderComment.txt")
    before = oem.to_dict()
    assert before["header"]["comments"]
    assert before["segments"][0]["comments"]

    after = OEM.from_str(oem.to_string("json")).to_dict()
    assert after["header"]["comments"] == before["header"]["comments"]
    assert after["segments"][0]["comments"] == before["segments"][0]["comments"]


def test_omm_json_round_trip_preserves_comments(eop):
    """Mirror of test_omm_json_round_trip_preserves_comments in Rust."""
    omm = OMM.from_str(_asset("test_assets/ccsds/omm/OMM-section-comments.txt"))
    reparsed = OMM.from_str(omm.to_string("json"))

    # KVN is the reference encoding; JSON must not lose anything against it.
    assert reparsed.to_string("kvn") == omm.to_string("kvn")


def test_opm_json_round_trip_preserves_comments(eop):
    """Mirror of test_opm_json_round_trip_preserves_comments in Rust."""
    opm = OPM.from_str(_asset("test_assets/ccsds/opm/OPM-section-comments.txt"))
    reparsed = OPM.from_str(opm.to_string("json"))

    assert reparsed.to_dict()["header"]["comments"] == ["header comment"]
    assert reparsed.maneuvers[0].comments == ["first maneuver comment"]
    assert reparsed.to_string("kvn") == opm.to_string("kvn")


def test_cdm_json_round_trip_preserves_comments(eop):
    """Mirror of test_cdm_json_round_trip_preserves_comments in Rust."""
    cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample2.txt")
    written = cdm.to_string("json")

    reparsed = CDM.from_str(written)
    kvn = reparsed.to_string("kvn")
    assert "COMMENT Relative Metadata/Data" in kvn
    assert "COMMENT Object1 Metadata" in kvn
    assert "COMMENT Object2 Metadata" in kvn


def test_oem_json_keeps_each_covariance_comment_with_its_own_block(eop):
    """Mirror of test_oem_json_keeps_each_covariance_comment_with_its_own_block in Rust."""
    # EPOCH delimits one covariance from the next, so comments emitted ahead of
    # it were flushed into the preceding block.
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    kvn = OEM.from_str(oem.to_string("json")).to_string("kvn")

    assert kvn == oem.to_string("kvn")


def test_cdm_json_round_trip_preserves_data_section_comments(eop):
    """Mirror of test_cdm_json_round_trip_preserves_data_section_comments in Rust."""
    # KVN cannot separate the Data comment from the first sub-block comment,
    # but JSON keeps them apart, so this level survives a JSON round trip.
    cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample2.txt")
    written = cdm.to_string("json")
    assert '"comments"' in written

    # Every comment the KVN form carries is still present after the JSON trip.
    kvn = CDM.from_str(written).to_string("kvn")
    for comment in [
        "COMMENT Relative Metadata/Data",
        "COMMENT Object1 Metadata",
        "COMMENT Object1 State Vector",
        "COMMENT Object2 Metadata",
    ]:
        assert comment in kvn, f"JSON round trip dropped {comment!r}"
