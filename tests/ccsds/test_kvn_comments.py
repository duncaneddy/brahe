"""Tests for CCSDS KVN comment attribution — parity with Rust tests in src/ccsds/kvn/parser.rs."""

import brahe  # noqa: F401
from brahe.ccsds import CDM, OMM, OPM


def _asset(path):
    """Read a CCSDS fixture; the same files back the Rust tests."""
    with open(path) as handle:
        return handle.read()


# Every one of these carries a COMMENT ahead of TCA and a comment introducing
# each object block, the shape CCSDS 508.0-B-1 uses in its own examples.
FIXTURES = [
    "CDMExample1.txt",
    "CDMExample2.txt",
    "CDMExample3.txt",
    "CDMExample4.txt",
    "CDMExample_issue_940.txt",
]


def test_cdm_comment_buckets_survive_a_kvn_round_trip(eop):
    """Mirror of test_cdm_comment_buckets_survive_a_kvn_round_trip in Rust."""
    for name in FIXTURES:
        cdm = CDM.from_file(f"test_assets/ccsds/cdm/{name}")
        written = cdm.to_string("kvn")
        rewritten = CDM.from_str(written).to_string("kvn")
        assert rewritten == written, f"{name} does not round-trip"


def test_cdm_section_comments_are_not_hoisted_into_the_header(eop):
    """A comment introducing the relative metadata section is not a header comment."""
    written = CDM.from_file("test_assets/ccsds/cdm/CDMExample2.txt").to_string("kvn")

    lines = [line.strip() for line in written.splitlines()]
    tca_index = next(i for i, line in enumerate(lines) if line.startswith("TCA"))
    creation_index = next(
        i for i, line in enumerate(lines) if line.startswith("CREATION_DATE")
    )

    # "Relative Metadata/Data" is emitted with the relative metadata block,
    # after the header keywords and before TCA.
    relative_comment = next(
        i for i, line in enumerate(lines) if line == "COMMENT Relative Metadata/Data"
    )
    assert creation_index < relative_comment < tca_index


def test_parse_omm_attributes_comments_to_the_block_they_introduce(eop):
    """Mirror of test_parse_omm_attributes_comments_to_the_block_they_introduce in Rust."""
    omm = OMM.from_str(_asset("test_assets/ccsds/omm/OMM-section-comments.txt"))

    assert omm.to_dict()["metadata"]["object_name"] == "GOES 9"
    written = omm.to_string("kvn")
    assert OMM.from_str(written).to_string("kvn") == written


def test_parse_opm_attributes_comments_to_the_block_they_introduce(eop):
    """Mirror of test_parse_opm_attributes_comments_to_the_block_they_introduce in Rust."""
    opm = OPM.from_str(_asset("test_assets/ccsds/opm/OPM-section-comments.txt"))

    assert len(opm.maneuvers) == 2
    assert opm.maneuvers[0].comments == ["first maneuver comment"]
    assert opm.maneuvers[1].comments == ["second maneuver comment"]

    written = opm.to_string("kvn")
    assert OPM.from_str(written).to_string("kvn") == written
