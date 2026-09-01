"""Tests for the CCSDS KVN writer — parity with Rust tests in src/ccsds/kvn/writer.rs.

CCSDS 502.0-B-3 section 7.4.8 (and 508.0-P-1.1 section 6.3.1.9) fix the order of
KVN assignments to that of the header tables, which place COMMENT immediately
after the version line and CLASSIFICATION after it.
"""

from pathlib import Path

import pytest

from brahe.ccsds import CDM, OEM, OMM, OPM

HEADER_COMMENTS = ["first header comment", "second"]
CLASSIFICATION = "public, test-data"


def _keyword(line):
    """Keyword of a KVN assignment line, or None for non-assignment lines."""
    key, sep, _ = line.partition("=")
    return key.strip() if sep else None


def _with_header_comments_and_classification(path, vers_keyword):
    """Read a KVN fixture and give its header both comments and a classification."""
    lines = [
        line
        for line in Path(path).read_text().splitlines()
        if _keyword(line) != "CLASSIFICATION"
    ]
    vers = next(i for i, line in enumerate(lines) if _keyword(line) == vers_keyword)
    inserted = [f"COMMENT {c}" for c in HEADER_COMMENTS] + [
        f"CLASSIFICATION = {CLASSIFICATION}"
    ]
    return "\n".join(lines[: vers + 1] + inserted + lines[vers + 1 :]) + "\n"


def _emitted_header_comments(written):
    """Comment text emitted ahead of CREATION_DATE, i.e. the header comments."""
    comments = []
    for line in written.splitlines():
        if _keyword(line) == "CREATION_DATE":
            break
        if line.strip().startswith("COMMENT"):
            comments.append(line.strip()[len("COMMENT") :].strip())
    return comments


def _assert_header_order(written, vers_keyword):
    """Assert the emitted header runs version line, then COMMENT, then CLASSIFICATION."""
    lines = written.splitlines()

    def index_of(predicate, label):
        for i, line in enumerate(lines):
            if predicate(line):
                return i
        pytest.fail(f"'{label}' missing from written message:\n{written}")

    vers = index_of(lambda ln: _keyword(ln) == vers_keyword, vers_keyword)
    comment = index_of(lambda ln: ln.strip().startswith("COMMENT"), "COMMENT")
    classification = index_of(
        lambda ln: _keyword(ln) == "CLASSIFICATION", "CLASSIFICATION"
    )

    assert vers < comment < classification, (
        f"expected {vers_keyword} < COMMENT < CLASSIFICATION, "
        f"got {vers} < {comment} < {classification} in:\n{written}"
    )


@pytest.mark.parametrize(
    ("cls", "vers_keyword", "fixture"),
    [
        (OEM, "CCSDS_OEM_VERS", "test_assets/ccsds/oem/OEMExample1.txt"),
        (OMM, "CCSDS_OMM_VERS", "test_assets/ccsds/omm/OMMExample2.txt"),
        (OPM, "CCSDS_OPM_VERS", "test_assets/ccsds/opm/OPMExample1.txt"),
        (CDM, "CCSDS_CDM_VERS", "test_assets/ccsds/cdm/CDMExample1.txt"),
    ],
    ids=["oem", "omm", "opm", "cdm"],
)
def test_write_header_comment_before_classification(eop, cls, vers_keyword, fixture):
    """Mirror of test_<type>_write_header_comment_before_classification in Rust."""
    content = _with_header_comments_and_classification(fixture, vers_keyword)

    written = cls.from_str(content).to_string("kvn")
    _assert_header_order(written, vers_keyword)
    assert _emitted_header_comments(written) == HEADER_COMMENTS

    # Header comments must survive a re-parse as header comments rather than
    # being absorbed into the metadata comments.
    rewritten = cls.from_str(written).to_string("kvn")
    _assert_header_order(rewritten, vers_keyword)
    assert _emitted_header_comments(rewritten) == HEADER_COMMENTS
