"""Tests for shared CCSDS header handling — parity with Rust tests in src/ccsds/common.rs."""

import json
import re

import pytest

import brahe as bh
from brahe.ccsds import OEM, OMM, OPM

MESSAGES = [
    (OEM, "test_assets/ccsds/oem/OEMExample1.txt"),
    (OMM, "test_assets/ccsds/omm/OMMExample1.txt"),
    (OPM, "test_assets/ccsds/opm/OPMExample1.txt"),
]


def _kvn_with_creation_date(path, value):
    """Read a KVN fixture and replace its CREATION_DATE value."""
    with open(path) as handle:
        content = handle.read()
    return re.sub(
        r"^CREATION_DATE[ \t]*=.*$",
        f"CREATION_DATE = {value}",
        content,
        flags=re.MULTILINE,
    )


@pytest.mark.parametrize("cls,path", MESSAGES)
@pytest.mark.parametrize("value", ["", "   "])
def test_parse_optional_ccsds_datetime_empty(eop, cls, path, value):
    """Mirror of test_parse_optional_ccsds_datetime_empty in Rust."""
    assert cls.from_str(_kvn_with_creation_date(path, value)).creation_date is None


@pytest.mark.parametrize("cls,path", MESSAGES)
def test_parse_optional_ccsds_datetime_value(eop, cls, path):
    """Mirror of test_parse_optional_ccsds_datetime_value in Rust."""
    msg = cls.from_str(_kvn_with_creation_date(path, "2024-03-01T12:00:00"))
    assert msg.creation_date == bh.Epoch.from_datetime(
        2024, 3, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC
    )

    with pytest.raises(Exception, match="not-a-date"):
        cls.from_str(_kvn_with_creation_date(path, "not-a-date"))


@pytest.mark.parametrize("cls,path", MESSAGES)
def test_format_optional_ccsds_datetime_in(eop, cls, path):
    """Mirror of test_format_optional_ccsds_datetime_in in Rust."""
    msg = cls.from_file(path)
    msg.creation_date = None
    assert re.search(
        r"^CREATION_DATE\s*=\s*$", msg.to_string("KVN"), flags=re.MULTILINE
    )

    msg.creation_date = bh.Epoch.from_datetime(
        2024, 3, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC
    )
    assert re.search(
        r"^CREATION_DATE\s*=\s*2024-03-01T12:00:00\.000\s*$",
        msg.to_string("KVN"),
        flags=re.MULTILINE,
    )


@pytest.mark.parametrize("cls,path", MESSAGES)
def test_json_null_creation_date_reads_as_none(eop, cls, path):
    """Mirror of test_json_null_creation_date_reads_as_none in Rust."""
    data = json.loads(cls.from_file(path).to_json_string(uppercase_keys=True))
    data["header"]["CREATION_DATE"] = None

    assert cls.from_str(json.dumps(data)).creation_date is None
