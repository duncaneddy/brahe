"""Tests for shared CCSDS header handling — parity with Rust tests in src/ccsds/common.rs."""

import json

import pytest

import brahe  # noqa: F401
from brahe.ccsds import OEM, OMM, OPM


@pytest.mark.parametrize(
    "cls,path",
    [
        (OEM, "test_assets/ccsds/oem/OEMExample1.txt"),
        (OMM, "test_assets/ccsds/omm/OMMExample1.txt"),
        (OPM, "test_assets/ccsds/opm/OPMExample1.txt"),
    ],
)
def test_json_null_creation_date_reads_as_none(eop, cls, path):
    """Mirror of test_json_null_creation_date_reads_as_none in Rust."""
    data = json.loads(cls.from_file(path).to_json_string(uppercase_keys=True))
    data["header"]["CREATION_DATE"] = None

    assert cls.from_str(json.dumps(data)).creation_date is None
