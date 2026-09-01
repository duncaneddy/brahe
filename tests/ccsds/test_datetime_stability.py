"""Tests for CCSDS datetime write stability — parity with Rust tests in src/ccsds/common.rs."""

import brahe
from brahe.ccsds import CDM, OEM, OMM, OPM

CASES = [
    (OEM, "test_assets/ccsds/oem/OEMExample1.txt"),
    (OMM, "test_assets/ccsds/omm/OMMExample2.txt"),
    (OPM, "test_assets/ccsds/opm/OPMExample1.txt"),
    (CDM, "test_assets/ccsds/cdm/CDMExample1.txt"),
]


def test_kvn_message_round_trip_is_a_fixed_point(eop):
    """Mirror of test_kvn_message_round_trip_is_a_fixed_point in Rust."""
    for cls, path in CASES:
        written = cls.from_file(path).to_string("kvn")
        current = written
        for _ in range(3):
            current = cls.from_str(current).to_string("kvn")
            assert current == written, f"{path} drifts across write/read cycles"


def test_oem_epochs_survive_every_output_format(eop):
    """Mirror of test_oem_epochs_survive_every_output_format in Rust."""
    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

    def epochs(message):
        found = []
        for seg in message.segments:
            found.append(seg.start_time)
            found.append(seg.stop_time)
            found.extend(sv.epoch for sv in seg.states)
        return [str(e) for e in found]

    for fmt in ["kvn", "xml", "json"]:
        reparsed = OEM.from_str(oem.to_string(fmt))
        assert epochs(reparsed) == epochs(oem), (
            f"OEM epochs shifted across a {fmt} round trip"
        )


def test_format_ccsds_datetime_writes_a_whole_second_as_whole(eop):
    """Mirror of test_format_ccsds_datetime_writes_a_whole_second_as_whole in Rust."""
    # An epoch built on a whole second is written with no fractional
    # nanoseconds; this used to emit ".000000001".
    epoch = brahe.Epoch.from_datetime(1996, 11, 4, 17, 22, 31.0, 0.0, brahe.UTC)
    assert epoch.to_datetime()[6] == 0.0

    oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")
    oem.segments[0].states[0].epoch = epoch
    written = [
        line for line in oem.to_string("kvn").splitlines() if "1996-11-04" in line
    ]
    assert all(".000000001" not in line for line in written)


def test_ccsds_datetime_round_trip_is_a_fixed_point(eop):
    """Mirror of test_ccsds_datetime_round_trip_is_a_fixed_point in Rust."""
    # Writing and re-reading must converge; each generation used to add a
    # nanosecond without bound.
    for cls, path in CASES:
        current = cls.from_file(path).to_string("kvn")
        first = current
        for generation in range(4):
            current = cls.from_str(current).to_string("kvn")
            assert current == first, f"{path} drifted at generation {generation}"
