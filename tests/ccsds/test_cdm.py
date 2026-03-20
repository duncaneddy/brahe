"""Tests for CCSDS Conjunction Data Message (CDM) support.

Mirrors Rust CDM tests for Python parity.
"""

import numpy as np
import pytest

import brahe as bh
from brahe.ccsds import CDM, CDMObject, CDMRTNCovariance, CDMStateVector


class TestCDMKVNParsing:
    """Test CDM parsing from KVN format."""

    def test_parse_example1(self):
        """Parse CDMExample1.txt — minimal v1.0 CDM."""
        cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.txt")

        # Header
        assert cdm.format_version == 1.0
        assert cdm.originator == "JSPOC"
        assert cdm.message_id == "201113719185"
        assert cdm.message_for is None

        # Relative metadata
        assert cdm.miss_distance == 715.0
        assert cdm.collision_probability is None

        # Object 1
        assert cdm.object1_name == "SATELLITE A"
        assert cdm.object1_designator == "12345"
        assert cdm.object1_ref_frame == "EME2000"

        # State vector (converted from km → m)
        s1 = cdm.object1_state
        assert pytest.approx(s1[0], abs=0.01) == 2570097.065
        assert pytest.approx(s1[1], abs=0.01) == 2244654.904
        assert pytest.approx(s1[2], abs=0.01) == 6281497.978
        assert pytest.approx(s1[3], abs=0.0001) == 4418.769571
        assert pytest.approx(s1[4], abs=0.0001) == 4833.547743
        assert pytest.approx(s1[5], abs=0.0001) == -3526.774282

        # Covariance (already in m²)
        cov1 = cdm.object1_covariance
        assert pytest.approx(cov1[0][0], rel=1e-6) == 4.142e01
        assert pytest.approx(cov1[1][0], rel=1e-6) == -8.579e00
        assert pytest.approx(cov1[1][1], rel=1e-6) == 2.533e03

        # Object 2
        assert cdm.object2_name == "FENGYUN 1C DEB"
        assert cdm.object2_designator == "30337"

        s2 = cdm.object2_state
        assert pytest.approx(s2[0], abs=0.01) == 2569540.800

    def test_parse_example2_extended_cov(self):
        """Parse CDMExample2.txt — full v1.0 with extended covariance."""
        cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample2.txt")

        assert cdm.message_for == "SATELLITE A"
        assert pytest.approx(cdm.collision_probability, rel=1e-6) == 4.835e-05
        assert cdm.collision_probability_method == "FOSTER-1992"
        assert pytest.approx(cdm.relative_speed) == 14762.0

    def test_parse_issue942_maneuverable_na(self):
        """CDMExample_issue942.txt — MANEUVERABLE=N/A."""
        cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample_issue942.txt")
        # Should parse without error
        assert cdm.miss_distance > 0

    def test_parse_alfano01(self):
        """AlfanoTestCase01.cdm — Alfano test case for Pc validation."""
        cdm = CDM.from_file("test_assets/ccsds/cdm/AlfanoTestCase01.cdm")
        assert cdm.miss_distance > 0
        s1 = cdm.object1_state
        assert abs(s1[0]) > 1.0  # Non-zero position

    def test_parse_real_world(self):
        """ION_SCV8_vs_STARLINK_1233.txt — real-world CDM."""
        cdm = CDM.from_file("test_assets/ccsds/cdm/ION_SCV8_vs_STARLINK_1233.txt")
        assert cdm.miss_distance > 0


class TestCDMXMLParsing:
    """Test CDM parsing from XML format."""

    def test_parse_example1_xml(self):
        """Parse CDMExample1.xml."""
        cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.xml")

        assert cdm.format_version == 1.0
        assert cdm.originator == "JSPOC"
        assert cdm.message_for == "SATELLITE A"
        assert cdm.miss_distance == 715.0

        assert cdm.object1_name == "SATELLITE A"
        s1 = cdm.object1_state
        assert pytest.approx(s1[0], abs=0.01) == 2570097.065


class TestCDMRoundTrip:
    """Test CDM parse → write → re-parse round-trips."""

    def test_kvn_round_trip(self):
        """Parse KVN → write KVN → re-parse → compare."""
        cdm1 = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.txt")
        kvn = cdm1.to_string("KVN")
        cdm2 = CDM.from_str(kvn)

        assert cdm1.originator == cdm2.originator
        assert cdm1.message_id == cdm2.message_id
        assert pytest.approx(cdm1.miss_distance, abs=1e-6) == cdm2.miss_distance

        for i in range(6):
            assert (
                pytest.approx(cdm1.object1_state[i], abs=0.01) == cdm2.object1_state[i]
            )
            assert (
                pytest.approx(cdm1.object2_state[i], abs=0.01) == cdm2.object2_state[i]
            )

    def test_xml_round_trip(self):
        """Parse XML → write XML → re-parse → compare."""
        cdm1 = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.xml")
        xml = cdm1.to_string("XML")
        cdm2 = CDM.from_str(xml)

        assert cdm1.originator == cdm2.originator
        assert pytest.approx(cdm1.miss_distance, abs=1e-6) == cdm2.miss_distance

    def test_json_round_trip(self):
        """Parse KVN → write JSON → re-parse → compare."""
        cdm1 = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.txt")
        json_str = cdm1.to_string("JSON")
        cdm2 = CDM.from_str(json_str)

        assert cdm1.originator == cdm2.originator
        assert pytest.approx(cdm1.miss_distance, abs=1e-6) == cdm2.miss_distance
        for i in range(6):
            assert (
                pytest.approx(cdm1.object1_state[i], abs=0.01) == cdm2.object1_state[i]
            )

    def test_kvn_to_xml_cross_format(self):
        """Parse KVN → write XML → re-parse → compare."""
        cdm_kvn = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.txt")
        xml = cdm_kvn.to_string("XML")
        cdm_xml = CDM.from_str(xml)

        assert cdm_kvn.originator == cdm_xml.originator
        assert pytest.approx(cdm_kvn.miss_distance, abs=1e-6) == cdm_xml.miss_distance
        for i in range(6):
            assert (
                pytest.approx(cdm_kvn.object1_state[i], abs=0.01)
                == cdm_xml.object1_state[i]
            )


class TestCDMErrorCases:
    """Test CDM error handling."""

    def test_missing_tca(self):
        """CDM-missing-TCA.txt should fail with clear error."""
        with pytest.raises(Exception, match="TCA"):
            CDM.from_file("test_assets/ccsds/cdm/CDM-missing-TCA.txt")

    def test_missing_obj2_state(self):
        """CDM-missing-object2-state-vector.txt should fail."""
        with pytest.raises(Exception):
            CDM.from_file("test_assets/ccsds/cdm/CDM-missing-object2-state-vector.txt")


class TestCDMRepr:
    """Test CDM string representation."""

    def test_repr(self):
        cdm = CDM.from_file("test_assets/ccsds/cdm/CDMExample1.txt")
        r = repr(cdm)
        assert "CDM" in r
        assert "SATELLITE A" in r
        assert "FENGYUN 1C DEB" in r


@pytest.fixture
def sample_cdm():
    """Create a sample CDM for testing."""
    sv1 = CDMStateVector(
        position=[7000e3, 0.0, 0.0],
        velocity=[0.0, 7500.0, 0.0],
    )
    sv2 = CDMStateVector(
        position=[7001e3, 0.0, 0.0],
        velocity=[0.0, -7500.0, 0.0],
    )
    cov1 = CDMRTNCovariance(matrix=np.eye(6).tolist())
    cov2 = CDMRTNCovariance(matrix=np.eye(6).tolist())

    obj1 = CDMObject(
        designator="12345",
        catalog_name="SATCAT",
        name="SAT A",
        international_designator="2020-001A",
        ephemeris_name="NONE",
        covariance_method="CALCULATED",
        maneuverable="YES",
        ref_frame="EME2000",
        state_vector=sv1,
        rtn_covariance=cov1,
    )
    obj2 = CDMObject(
        designator="67890",
        catalog_name="SATCAT",
        name="SAT B",
        international_designator="2021-002B",
        ephemeris_name="NONE",
        covariance_method="CALCULATED",
        maneuverable="NO",
        ref_frame="EME2000",
        state_vector=sv2,
        rtn_covariance=cov2,
    )

    tca = bh.Epoch.from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    return CDM(
        originator="TEST_ORG",
        message_id="MSG001",
        tca=tca,
        miss_distance=715.0,
        object1=obj1,
        object2=obj2,
    )


class TestCDMCreation:
    """Test CDM programmatic creation."""

    def test_cdm_new(self, sample_cdm):
        """Create CDM from scratch, verify all mandatory fields."""
        cdm = sample_cdm

        assert cdm.originator == "TEST_ORG"
        assert cdm.message_id == "MSG001"
        assert cdm.format_version == 1.0
        assert cdm.miss_distance == 715.0

        assert cdm.object1_name == "SAT A"
        assert cdm.object1_designator == "12345"
        assert cdm.object1_ref_frame == "EME2000"

        s1 = cdm.object1_state
        assert s1[0] == 7000e3
        assert s1[4] == 7500.0

        assert cdm.object2_name == "SAT B"
        assert cdm.object2_designator == "67890"

        s2 = cdm.object2_state
        assert s2[0] == 7001e3
        assert s2[4] == -7500.0

        assert cdm.collision_probability is None

    def test_cdm_create_kvn_round_trip(self, sample_cdm):
        """Create → write KVN → re-parse → verify equality."""
        kvn = sample_cdm.to_string("KVN")
        cdm2 = CDM.from_str(kvn)

        assert cdm2.originator == "TEST_ORG"
        assert cdm2.message_id == "MSG001"
        assert pytest.approx(cdm2.miss_distance, abs=1e-6) == 715.0
        assert cdm2.object1_name == "SAT A"
        assert cdm2.object2_name == "SAT B"

        for i in range(6):
            assert (
                pytest.approx(sample_cdm.object1_state[i], abs=0.01)
                == cdm2.object1_state[i]
            )
            assert (
                pytest.approx(sample_cdm.object2_state[i], abs=0.01)
                == cdm2.object2_state[i]
            )

    def test_cdm_create_xml_round_trip(self, sample_cdm):
        """Create → write XML → re-parse → verify equality."""
        xml = sample_cdm.to_string("XML")
        cdm2 = CDM.from_str(xml)

        assert cdm2.originator == "TEST_ORG"
        assert pytest.approx(cdm2.miss_distance, abs=1e-6) == 715.0
        assert cdm2.object1_name == "SAT A"

        for i in range(6):
            assert (
                pytest.approx(sample_cdm.object1_state[i], abs=0.01)
                == cdm2.object1_state[i]
            )

    def test_cdm_create_json_round_trip(self, sample_cdm):
        """Create → write JSON → re-parse → verify equality."""
        json_str = sample_cdm.to_string("JSON")
        cdm2 = CDM.from_str(json_str)

        assert cdm2.originator == "TEST_ORG"
        assert pytest.approx(cdm2.miss_distance, abs=1e-6) == 715.0
        assert cdm2.object1_name == "SAT A"

        for i in range(6):
            assert (
                pytest.approx(sample_cdm.object1_state[i], abs=0.01)
                == cdm2.object1_state[i]
            )

    def test_cdm_optional_fields(self, sample_cdm):
        """Set collision_probability and method after construction."""
        cdm = sample_cdm
        assert cdm.collision_probability is None
        assert cdm.collision_probability_method is None

        cdm.collision_probability = 4.835e-05
        cdm.collision_probability_method = "FOSTER-1992"

        assert pytest.approx(cdm.collision_probability, rel=1e-6) == 4.835e-05
        assert cdm.collision_probability_method == "FOSTER-1992"

        # Round-trip with optional fields
        kvn = cdm.to_string("KVN")
        cdm2 = CDM.from_str(kvn)
        assert pytest.approx(cdm2.collision_probability, rel=1e-6) == 4.835e-05
        assert cdm2.collision_probability_method == "FOSTER-1992"


class TestCDMSubObjects:
    """Test CDM sub-object (CDMStateVector, CDMRTNCovariance, CDMObject) properties."""

    def test_state_vector_properties(self):
        """Verify CDMStateVector getters."""
        sv = CDMStateVector(
            position=[1.0, 2.0, 3.0],
            velocity=[4.0, 5.0, 6.0],
        )
        assert sv.position == pytest.approx([1.0, 2.0, 3.0])
        assert sv.velocity == pytest.approx([4.0, 5.0, 6.0])
        assert "CDMStateVector" in repr(sv)

    def test_state_vector_from_numpy(self):
        """CDMStateVector accepts numpy arrays."""
        sv = CDMStateVector(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
        )
        assert sv.position == pytest.approx([1.0, 2.0, 3.0])

    def test_rtn_covariance_round_trip(self):
        """Construct covariance → read back matrix."""
        mat = [[float(i * 6 + j) for j in range(6)] for i in range(6)]
        cov = CDMRTNCovariance(matrix=mat)
        result = cov.matrix
        for i in range(6):
            for j in range(6):
                assert result[i][j] == mat[i][j]
        assert "CDMRTNCovariance" in repr(cov)

    def test_rtn_covariance_from_numpy(self):
        """CDMRTNCovariance accepts numpy arrays."""
        mat = np.eye(6) * 42.0
        cov = CDMRTNCovariance(matrix=mat)
        assert cov.matrix[0][0] == 42.0
        assert cov.matrix[1][1] == 42.0
        assert cov.matrix[0][1] == 0.0

    def test_cdm_object_properties(self):
        """Verify CDMObject getters."""
        sv = CDMStateVector([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
        cov = CDMRTNCovariance(np.eye(6).tolist())
        obj = CDMObject(
            designator="12345",
            catalog_name="SATCAT",
            name="SAT A",
            international_designator="2020-001A",
            ephemeris_name="NONE",
            covariance_method="CALCULATED",
            maneuverable="YES",
            ref_frame="EME2000",
            state_vector=sv,
            rtn_covariance=cov,
        )
        assert obj.designator == "12345"
        assert obj.catalog_name == "SATCAT"
        assert obj.name == "SAT A"
        assert obj.international_designator == "2020-001A"
        assert obj.ephemeris_name == "NONE"
        assert obj.covariance_method == "CALCULATED"
        assert obj.maneuverable == "YES"
        assert obj.ref_frame == "EME2000"
        assert obj.state[0] == 7000e3
        assert obj.covariance[0][0] == 1.0
        assert "CDMObject" in repr(obj)
