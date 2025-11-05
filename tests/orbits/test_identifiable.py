"""Tests for Identifiable trait implementation on propagators."""

import brahe as bh
import uuid


# Test data
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


class TestSGPPropagatorIdentifiable:
    """Test Identifiable trait implementation for SGPPropagator."""

    def test_sgp_with_name(self):
        """Test with_name method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        prop = prop.with_name("Test Satellite")

        assert prop.get_name() == "Test Satellite"
        assert prop.get_id() is None
        assert prop.get_uuid() is None

    def test_sgp_with_id(self):
        """Test with_id method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        prop = prop.with_id(12345)

        assert prop.get_id() == 12345
        assert prop.get_name() is None
        assert prop.get_uuid() is None

    def test_sgp_with_uuid(self):
        """Test with_uuid method."""
        test_uuid = str(uuid.uuid4())
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        prop = prop.with_uuid(test_uuid)

        assert prop.get_uuid() == test_uuid
        assert prop.get_name() is None
        assert prop.get_id() is None

    def test_sgp_with_new_uuid(self):
        """Test with_new_uuid method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        prop = prop.with_new_uuid()

        assert prop.get_uuid() is not None
        assert prop.get_name() is None
        assert prop.get_id() is None

        # Verify it's a valid UUID
        uuid.UUID(prop.get_uuid())

    def test_sgp_with_identity(self):
        """Test with_identity method."""
        test_uuid = str(uuid.uuid4())
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        prop = prop.with_identity("Satellite A", test_uuid, 999)

        assert prop.get_name() == "Satellite A"
        assert prop.get_id() == 999
        assert prop.get_uuid() == test_uuid

    def test_sgp_set_name(self):
        """Test set_name method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)

        prop.set_name("Test Name")
        assert prop.get_name() == "Test Name"

        prop.set_name(None)
        assert prop.get_name() is None

    def test_sgp_set_id(self):
        """Test set_id method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)

        prop.set_id(42)
        assert prop.get_id() == 42

        prop.set_id(None)
        assert prop.get_id() is None

    def test_sgp_generate_uuid(self):
        """Test generate_uuid method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)

        assert prop.get_uuid() is None

        prop.generate_uuid()
        uuid1 = prop.get_uuid()
        assert uuid1 is not None

        # Generate another and verify it's different
        prop.generate_uuid()
        uuid2 = prop.get_uuid()
        assert uuid2 is not None
        assert uuid1 != uuid2

    def test_sgp_set_identity(self):
        """Test set_identity method."""
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        test_uuid = str(uuid.uuid4())

        prop.set_identity("Updated Name", test_uuid, 777)

        assert prop.get_name() == "Updated Name"
        assert prop.get_id() == 777
        assert prop.get_uuid() == test_uuid

        # Clear all
        prop.set_identity(None, None, None)
        assert prop.get_name() is None
        assert prop.get_id() is None
        assert prop.get_uuid() is None

    def test_sgp_chaining(self):
        """Test method chaining."""
        test_uuid = str(uuid.uuid4())
        prop = bh.SGPPropagator.from_tle(ISS_LINE1, ISS_LINE2)
        prop = prop.with_name("Chained Satellite").with_id(123).with_uuid(test_uuid)

        assert prop.get_name() == "Chained Satellite"
        assert prop.get_id() == 123
        assert prop.get_uuid() == test_uuid


class TestKeplerianPropagatorIdentifiable:
    """Test Identifiable trait implementation for KeplerianPropagator."""

    def setup_method(self):
        """Set up test propagator."""
        import numpy as np

        self.epoch = bh.Epoch.from_datetime(
            2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC
        )
        self.elements = np.array([7000e3, 0.01, 97.8, 15.0, 45.0, 60.0])

    def test_keplerian_with_name(self):
        """Test with_name method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        prop = prop.with_name("My Orbit")

        assert prop.get_name() == "My Orbit"
        assert prop.get_id() is None
        assert prop.get_uuid() is None

    def test_keplerian_with_id(self):
        """Test with_id method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        prop = prop.with_id(54321)

        assert prop.get_id() == 54321
        assert prop.get_name() is None
        assert prop.get_uuid() is None

    def test_keplerian_with_uuid(self):
        """Test with_uuid method."""
        test_uuid = str(uuid.uuid4())
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        prop = prop.with_uuid(test_uuid)

        assert prop.get_uuid() == test_uuid
        assert prop.get_name() is None
        assert prop.get_id() is None

    def test_keplerian_with_new_uuid(self):
        """Test with_new_uuid method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        prop = prop.with_new_uuid()

        assert prop.get_uuid() is not None
        assert prop.get_name() is None
        assert prop.get_id() is None

        # Verify it's a valid UUID
        uuid.UUID(prop.get_uuid())

    def test_keplerian_with_identity(self):
        """Test with_identity method."""
        test_uuid = str(uuid.uuid4())
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        prop = prop.with_identity("Orbit X", test_uuid, 888)

        assert prop.get_name() == "Orbit X"
        assert prop.get_id() == 888
        assert prop.get_uuid() == test_uuid

    def test_keplerian_set_name(self):
        """Test set_name method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )

        prop.set_name("Name 1")
        assert prop.get_name() == "Name 1"

        prop.set_name(None)
        assert prop.get_name() is None

    def test_keplerian_set_id(self):
        """Test set_id method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )

        prop.set_id(100)
        assert prop.get_id() == 100

        prop.set_id(None)
        assert prop.get_id() is None

    def test_keplerian_generate_uuid(self):
        """Test generate_uuid method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )

        assert prop.get_uuid() is None

        prop.generate_uuid()
        uuid1 = prop.get_uuid()
        assert uuid1 is not None

        # Generate another and verify it's different
        prop.generate_uuid()
        uuid2 = prop.get_uuid()
        assert uuid2 is not None
        assert uuid1 != uuid2

    def test_keplerian_set_identity(self):
        """Test set_identity method."""
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        test_uuid = str(uuid.uuid4())

        prop.set_identity("ID Test", test_uuid, 555)

        assert prop.get_name() == "ID Test"
        assert prop.get_id() == 555
        assert prop.get_uuid() == test_uuid

        # Clear all
        prop.set_identity(None, None, None)
        assert prop.get_name() is None
        assert prop.get_id() is None
        assert prop.get_uuid() is None

    def test_keplerian_chaining(self):
        """Test method chaining."""
        test_uuid = str(uuid.uuid4())
        prop = bh.KeplerianPropagator.from_keplerian(
            self.epoch, self.elements, bh.AngleFormat.DEGREES, 60.0
        )
        prop = prop.with_name("Chained Orbit").with_id(999).with_uuid(test_uuid)

        assert prop.get_name() == "Chained Orbit"
        assert prop.get_id() == 999
        assert prop.get_uuid() == test_uuid
