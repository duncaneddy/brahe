"""Tests for SpaceTrack Python bindings."""

import pytest
import brahe as bh


class TestSpaceTrackClientClass:
    """Tests for SpaceTrackClient class structure and attributes."""

    def test_spacetrack_client_class_exists(self):
        """Test SpaceTrackClient class is available."""
        assert hasattr(bh, "SpaceTrackClient")
        assert callable(bh.SpaceTrackClient)

    def test_client_has_expected_methods(self):
        """Test that SpaceTrackClient class has all expected methods."""
        # Check class attributes without instantiation (avoids auth)
        methods = dir(bh.SpaceTrackClient)

        # BasicSpaceData methods
        assert "gp" in methods
        assert "satcat" in methods
        assert "tle" in methods
        assert "decay" in methods
        assert "tip" in methods
        assert "cdm_public" in methods
        assert "boxscore" in methods
        assert "launch_site" in methods
        assert "satcat_change" in methods
        assert "satcat_debut" in methods
        assert "announcement" in methods

        # Utility methods
        assert "gp_as_propagators" in methods
        assert "generic_request" in methods

        # Properties
        assert "base_url" in methods
        assert "is_authenticated" in methods


class TestSpaceTrackQueryTypes:
    """Tests for SpaceTrack query builder types."""

    def test_spacetrack_value_from_int(self):
        """Test SpaceTrackValue creation from integer."""
        val = bh.SpaceTrackValue.from_int(12345)
        assert "12345" in str(val)

    def test_spacetrack_value_from_float(self):
        """Test SpaceTrackValue creation from float."""
        val = bh.SpaceTrackValue.from_float(3.14159)
        assert "3.14159" in str(val)

    def test_spacetrack_value_from_string(self):
        """Test SpaceTrackValue creation from string."""
        val = bh.SpaceTrackValue.from_string("ISS")
        assert "ISS" in str(val)

    def test_spacetrack_value_from_bool(self):
        """Test SpaceTrackValue creation from boolean."""
        val = bh.SpaceTrackValue.from_bool(True)
        assert "true" in str(val)

    def test_spacetrack_value_from_epoch(self):
        """Test SpaceTrackValue creation from Epoch."""
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        val = bh.SpaceTrackValue.from_epoch(epoch)
        assert "2024-01-01" in str(val)

    def test_spacetrack_order_ascending(self):
        """Test SpaceTrackOrder ascending constant."""
        assert bh.SpaceTrackOrder.ASCENDING is not None

    def test_spacetrack_order_descending(self):
        """Test SpaceTrackOrder descending constant."""
        assert bh.SpaceTrackOrder.DESCENDING is not None

    def test_spacetrack_predicate_norad_cat_id(self):
        """Test SpaceTrackPredicate.norad_cat_id() returns builder."""
        builder = bh.SpaceTrackPredicate.norad_cat_id()
        assert isinstance(builder, bh.SpaceTrackPredicateBuilder)

    def test_spacetrack_predicate_builder_eq(self):
        """Test SpaceTrackPredicateBuilder.eq() returns predicate."""
        builder = bh.SpaceTrackPredicate.norad_cat_id()
        pred = builder.eq(bh.SpaceTrackValue.from_int(25544))
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_gt(self):
        """Test SpaceTrackPredicateBuilder.gt() returns predicate."""
        builder = bh.SpaceTrackPredicate.eccentricity()
        pred = builder.gt(bh.SpaceTrackValue.from_float(0.1))
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_lt(self):
        """Test SpaceTrackPredicateBuilder.lt() returns predicate."""
        builder = bh.SpaceTrackPredicate.inclination()
        pred = builder.lt(bh.SpaceTrackValue.from_float(90.0))
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_between(self):
        """Test SpaceTrackPredicateBuilder.between() returns predicate."""
        builder = bh.SpaceTrackPredicate.mean_motion()
        pred = builder.between(
            bh.SpaceTrackValue.from_float(14.0), bh.SpaceTrackValue.from_float(16.0)
        )
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_like(self):
        """Test SpaceTrackPredicateBuilder.like() returns predicate."""
        builder = bh.SpaceTrackPredicate.object_name()
        pred = builder.like("ISS%")
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_starts_with(self):
        """Test SpaceTrackPredicateBuilder.starts_with() returns predicate."""
        builder = bh.SpaceTrackPredicate.object_name()
        pred = builder.starts_with("STARLINK")
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_contains(self):
        """Test SpaceTrackPredicateBuilder.contains() returns predicate."""
        builder = bh.SpaceTrackPredicate.object_name()
        pred = builder.contains("LINK")
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_builder_is_null(self):
        """Test SpaceTrackPredicateBuilder.is_null() returns predicate."""
        builder = bh.SpaceTrackPredicate.decay_date()
        pred = builder.is_null()
        assert isinstance(pred, bh.SpaceTrackPredicate)

    def test_spacetrack_predicate_generic_field(self):
        """Test SpaceTrackPredicate.field() for custom fields."""
        builder = bh.SpaceTrackPredicate.field("CUSTOM_FIELD")
        assert isinstance(builder, bh.SpaceTrackPredicateBuilder)

    def test_spacetrack_query_gp(self):
        """Test SpaceTrackQuery.gp() creates query."""
        query = bh.SpaceTrackQuery.gp()
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_satcat(self):
        """Test SpaceTrackQuery.satcat() creates query."""
        query = bh.SpaceTrackQuery.satcat()
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_tle(self):
        """Test SpaceTrackQuery.tle() creates query."""
        query = bh.SpaceTrackQuery.tle()
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_filter(self):
        """Test SpaceTrackQuery.filter() returns query."""
        query = bh.SpaceTrackQuery.gp()
        pred = bh.SpaceTrackPredicate.norad_cat_id().eq(
            bh.SpaceTrackValue.from_int(25544)
        )
        filtered = query.filter(pred)
        assert isinstance(filtered, bh.SpaceTrackQuery)

    def test_spacetrack_query_limit(self):
        """Test SpaceTrackQuery.limit() returns query."""
        query = bh.SpaceTrackQuery.gp().limit(10)
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_order_by(self):
        """Test SpaceTrackQuery.order_by() returns query."""
        query = bh.SpaceTrackQuery.gp().order_by(
            bh.SpaceTrackPredicate.epoch(), bh.SpaceTrackOrder.DESCENDING
        )
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_order_by_asc(self):
        """Test SpaceTrackQuery.order_by_asc() returns query."""
        query = bh.SpaceTrackQuery.gp().order_by_asc(bh.SpaceTrackPredicate.epoch())
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_order_by_desc(self):
        """Test SpaceTrackQuery.order_by_desc() returns query."""
        query = bh.SpaceTrackQuery.gp().order_by_desc(bh.SpaceTrackPredicate.epoch())
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_chained(self):
        """Test chained query building."""
        query = (
            bh.SpaceTrackQuery.gp()
            .filter(
                bh.SpaceTrackPredicate.norad_cat_id().eq(
                    bh.SpaceTrackValue.from_int(25544)
                )
            )
            .filter(
                bh.SpaceTrackPredicate.eccentricity().lt(
                    bh.SpaceTrackValue.from_float(0.1)
                )
            )
            .order_by_desc(bh.SpaceTrackPredicate.epoch())
            .limit(100)
        )
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_with_epoch_range(self):
        """Test query with Epoch-based range filter."""
        start = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        end = bh.Epoch.from_datetime(2024, 12, 31, 23, 59, 59.0, 0.0, bh.TimeSystem.UTC)
        query = bh.SpaceTrackQuery.gp().filter(
            bh.SpaceTrackPredicate.epoch().between(
                bh.SpaceTrackValue.from_epoch(start), bh.SpaceTrackValue.from_epoch(end)
            )
        )
        assert isinstance(query, bh.SpaceTrackQuery)

    def test_spacetrack_query_custom_controller_class(self):
        """Test creating query with custom controller and class."""
        query = bh.SpaceTrackQuery("customcontroller", "customclass")
        assert isinstance(query, bh.SpaceTrackQuery)


class TestSpaceTrackClientCreation:
    """Tests for SpaceTrackClient instantiation (requires network)."""

    @pytest.mark.ci
    @pytest.mark.xfail(reason="SpaceTrack authentication required")
    def test_client_creation_fails_with_invalid_credentials(self):
        """Test client creation fails with invalid credentials."""
        # SpaceTrack authenticates immediately on creation
        with pytest.raises(RuntimeError):
            bh.SpaceTrackClient("invalid_user", "invalid_password")

    @pytest.mark.ci
    @pytest.mark.xfail(reason="Custom URL may not respond")
    def test_client_creation_with_custom_base_url(self):
        """Test client creation with custom base URL fails gracefully."""
        with pytest.raises(RuntimeError):
            bh.SpaceTrackClient(
                "test_user", "test_password", base_url="https://custom.example.com/"
            )

    @pytest.mark.ci
    @pytest.mark.xfail(reason="Test server requires valid credentials")
    def test_client_creation_with_test_server(self):
        """Test client creation pointing to test server."""
        # Test server still requires authentication
        with pytest.raises(RuntimeError):
            bh.SpaceTrackClient(
                "test_user",
                "test_password",
                base_url="https://for-testing-only.space-track.org/",
            )


# ============================================================================
# Integration tests that require valid SpaceTrack credentials
# These are marked with @pytest.mark.ci and @pytest.mark.xfail for CI/CD
# ============================================================================


@pytest.fixture
def spacetrack_client():
    """Create authenticated client using environment variables.

    Requires TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables.
    Uses the test server for automated testing.
    """
    import os

    user = os.environ.get("TEST_SPACETRACK_USER", "")
    password = os.environ.get("TEST_SPACETRACK_PASS", "")
    if not user or not password:
        pytest.skip(
            "SpaceTrack credentials not set (TEST_SPACETRACK_USER, TEST_SPACETRACK_PASS)"
        )
    return bh.SpaceTrackClient(
        user, password, base_url="https://for-testing-only.space-track.org/"
    )


@pytest.mark.ci
class TestSpaceTrackIntegration:
    """Integration tests requiring SpaceTrack account.

    To run these tests locally with valid credentials:
    - Set TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables
    """

    def test_gp_query_iss(self, spacetrack_client):
        """Test GP query for ISS (NORAD ID 25544)."""
        records = spacetrack_client.gp(norad_cat_id=25544, limit=1)

        assert isinstance(records, list)
        assert len(records) == 1

        record = records[0]
        # Records are GPRecord objects with attribute access
        assert hasattr(record, "norad_cat_id")
        assert record.norad_cat_id == 25544

    def test_gp_query_with_orderby(self, spacetrack_client):
        """Test GP query with ordering."""
        records = spacetrack_client.gp(
            norad_cat_id=25544,
            orderby="EPOCH desc",
            limit=5,  # Most recent first
        )

        assert isinstance(records, list)
        assert len(records) <= 5

    def test_authenticated_client_properties(self, spacetrack_client):
        """Test that authenticated client has correct properties."""
        assert spacetrack_client.base_url == "https://for-testing-only.space-track.org/"
        assert spacetrack_client.is_authenticated is True

    def test_satcat_query(self, spacetrack_client):
        """Test SATCAT query for ISS."""
        records = spacetrack_client.satcat(norad_cat_id=25544, limit=1)

        assert isinstance(records, list)
        assert len(records) == 1

        record = records[0]
        # Records are SATCATRecord objects with attribute access
        assert hasattr(record, "object_name")

    def test_gp_as_propagators(self, spacetrack_client):
        """Test converting GP records to SGP propagators."""
        propagators = spacetrack_client.gp_as_propagators(
            step_size=60.0, norad_cat_id=25544, limit=1
        )

        assert isinstance(propagators, list)
        assert len(propagators) == 1

        prop = propagators[0]
        assert hasattr(prop, "propagate_to")
        assert hasattr(prop, "current_state")

    def test_tle_query(self, spacetrack_client):
        """Test TLE query for ISS."""
        records = spacetrack_client.tle(norad_cat_id=25544, limit=1)

        assert isinstance(records, list)
        if len(records) > 0:
            record = records[0]
            # Records are TLERecord objects with attribute access
            assert hasattr(record, "norad_cat_id")

    def test_boxscore_query(self, spacetrack_client):
        """Test BOXSCORE query."""
        records = spacetrack_client.boxscore()

        assert isinstance(records, list)
        if len(records) > 0:
            record = records[0]
            # Records are BoxscoreRecord objects with attribute access
            assert hasattr(record, "country")

    def test_launch_site_query(self, spacetrack_client):
        """Test LAUNCH_SITE query."""
        records = spacetrack_client.launch_site()

        assert isinstance(records, list)
        if len(records) > 0:
            record = records[0]
            # Records are LaunchSiteRecord objects with attribute access
            assert hasattr(record, "site_code")

    def test_generic_request(self, spacetrack_client):
        """Test generic request method."""
        response = spacetrack_client.generic_request(
            controller="basicspacedata",
            class_name="gp",
            predicates={"NORAD_CAT_ID": "25544", "limit": "1"},
        )

        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.ci
class TestSpaceTrackPropagatorIntegration:
    """Integration tests for SGP propagator conversion."""

    def test_propagator_propagation(self, spacetrack_client):
        """Test that propagators from GP records can propagate."""
        import numpy as np

        propagators = spacetrack_client.gp_as_propagators(
            step_size=60.0, norad_cat_id=25544, limit=1
        )

        assert len(propagators) == 1
        prop = propagators[0]

        # Propagate to current time
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        prop.propagate_to(epoch)
        state = prop.current_state()

        assert len(state) == 6

        # Verify ISS-like altitude (300-500 km)
        r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        altitude = r - bh.R_EARTH
        assert 300e3 < altitude < 500e3
