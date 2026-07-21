"""Tests for the Horizons SPK Python bindings."""

import pytest

import brahe as bh


class TestHorizonsConstruction:
    def test_client_default(self):
        assert bh.datasets.horizons.HorizonsClient() is not None

    def test_client_with_base_url(self):
        assert (
            bh.datasets.horizons.HorizonsClient(base_url="https://example.test")
            is not None
        )

    def test_request_for_spkid(self):
        t0 = bh.Epoch.from_datetime(2015, 12, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)
        t1 = bh.Epoch.from_datetime(2016, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)
        req = bh.datasets.horizons.HorizonsSPKRequest.for_spkid(2000001, t0, t1)
        assert req.command == "DES=2000001;"
        assert req.center == "500@0"
        assert req.with_center("10").center == "10"


@pytest.mark.integration
class TestHorizonsIntegration:
    def test_get_and_load_ceres_spk(self):
        t0 = bh.Epoch.from_datetime(2016, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)
        t1 = bh.Epoch.from_datetime(2016, 1, 3, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)
        client = bh.datasets.horizons.HorizonsClient()
        req = bh.datasets.horizons.HorizonsSPKRequest.for_spkid(2000001, t0, t1)
        resp = client.get_spk(req)
        assert resp.path.endswith(".bsp")
        assert resp.bytes()[:7] == b"DAF/SPK"
        resp.load()  # must not raise
