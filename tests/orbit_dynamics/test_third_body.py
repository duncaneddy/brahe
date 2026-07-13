"""
Tests for orbit dynamics third-body acceleration functions.

This module tests both analytical and DE-based third-body acceleration calculations.
"""

import pytest
import numpy as np
import brahe as bh


@pytest.fixture(scope="module", autouse=True)
def initialize_ephemeris():
    """Initialize DE ephemeris for all tests."""
    bh.initialize_ephemeris()


class TestAnalyticalThirdBody:
    """Tests for analytical third-body acceleration functions."""

    def test_accel_third_body_sun_returns_vector(self):
        """Test that accel_third_body_sun returns a 3D vector."""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        a = bh.accel_third_body_sun(epc, r_object)

        assert a.shape == (3,)
        assert a.dtype == np.float64
        assert np.linalg.norm(a) > 0.0

    def test_accel_third_body_moon_returns_vector(self):
        """Test that accel_third_body_moon returns a 3D vector."""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        a = bh.accel_third_body_moon(epc, r_object)

        assert a.shape == (3,)
        assert a.dtype == np.float64
        assert np.linalg.norm(a) > 0.0

    @pytest.mark.parametrize(
        "mjd_tt,rx,ry,rz,ax,ay,az",
        [
            (
                60310.0,
                4884992.30378986,
                4553508.53744864,
                1330313.60479734,
                -2.83676856237279e-07,
                2.42660636226875e-07,
                1.32048201247083e-07,
            ),
            (
                60310.0,
                2670937.8974923,
                5898362.79515022,
                2124959.71017719,
                -2.31115657850035e-07,
                4.01378977924412e-07,
                1.92039921303102e-07,
            ),
            (
                60310.0,
                38796.9774858514,
                6320698.88514676,
                2587294.93626938,
                -1.42403095448685e-07,
                4.97330766046125e-07,
                2.21999834460446e-07,
            ),
            (
                60310.0,
                -2599961.45855466,
                5760720.19357889,
                2647597.12683792,
                -3.15422631697234e-08,
                5.16014363543264e-07,
                2.17465940218504e-07,
            ),
            (
                60310.0,
                -4839229.61832879,
                4313760.58255103,
                2300338.34996557,
                8.42078268885445e-08,
                4.55276781915684e-07,
                1.79457446434353e-07,
            ),
            (
                60310.0,
                -6342536.88656784,
                2209712.29939824,
                1602811.60820791,
                1.87182166643884e-07,
                3.25221817468977e-07,
                1.14120959837502e-07,
            ),
            (
                60310.0,
                -6891477.8215365,
                -227551.810286937,
                663813.896586629,
                2.62015981019115e-07,
                1.46169083244472e-07,
                3.1584377565069e-08,
            ),
            (
                60310.0,
                -6412800.79978623,
                -2631381.70900648,
                -374371.749654303,
                2.97795813116627e-07,
                -5.47304407469416e-08,
                -5.56848123002086e-08,
            ),
            (
                60310.0,
                -4983679.01699774,
                -4645498.60891225,
                -1357188.62711648,
                2.89444343311254e-07,
                -2.4755353121528e-07,
                -1.34716105636085e-07,
            ),
            (
                60310.0,
                -2817603.71414268,
                -5972669.87274763,
                -2139313.41892538,
                2.38284972069453e-07,
                -4.03777237061026e-07,
                -1.93828712473536e-07,
            ),
            (
                60310.0,
                -234236.587976406,
                -6414628.84861909,
                -2604335.85309436,
                1.51812336070211e-07,
                -5.00147992775384e-07,
                -2.24210512692769e-07,
            ),
            (
                60310.0,
                2383524.57084058,
                -5900075.61185268,
                -2680956.18196418,
                4.27172228608481e-08,
                -5.21918073677521e-07,
                -2.21152074692176e-07,
            ),
            (
                60310.0,
                4641862.77023787,
                -4497734.59354263,
                -2354086.60315067,
                -7.27753469123527e-08,
                -4.65126279082538e-07,
                -1.84808262415947e-07,
            ),
            (
                60310.0,
                6193136.2430559,
                -2411369.56203787,
                -1669079.86356028,
                -1.77151028990413e-07,
                -3.3756567861856e-07,
                -1.20350830019883e-07,
            ),
            (
                60310.0,
                6790850.71407875,
                45505.4274329756,
                -727399.838172203,
                -2.54224503688731e-07,
                -1.580949129909e-07,
                -3.73927783617869e-08,
            ),
            (
                60310.0,
                6333183.86841522,
                2494761.03873549,
                327102.634966258,
                -2.91770539678173e-07,
                4.58908225325491e-08,
                5.13518087266698e-08,
            ),
        ],
    )
    def test_accel_third_body_sun(self, mjd_tt, rx, ry, rz, ax, ay, az):
        """Test analytical Sun third-body acceleration with reference values."""
        epc = bh.Epoch.from_mjd(mjd_tt, bh.TimeSystem.TT)
        r_object = np.array([rx, ry, rz])

        a = bh.accel_third_body_sun(epc, r_object)

        assert a[0] == pytest.approx(ax, abs=1e-9)
        assert a[1] == pytest.approx(ay, abs=1e-9)
        assert a[2] == pytest.approx(az, abs=1e-9)

    @pytest.mark.parametrize(
        "mjd_tt,rx,ry,rz,ax,ay,az",
        [
            (
                60310.0,
                4884992.30378986,
                4553508.53744864,
                1330313.60479734,
                1.62360236246851e-07,
                -5.30930401572647e-07,
                -2.22022756088401e-07,
            ),
            (
                60310.0,
                2670937.8974923,
                5898362.79515022,
                2124959.71017719,
                -2.10084628821528e-07,
                -4.31933921171218e-07,
                -1.54339381002608e-07,
            ),
            (
                60310.0,
                38796.9774858514,
                6320698.88514676,
                2587294.93626938,
                -5.58483235850665e-07,
                -2.6203733817308e-07,
                -6.05903753125981e-08,
            ),
            (
                60310.0,
                -2599961.45855466,
                5760720.19357889,
                2647597.12683792,
                -8.25046337841761e-07,
                -4.53028242796273e-08,
                4.53066427075969e-08,
            ),
            (
                60310.0,
                -4839229.61832879,
                4313760.58255103,
                2300338.34996557,
                -9.63108738027384e-07,
                1.83858250202633e-07,
                1.4622908513799e-07,
            ),
            (
                60310.0,
                -6342536.88656784,
                2209712.29939824,
                1602811.60820791,
                -9.48011832170594e-07,
                3.86674684929409e-07,
                2.25026995803795e-07,
            ),
            (
                60310.0,
                -6891477.8215365,
                -227551.810286937,
                663813.896586629,
                -7.83191277225506e-07,
                5.28327949832493e-07,
                2.68246894531318e-07,
            ),
            (
                60310.0,
                -6412800.79978623,
                -2631381.70900648,
                -374371.749654303,
                -4.98912678830928e-07,
                5.85738566093379e-07,
                2.6909714100787e-07,
            ),
            (
                60310.0,
                -4983679.01699774,
                -4645498.60891225,
                -1357188.62711648,
                -1.44380586166042e-07,
                5.51955765893565e-07,
                2.28583689612585e-07,
            ),
            (
                60310.0,
                -2817603.71414268,
                -5972669.87274763,
                -2139313.41892538,
                2.23328070379479e-07,
                4.35988467235581e-07,
                1.54614554610566e-07,
            ),
            (
                60310.0,
                -234236.587976406,
                -6414628.84861909,
                -2604335.85309436,
                5.49604398045391e-07,
                2.59053532360343e-07,
                5.97432211499815e-08,
            ),
            (
                60310.0,
                2383524.57084058,
                -5900075.61185268,
                -2680956.18196418,
                7.89599228288718e-07,
                4.93434256460948e-08,
                -4.14510695387178e-08,
            ),
            (
                60310.0,
                4641862.77023787,
                -4497734.59354263,
                -2354086.60315067,
                9.12218233923866e-07,
                -1.62830886237673e-07,
                -1.34595434862506e-07,
            ),
            (
                60310.0,
                6193136.2430559,
                -2411369.56203787,
                -1669079.86356028,
                9.01868211930885e-07,
                -3.48656149518958e-07,
                -2.07100394322338e-07,
            ),
            (
                60310.0,
                6790850.71407875,
                45505.4274329756,
                -727399.838172203,
                7.59196602766636e-07,
                -4.83281433661868e-07,
                -2.49203881536061e-07,
            ),
            (
                60310.0,
                6333183.86841522,
                2494761.03873549,
                327102.634966258,
                5.01475600782815e-07,
                -5.47736810287354e-07,
                -2.54764046632745e-07,
            ),
        ],
    )
    def test_accel_third_body_moon(self, mjd_tt, rx, ry, rz, ax, ay, az):
        """Test analytical Moon third-body acceleration with reference values."""
        epc = bh.Epoch.from_mjd(mjd_tt, bh.TimeSystem.TT)
        r_object = np.array([rx, ry, rz])

        a = bh.accel_third_body_moon(epc, r_object)

        assert a[0] == pytest.approx(ax, abs=1e-9)
        assert a[1] == pytest.approx(ay, abs=1e-9)
        assert a[2] == pytest.approx(az, abs=1e-9)

    def test_accel_third_body_sun_with_state_vector(self):
        """Test that Sun third-body acceleration works with 6D state vector."""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_pos = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
        x_state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])

        # Compute with both inputs
        a_from_pos = bh.accel_third_body_sun(epc, r_pos)
        a_from_state = bh.accel_third_body_sun(epc, x_state)

        # Results should be identical
        assert np.allclose(a_from_pos, a_from_state, atol=1e-15)

    def test_accel_third_body_moon_with_state_vector(self):
        """Test that Moon third-body acceleration works with 6D state vector."""
        epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
        r_pos = np.array([bh.R_EARTH + 500e3, 1000e3, 2000e3])
        x_state = np.array([bh.R_EARTH + 500e3, 1000e3, 2000e3, 7500.0, 1000.0, -500.0])

        # Compute with both inputs
        a_from_pos = bh.accel_third_body_moon(epc, r_pos)
        a_from_state = bh.accel_third_body_moon(epc, x_state)

        # Results should be identical
        assert np.allclose(a_from_pos, a_from_state, atol=1e-15)


class TestDEThirdBody:
    """Tests for DE-based third-body acceleration functions."""

    def test_accel_third_body_sun_spice(self):
        """Test Sun DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_sun_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-5  # Should be on order of 1e-6 to 1e-7 m/s²

    def test_accel_third_body_moon_spice(self):
        """Test Moon DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_moon_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-5  # Should be on order of 1e-6 to 1e-7 m/s²

    def test_accel_third_body_mercury_spice(self):
        """Test Mercury DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_mercury_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-10  # Mercury effect is very small

    def test_accel_third_body_venus_spice(self):
        """Test Venus DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_venus_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-9  # Venus effect is small

    def test_accel_third_body_mars_spice(self):
        """Test Mars DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_mars_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-10  # Mars effect is very small

    def test_accel_third_body_jupiter_spice(self):
        """Test Jupiter DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_jupiter_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert (
            np.linalg.norm(a) < 1e-9
        )  # Jupiter effect is relatively larger but still small

    def test_accel_third_body_saturn_spice(self):
        """Test Saturn DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_saturn_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-10  # Saturn effect is small

    def test_accel_third_body_uranus_spice(self):
        """Test Uranus DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_uranus_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-11  # Uranus effect is very small

    def test_accel_third_body_neptune_spice(self):
        """Test Neptune DE440s third-body acceleration."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_object = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])

        a = bh.accel_third_body_neptune_spice(epc, r_object, bh.EphemerisSource.DE440s)

        assert a.shape == (3,)
        assert np.linalg.norm(a) > 0.0
        assert np.linalg.norm(a) < 1e-11  # Neptune effect is very small

    def test_accel_third_body_de_with_state_vector(self):
        """Test that DE440s third-body functions work with 6D state vectors."""
        epc = bh.Epoch.from_mjd(60310.0, bh.TimeSystem.TT)
        r_pos = np.array([4884992.30378986, 4553508.53744864, 1330313.60479734])
        x_state = np.array(
            [
                4884992.30378986,
                4553508.53744864,
                1330313.60479734,
                7500.0,
                1000.0,
                -500.0,
            ]
        )

        # Test Sun
        a_sun_pos = bh.accel_third_body_sun_spice(epc, r_pos, bh.EphemerisSource.DE440s)
        a_sun_state = bh.accel_third_body_sun_spice(
            epc, x_state, bh.EphemerisSource.DE440s
        )
        assert np.allclose(a_sun_pos, a_sun_state, atol=1e-15)

        # Test Moon
        a_moon_pos = bh.accel_third_body_moon_spice(
            epc, r_pos, bh.EphemerisSource.DE440s
        )
        a_moon_state = bh.accel_third_body_moon_spice(
            epc, x_state, bh.EphemerisSource.DE440s
        )
        assert np.allclose(a_moon_pos, a_moon_state, atol=1e-15)

        # Test Jupiter (representative planet)
        a_jup_pos = bh.accel_third_body_jupiter_spice(
            epc, r_pos, bh.EphemerisSource.DE440s
        )
        a_jup_state = bh.accel_third_body_jupiter_spice(
            epc, x_state, bh.EphemerisSource.DE440s
        )
        assert np.allclose(a_jup_pos, a_jup_state, atol=1e-15)


class TestThirdBodyForBody:
    """Tests for the central-body-aware accel_third_body_for_body."""

    def test_accel_third_body_for_body_earth_center_matches_legacy(self):
        """CentralBody.Earth + ThirdBody.SUN matches the legacy accel_third_body_sun_spice."""
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        legacy = bh.accel_third_body_sun_spice(epc, r, bh.EphemerisSource.DE440s)
        new = bh.accel_third_body_for_body(
            bh.CentralBody.Earth, bh.ThirdBody.SUN, bh.EphemerisSource.DE440s, epc, r
        )

        assert np.allclose(new, legacy, atol=1e-15)

    def test_accel_third_body_for_body_rejects_body_equal_to_center(self):
        """A third body with the same NAIF ID as the central body raises."""
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        with pytest.raises(RuntimeError):
            bh.accel_third_body_for_body(
                bh.CentralBody.Earth,
                bh.ThirdBody.EARTH,
                bh.EphemerisSource.DE440s,
                epc,
                r,
            )

    def test_accel_third_body_for_body_low_precision_rejected_for_non_earth_center(
        self,
    ):
        """LowPrecision ephemerides are geocentric and rejected for a non-Earth center."""
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([2e6, 0.0, 0.0])

        with pytest.raises(RuntimeError):
            bh.accel_third_body_for_body(
                bh.CentralBody.Moon,
                bh.ThirdBody.SUN,
                bh.EphemerisSource.LowPrecision,
                epc,
                r,
            )

    @pytest.mark.integration
    def test_emb_internal_bodies_use_direct_form(self):
        """Earth about EMB uses the direct term only (no indirect term at the barycenter)."""
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([1e8, 2e8, -5e7])
        s = bh.spk_position(399, 3, epc)
        d = s - r
        expected = bh.GM_EARTH * d / np.linalg.norm(d) ** 3

        got = bh.accel_third_body_for_body(
            bh.CentralBody.EMB, bh.ThirdBody.EARTH, bh.EphemerisSource.DE440s, epc, r
        )

        assert np.allclose(got, expected, atol=1e-18)

    @pytest.mark.integration
    def test_emb_external_body_uses_differential_form(self):
        """The Sun about EMB uses the full differential (direct minus indirect) form."""
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([1e8, 2e8, -5e7])
        s = bh.spk_position(10, 3, epc)
        d = s - r
        expected = bh.GM_SUN * (d / np.linalg.norm(d) ** 3 - s / np.linalg.norm(s) ** 3)

        got = bh.accel_third_body_for_body(
            bh.CentralBody.EMB, bh.ThirdBody.SUN, bh.EphemerisSource.DE440s, epc, r
        )

        assert np.allclose(got, expected, atol=1e-18)

    @pytest.mark.integration
    def test_ssb_all_bodies_direct(self):
        """Nothing accelerates the SSB, so every perturber uses the direct term only."""
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([1e8, 2e8, -5e7])
        s = bh.spk_position(5, 0, epc)
        d = s - r
        expected = bh.GM_JUPITER * d / np.linalg.norm(d) ** 3

        got = bh.accel_third_body_for_body(
            bh.CentralBody.SSB, bh.ThirdBody.JUPITER, bh.EphemerisSource.DE440s, epc, r
        )

        assert np.allclose(got, expected, atol=1e-18)

    @pytest.mark.integration
    def test_phobos_third_body_about_mars(self):
        """Phobos as a third body about a Mars-centered object is small but nonzero."""
        bh.load_kernel("mar099s")
        epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        r = np.array([bh.R_MARS + 400e3, 0.0, 0.0])

        a = bh.accel_third_body_for_body(
            bh.CentralBody.Mars, bh.ThirdBody.PHOBOS, bh.EphemerisSource.DE440s, epc, r
        )

        assert 0.0 < np.linalg.norm(a) < 1e-8
