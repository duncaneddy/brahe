import numpy as np
import brahe as bh


def _assert_koe_close(back, koe):
    np.testing.assert_allclose(back[0], koe[0], atol=1e-6)
    np.testing.assert_allclose(back[1], koe[1], atol=1e-12)
    for idx in range(2, 6):
        d = abs(back[idx] - koe[idx]) % 360.0
        d = min(d, 360.0 - d)
        np.testing.assert_allclose(d, 0.0, atol=1e-7)


def test_equinoctial_round_trip_direct():
    koe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
    eqn = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES, 1)
    back = bh.state_equinoctial_to_koe(eqn, bh.AngleFormat.DEGREES, 1)
    _assert_koe_close(back, koe)


def test_equinoctial_round_trip_near_circular_near_equatorial():
    koe = np.array([bh.R_EARTH + 700e3, 1e-5, 0.01, 10.0, 20.0, 30.0])
    eqn = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES, 1)
    back = bh.state_equinoctial_to_koe(eqn, bh.AngleFormat.DEGREES, 1)
    # a, e, and the fast/longitude combinations are recoverable even as
    # omega, Omega individually blur.
    np.testing.assert_allclose(back[0], koe[0], atol=1e-6)
    np.testing.assert_allclose(back[1], koe[1], atol=1e-9)


def test_equinoctial_round_trip_retrograde():
    koe = np.array([bh.R_EARTH + 800e3, 0.02, 175.0, 40.0, 50.0, 120.0])
    eqn = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES, -1)
    back = bh.state_equinoctial_to_koe(eqn, bh.AngleFormat.DEGREES, -1)
    _assert_koe_close(back, koe)


def test_equinoctial_fr_defaults_to_one():
    koe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
    eqn_default = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES)
    eqn_explicit = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES, 1)
    np.testing.assert_allclose(eqn_default, eqn_explicit)


def test_batch_equinoctial():
    koe = np.array(
        [
            [bh.R_EARTH + 500e3 + 1e3 * i, 0.01, 45.0, 30.0 + i, 60.0, 90.0]
            for i in range(3)
        ]
    )
    for fr in (1, -1):
        eqn = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES, fr)
        back = bh.state_equinoctial_to_koe(eqn, bh.AngleFormat.DEGREES, fr)
        assert eqn.shape == (3, 6)
        for i in range(3):
            np.testing.assert_array_equal(
                eqn[i], bh.state_koe_to_equinoctial(koe[i], bh.AngleFormat.DEGREES, fr)
            )
            np.testing.assert_array_equal(
                back[i], bh.state_equinoctial_to_koe(eqn[i], bh.AngleFormat.DEGREES, fr)
            )
        np.testing.assert_allclose(back, koe, atol=1e-6)
        np.testing.assert_array_equal(
            bh.state_koe_to_equinoctial(koe.T, bh.AngleFormat.DEGREES, fr, axis=0),
            eqn.T,
        )
