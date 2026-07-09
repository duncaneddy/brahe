"""
End-to-end tests for tidal-force modeling via the Python API.

Covers:
  - GravityModel.convert_tide_system() round-trip fidelity
  - Full propagation with SolidTideConfig enabled via ForceModelConfig
"""

import numpy as np
import brahe as bh


# ===========================================================================
# Helpers
# ===========================================================================


def _leo_state():
    """ECI Cartesian state for a 500 km, near-circular LEO orbit."""
    oe = np.array(
        [
            bh.R_EARTH + 500e3,
            0.01,
            np.radians(97.8),
            np.radians(15.0),
            np.radians(30.0),
            np.radians(45.0),
        ]
    )
    return bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)


# ===========================================================================
# Tests
# ===========================================================================


def test_convert_tide_system_roundtrip():
    """TideFree → ZeroTide → TideFree must recover C20 to machine precision."""
    m = bh.GravityModel.from_model_type(bh.GravityModelType.JGM3)
    c20_orig = m.get(2, 0)[0]

    m.convert_tide_system(
        bh.GravityModelTideSystem.TideFree,
        bh.GravityModelTideSystem.ZeroTide,
    )
    m.convert_tide_system(
        bh.GravityModelTideSystem.ZeroTide,
        bh.GravityModelTideSystem.TideFree,
    )

    c20_final = m.get(2, 0)[0]
    assert abs(c20_final - c20_orig) < 1e-18


def test_propagate_with_solid_tides():
    """Propagate a LEO orbit for two steps with solid tides enabled.

    Uses ForceModelConfig.earth_gravity() (20x20 EGM2008 gravity, no drag/SRP)
    to avoid the space-weather and ephemeris initialisation required by
    leo_default().  A TidesConfiguration with SolidTideConfig is injected via
    the .tides setter.  The resulting state must be finite and physically
    bounded (altitude 350–750 km, speed 6.5–8.5 km/s).
    """
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state = _leo_state()

    cfg = bh.ForceModelConfig.earth_gravity()
    cfg.tides = bh.TidesConfiguration(
        permanent=bh.PermanentTideConfig.AUTO,
        solid=bh.SolidTideConfig(frequency_dependent=True),
    )

    prop = bh.NumericalOrbitPropagator(
        epoch,
        state,
        bh.NumericalPropagationConfig.default(),
        cfg,
        None,
    )

    # Step forward two integration steps (60 s each)
    prop.step_by(60.0)
    prop.step_by(60.0)

    final_state = prop.current_state()

    assert len(final_state) == 6
    assert all(np.isfinite(final_state)), "State contains NaN or Inf"

    r = np.linalg.norm(final_state[:3])
    v = np.linalg.norm(final_state[3:])

    alt_km = (r - bh.R_EARTH) / 1e3
    v_km_s = v / 1e3

    assert 350.0 < alt_km < 750.0, f"Altitude {alt_km:.1f} km out of expected range"
    assert 6.5 < v_km_s < 8.5, f"Speed {v_km_s:.3f} km/s out of expected range"
