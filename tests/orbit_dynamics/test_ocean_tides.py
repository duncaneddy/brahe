"""
End-to-end tests for ocean-tide force modeling via the Python API.

Covers:
  - OceanTideConfig round-trip through TidesConfiguration
  - Full propagation with ocean tides enabled via ForceModelConfig

Rust-internal items exercised by src/orbit_dynamics/ocean_tides.rs (the
FES2004 coefficient parser, the internal TideField representation, and the
Delta-Cnm/Snm computation functions) have no Python surface and are
therefore not mirrored here — only the public `OceanTideConfig` /
`TidesConfiguration` / `ForceModelConfig` API is exercised.
"""

import numpy as np
import brahe as bh


# ===========================================================================
# Helpers
# ===========================================================================


def _leo_state():
    """ECI Cartesian state for an 800 km, near-circular LEO orbit."""
    oe = np.array(
        [
            bh.R_EARTH + 800e3,
            0.001,
            98.7,
            30.0,
            40.0,
            0.0,
        ]
    )
    return bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)


# ===========================================================================
# Tests
# ===========================================================================


def test_ocean_tide_config_roundtrip_through_tides_configuration():
    """OceanTideConfig survives construction round-trip via TidesConfiguration."""
    ocean = bh.OceanTideConfig(degree=30, order=30, include_admittance=False)
    tides = bh.TidesConfiguration(
        permanent=bh.PermanentTideConfig.AUTO,
        ocean=ocean,
    )

    assert tides.ocean is not None
    assert tides.ocean.degree == 30
    assert tides.ocean.order == 30
    assert not tides.ocean.include_admittance
    assert not tides.ocean.pole_tide


def test_propagation_ocean_tides_effect_magnitude(_fes2004_cache_setup):
    """Ocean tides in LEO shift the trajectory by a small but non-zero amount.

    Compact Python mirror of the Rust
    `test_propagation_ocean_tides_effect_magnitude` test: propagate the same
    initial state for one day with ocean tides off vs. on (permanent and
    solid tides disabled in both cases) and confirm the resulting position
    difference is small (sub-meter to tens of meters) but non-zero.
    """
    epoch = bh.Epoch.from_datetime(2020, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state = _leo_state()

    base_cfg = bh.ForceModelConfig.earth_gravity()
    base_cfg.tides = bh.TidesConfiguration(permanent=bh.PermanentTideConfig.OFF)

    ocean_cfg = bh.ForceModelConfig.earth_gravity()
    ocean_cfg.tides = bh.TidesConfiguration(
        permanent=bh.PermanentTideConfig.OFF,
        ocean=bh.OceanTideConfig(degree=20, order=20),
    )

    p0 = bh.NumericalOrbitPropagator(
        epoch,
        state,
        bh.NumericalPropagationConfig.default(),
        base_cfg,
        None,
    )
    p1 = bh.NumericalOrbitPropagator(
        epoch,
        state,
        bh.NumericalPropagationConfig.default(),
        ocean_cfg,
        None,
    )

    p0.step_by(86400.0)
    p1.step_by(86400.0)

    d = np.linalg.norm(p1.current_state()[:3] - p0.current_state()[:3])

    assert 1e-3 < d < 50.0, f"ocean-tide displacement over 1 day: {d} m"
