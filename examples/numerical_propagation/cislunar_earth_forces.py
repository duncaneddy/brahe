# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
EMB-centered cislunar propagation with Earth-attributed force models.
Earth contributes an 8x8 spherical-harmonic field as a third body and
NRLMSISE-00 drag evaluated at the object's Earth-relative state, so a
trajectory passing through LEO altitudes keeps Earth-fidelity forces
while the integration state stays Earth-Moon-barycenter-centered.
"""

import numpy as np
import brahe as bh

# Initialize EOP and space weather data (required for NRLMSISE-00)
bh.initialize_eop()
bh.initialize_sw()

epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# EMB-centered force model: point-mass "central" gravity (the barycenter has
# no mass of its own); Earth carries a spherical-harmonic field and the
# atmosphere; the Moon and Sun are point-mass perturbers.
force_config = bh.ForceModelConfig.for_body(
    bh.CentralBody.EMB,
    bh.GravityConfiguration.point_mass(),
    drag=bh.DragConfiguration(
        model=bh.AtmosphericModel.NRLMSISE00,
        area=bh.ParameterSource.value(10.0),
        cd=bh.ParameterSource.value(2.2),
        # Attribute the drag to Earth: density and relative wind are
        # evaluated at the object's state relative to Earth.
        body=bh.CentralBody.Earth,
    ),
    third_bodies=[
        bh.ThirdBodyConfiguration(
            bh.ThirdBody.EARTH,
            gravity=bh.GravityConfiguration.spherical_harmonic(degree=8, order=8),
        ),
        bh.ThirdBody.MOON,
        bh.ThirdBody.SUN,
    ],
    mass=bh.ParameterSource.value(1000.0),
)
force_config.validate()

# Start from a 500 km Earth orbit, re-expressed about the EMB by adding
# Earth's barycentric state from the DE ephemeris.
oe = np.array([bh.R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 45.0])
x_earth = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
x_emb = x_earth + bh.spk_state(399, 3, epoch)

prop = bh.NumericalOrbitPropagator(
    epoch,
    x_emb,
    bh.NumericalPropagationConfig.default(),
    force_config,
    None,
)

# Propagate for one day
epoch_end = epoch + 86400.0
prop.propagate_to(epoch_end)

x_final = prop.current_state()
print(f"Initial EMB-centered state: {x_emb}")
print(f"Final EMB-centered state:   {x_final}")

# Re-express the final state about Earth for reference
altitude = (
    np.linalg.norm(x_final[:3] - bh.spk_state(399, 3, epoch_end)[:3]) - bh.R_EARTH
)
print(f"Final altitude above Earth: {altitude / 1e3:.1f} km")
