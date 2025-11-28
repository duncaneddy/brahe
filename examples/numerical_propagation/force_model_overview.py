# /// script
# dependencies = ["brahe"]
# ///
"""
Complete overview of ForceModelConfig showing all configuration fields.
This example demonstrates every configurable option for force modeling.
"""

import brahe as bh

# Create a fully-configured force model
force_config = bh.ForceModelConfig(
    # Gravity: Spherical harmonic model (EGM2008, 20x20 degree/order)
    gravity=bh.GravityConfiguration.spherical_harmonic(
        degree=20,
        order=20,
        model_type=bh.GravityModelType.EGM2008_360,
    ),
    # Atmospheric drag: Harris-Priester model with parameter indices
    drag=bh.DragConfiguration(
        model=bh.AtmosphericModel.HARRIS_PRIESTER,
        area=bh.ParameterSource.parameter_index(1),  # Index into parameter vector
        cd=bh.ParameterSource.parameter_index(2),
    ),
    # Solar radiation pressure: Conical eclipse model
    srp=bh.SolarRadiationPressureConfiguration(
        area=bh.ParameterSource.parameter_index(3),
        cr=bh.ParameterSource.parameter_index(4),
        eclipse_model=bh.EclipseModel.CONICAL,
    ),
    # Third-body: Sun and Moon with DE440s ephemeris
    third_body=bh.ThirdBodyConfiguration(
        ephemeris_source=bh.EphemerisSource.DE440s,
        bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
    ),
    # General relativistic corrections
    relativity=True,
    # Spacecraft mass (can also use parameter_index for estimation)
    mass=bh.ParameterSource.value(1000.0),  # kg
)

print(f"Gravity: {force_config.gravity}")
print(f"Drag: {force_config.drag}")
print(f"SRP: {force_config.srp}")
print(f"Third-body: {force_config.third_body}")
print(f"Relativity: {force_config.relativity}")
print(f"Mass: {force_config.mass}")
# Gravity: SphericalHarmonic(source=EGM2008_360, degree=20, order=20)
# Drag: DragConfiguration(model=HarrisPriester, area=ParameterIndex(1), cd=ParameterIndex(2))
# SRP: SolarRadiationPressureConfiguration(area=ParameterIndex(3), cr=ParameterIndex(4), eclipse_model=Conical)
# Third-body: ThirdBodyConfiguration(ephemeris_source=DE440s, bodies=[Sun, Moon])
# Relativity: True
# Mass: Value(1000.0)
