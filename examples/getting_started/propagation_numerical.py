import numpy as np
import brahe as bh

# Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
bh.initialize_eop()
bh.initialize_sw()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Define orbital elements: [a, e, i, raan, argp, M] in SI units
# LEO satellite: 500 km altitude, near-circular, sun-synchronous inclination
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Full Froce model configuration options
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
    mass=bh.ParameterSource.parameter_index(0),  # kg
)

# Full Numerical integration configuration options
integrator_config = bh.NumericalPropagationConfig(
    # Integration method: Dormand-Prince 5(4)
    bh.IntegrationMethod.DP54,
    # Integrator settings: tolerances and step control
    bh.IntegratorConfig(
        abs_tol=1e-9,
        rel_tol=1e-6,
        initial_step=60.0,  # 60 second initial step
        min_step=1e-6,  # Minimum step size
        max_step=300.0,  # Maximum step size (5 minutes)
        step_safety_factor=0.9,  # Safety margin for step control
        min_step_scale_factor=0.2,  # Minimum step reduction
        max_step_scale_factor=10.0,  # Maximum step growth
        max_step_attempts=10,  # Max attempts per step
    ),
    # Variational configuration: STM and sensitivity settings
    bh.VariationalConfig(
        enable_stm=True,
        enable_sensitivity=False,
        store_stm_history=True,
        store_sensitivity_history=False,
    ),
)

# Create propagator with default configuration
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    integrator_config,
    force_config,
    params,
)

# Propagate for 1 hour
prop.propagate_to(epoch + 3600.0)

# Get final state
final_epoch = prop.current_epoch()
final_state = prop.current_state()

print(f"Initial epoch: {epoch}")
print(f"Final epoch:   {final_epoch}")
print(
    f"Position (km): [{final_state[0] / 1e3:.3f}, {final_state[1] / 1e3:.3f}, {final_state[2] / 1e3:.3f}]"
)
print(
    f"Velocity (m/s): [{final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}]"
)