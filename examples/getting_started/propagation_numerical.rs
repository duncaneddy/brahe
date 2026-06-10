use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define orbital elements: [a, e, i, raan, argp, M] in SI units
    // LEO satellite: 500 km altitude, near-circular, sun-synchronous inclination
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.001,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Parameters: [mass, drag_area, Cd, srp_area, Cr]
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create a fully-configured force model
    let force_config = bh::ForceModelConfig {
        // Gravity: Spherical harmonic model (EGM2008, 20x20 degree/order)
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(bh::GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
        },
        // Atmospheric drag: Harris-Priester model with parameter indices
        drag: Some(bh::DragConfiguration {
            model: bh::AtmosphericModel::HarrisPriester,
            area: bh::ParameterSource::ParameterIndex(1), // Index into parameter vector
            cd: bh::ParameterSource::ParameterIndex(2),
        }),
        // Solar radiation pressure: Conical eclipse model
        srp: Some(bh::SolarRadiationPressureConfiguration {
            area: bh::ParameterSource::ParameterIndex(3),
            cr: bh::ParameterSource::ParameterIndex(4),
            eclipse_model: bh::EclipseModel::Conical,
        }),
        // Third-body: Sun and Moon with DE440s ephemeris
        third_body: Some(bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::DE440s,
            bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
        }),
        // General relativistic corrections
        relativity: true,
        // Spacecraft mass (can also use ParameterIndex for estimation)
        mass: Some(bh::ParameterSource::Value(1000.0)), // kg
        // ECI<->ECEF rotation precision used by gravity, drag, and density models.
        // FullEarthRotation is the default and applies the full IAU 2006/2000A
        // bias-precession-nutation + Earth rotation + polar motion. Use
        // EarthRotationOnly for a faster, slightly less accurate alternative.
        frame_transform: bh::FrameTransformationModel::FullEarthRotation,
    };

    use brahe as bh;

    // Create a fully-configured integrator configuration
    let integrator_config = bh::NumericalPropagationConfig {
        // Integration method: Dormand-Prince 5(4)
        method: bh::IntegratorMethod::DP54,
        // Integrator settings: tolerances and step control
        integrator: bh::IntegratorConfig {
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            initial_step: Some(60.0), // 60 second initial step
            min_step: Some(1e-6),     // Minimum step size
            max_step: Some(300.0),    // Maximum step size (5 minutes)
            step_safety_factor: Some(0.9),      // Safety margin
            min_step_scale_factor: Some(0.2),   // Minimum step reduction
            max_step_scale_factor: Some(10.0),  // Maximum step growth
            max_step_attempts: 10,              // Max attempts per step
            fixed_step_size: None,              // Not using fixed step
        },
        // Variational configuration: STM and sensitivity settings
        variational: bh::VariationalConfig {
            enable_stm: true,
            enable_sensitivity: false,
            store_stm_history: true,
            store_sensitivity_history: false,
            jacobian_method: bh::DifferenceMethod::Central,
            sensitivity_method: bh::DifferenceMethod::Central,
        },
        // Acceleration storage for higher-order interpolation
        store_accelerations: true,
        // Interpolation method: Linear is safe for any state dimension
        // Use HermiteCubic or HermiteQuintic for 6D orbital states
        interpolation_method: bh::InterpolationMethod::Linear,
    };

    // Create propagator with default configuration
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        integrator_config,
        force_config,
        Some(params),
        None,  // No additional dynamics
        None,  // No control input
        None,  // No initial covariance
    )
    .unwrap();

    // Propagate for 1 hour
    prop.propagate_to(epoch + 3600.0);

    // Get final state
    let final_epoch = prop.current_epoch();
    let final_state = prop.current_state();

    println!("Initial epoch: {}", epoch);
    println!("Final epoch:   {}", final_epoch);
    println!(
        "Position (km): [{:.3}, {:.3}, {:.3}]",
        final_state[0] / 1e3,
        final_state[1] / 1e3,
        final_state[2] / 1e3
    );
    println!(
        "Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        final_state[3], final_state[4], final_state[5]
    );
}