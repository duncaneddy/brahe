//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! EMB-centered cislunar propagation with Earth-attributed force models.
//! Earth contributes an 8x8 spherical-harmonic field as a third body and
//! NRLMSISE-00 drag evaluated at the object's Earth-relative state, so a
//! trajectory passing through LEO altitudes keeps Earth-fidelity forces
//! while the integration state stays Earth-Moon-barycenter-centered.

use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // EMB-centered force model: no central gravity term (the barycenter has
    // no mass of its own); Earth carries a spherical-harmonic field and the
    // atmosphere; the Moon and Sun are point-mass perturbers.
    let force_config = bh::ForceModelConfig {
        central_body: bh::CentralBody::EMB,
        gravity: bh::GravityConfiguration::Zero,
        drag: Some(bh::DragConfiguration {
            model: bh::AtmosphericModel::NRLMSISE00,
            area: bh::ParameterSource::Value(10.0),
            cd: bh::ParameterSource::Value(2.2),
            // Attribute the drag to Earth: density and relative wind are
            // evaluated at the object's state relative to Earth.
            body: Some(bh::CentralBody::Earth),
        }),
        srp: None,
        third_body: Some(vec![
            bh::ThirdBodyConfiguration {
                body: bh::ThirdBody::Earth,
                ephemeris_source: bh::EphemerisSource::DE440s,
                gravity: bh::GravityConfiguration::SphericalHarmonic {
                    source: bh::GravityModelSource::default(),
                    degree: 8,
                    order: 8,
                    parallel: bh::orbit_dynamics::ParallelMode::Auto,
                },
            },
            bh::ThirdBody::Moon.into(),
            bh::ThirdBody::Sun.into(),
        ]),
        relativity: false,
        mass: Some(bh::ParameterSource::Value(1000.0)),
        frame_transform: bh::FrameTransformationModel::default(),
        tides: None,
    };
    force_config.validate().unwrap();

    // Start from a 500 km Earth orbit, re-expressed about the EMB via the
    // ECI->EMBI frame translation.
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 45.0);
    let x_earth = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let x_emb = bh::frames::state_eci_to_emb(epoch, x_earth);

    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(x_emb.as_slice()),
        bh::NumericalPropagationConfig::default(),
        force_config,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // Propagate for one day
    let epoch_end = epoch + 86400.0;
    prop.propagate_to(epoch_end);

    let x_final = prop.current_state();
    println!("Initial EMB-centered state: {:?}", x_emb.as_slice());
    println!(
        "Final EMB-centered state:   [{:.3}, {:.3}, {:.3}, {:.6}, {:.6}, {:.6}]",
        x_final[0], x_final[1], x_final[2], x_final[3], x_final[4], x_final[5]
    );

    // Re-express the final state about Earth for reference
    let x_final_eci = bh::frames::state_emb_to_eci(
        epoch_end,
        na::SVector::<f64, 6>::from_iterator(x_final.iter().cloned()),
    );
    let altitude =
        (x_final_eci[0].powi(2) + x_final_eci[1].powi(2) + x_final_eci[2].powi(2)).sqrt()
            - bh::R_EARTH;
    println!("Final altitude above Earth: {:.1} km", altitude / 1e3);
}
