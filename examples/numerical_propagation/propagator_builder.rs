//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```
//! Constructing a NumericalOrbitPropagator using the builder API.
//!
//! The builder takes the three required fields — epoch, state, and
//! force_config — directly as arguments to `builder()`. Optional fields such
//! as initial_covariance default to None when omitted and are set through
//! chained setters.

use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let state = na::DVector::from_column_slice(
        bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees).as_slice(),
    );
    

    // Minimal: only the three required fields
    let mut prop = bh::DNumericalOrbitPropagator::builder(
        epoch,
        state.clone(),
        bh::ForceModelConfig::earth_gravity(),
    )
    .build()
    .unwrap();

    prop.propagate_to(epoch + 3600.0);
    println!("Minimal builder — epoch: {}", prop.current_epoch());

    // With optional fields: custom propagation config and initial covariance
    let p0 = na::DMatrix::<f64>::identity(6, 6) * 1e6;

    let mut prop_with_cov = bh::DNumericalOrbitPropagator::builder(
        epoch,
        state,
        bh::ForceModelConfig::earth_gravity(),
    )
    .propagation_config(bh::NumericalPropagationConfig::high_precision())
    .initial_covariance(p0)
    .build()
    .unwrap();

    prop_with_cov.propagate_to(epoch + 3600.0);

    assert_eq!(DStatePropagator::state_dim(&prop_with_cov), 6);
    assert!(prop_with_cov.current_covariance().is_some());

    println!("Builder with covariance — epoch: {}", prop_with_cov.current_epoch());
    println!("Example validated successfully!");
}
