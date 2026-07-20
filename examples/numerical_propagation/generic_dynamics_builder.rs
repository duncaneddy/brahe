//! ```cargo
//! [dependencies]
//! brahe = { path = "../../" }
//! nalgebra = "0.33"
//! ```
//! Constructing a NumericalPropagator using the builder API.
//!
//! The builder takes the three required fields -- epoch, state, and
//! dynamics_fn -- directly as arguments to `builder()`. Optional fields
//! such as params and initial_covariance default when omitted and are
//! set through chained setters.

use brahe as bh;
use bh::traits::{DStatePropagator, DStateProvider};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    let omega = 2.0 * PI; // 1 Hz oscillation frequency
    let initial_state = na::DVector::from_vec(vec![1.0, 0.0]); // [x0, v0]

    let dynamics_fn: bh::DStateDynamics = Box::new(
        move |_t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| {
            let x = state[0];
            let v = state[1];
            let omega_sq = params.map(|p| p[0]).unwrap_or(omega * omega);
            na::DVector::from_vec(vec![v, -omega_sq * x])
        },
    );

    let mut prop = bh::DNumericalPropagator::builder(epoch, initial_state, dynamics_fn)
        .propagation_config(bh::NumericalPropagationConfig::default())
        .params(na::DVector::from_vec(vec![omega * omega]))
        .build()
        .unwrap();

    let period = 2.0 * PI / omega; // Period = 2*pi/omega = 1 second
    prop.propagate_to(epoch + 5.0 * period);

    let final_state = prop.state(epoch + 5.0 * period).unwrap();
    println!("Position after 5 periods: {:+.6} m", final_state[0]);
    println!("Velocity after 5 periods: {:+.6} m/s", final_state[1]);

    assert!((final_state[0] - 1.0).abs() < 0.01);
    println!("Example validated successfully!");
}
