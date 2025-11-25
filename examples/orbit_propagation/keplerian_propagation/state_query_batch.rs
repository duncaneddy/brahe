//! Query KeplerianPropagator states at multiple epochs in batch

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::DOrbitStateProvider;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    // Generate states at irregular intervals
    let query_epochs = vec![
        epoch, epoch + 100.0, epoch + 500.0, epoch + 1000.0, epoch + 3600.0
    ];
    let states_eci = prop.states_eci(&query_epochs).unwrap();

    println!("Generated {} states", states_eci.len());
    // Generated 5 states
    for (i, state) in states_eci.iter().enumerate() {
        println!("  Epoch {}: position magnitude = {:.1} km",
                 i, state.fixed_rows::<3>(0).norm() / 1e3);
    }
}

// Output:
// Generated 5 states
//   Epoch 0: position magnitude = 6873.3 km
//   Epoch 1: position magnitude = 6873.8 km
//   Epoch 2: position magnitude = 6876.6 km
//   Epoch 3: position magnitude = 6880.3 km
//   Epoch 4: position magnitude = 6877.7 km