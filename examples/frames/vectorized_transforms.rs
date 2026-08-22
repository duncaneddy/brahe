//! Transform batches of states and positions between ECI and ECEF in one call

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Build a batch of ECI states, one per satellite
    let states_eci: Vec<na::SVector<f64, 6>> = (0..8)
        .map(|i| {
            let raan = 45.0 * i as f64;
            let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, raan, 0.0, 0.0);
            bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees)
        })
        .collect();
    println!("ECI states: {}", states_eci.len());

    // One epoch, many states: the rotation matrices are computed once
    let states_ecef = bh::states_eci_to_ecef(&[epc], &states_eci).unwrap();
    println!("ECEF states: {}", states_ecef.len());
    let s = states_ecef[0];
    println!(
        "First ECEF state: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        s[0], s[1], s[2], s[3], s[4], s[5]
    );

    // Many epochs, one position: a ground station tracked through inertial space
    let epochs: Vec<bh::Epoch> = (0..6).map(|i| epc + 600.0 * i as f64).collect();
    let station_ecef = bh::position_geodetic_to_ecef(
        na::Vector3::new(-122.4, 37.8, 0.0),
        bh::AngleFormat::Degrees,
    )
    .unwrap();
    let station_eci = bh::positions_ecef_to_eci(&epochs, &[station_ecef]).unwrap();
    println!("Station ECI positions: {}", station_eci.len());
    for (e, r) in epochs.iter().zip(&station_eci) {
        println!("  {}: [{:.1}, {:.1}, {:.1}] m", e, r[0], r[1], r[2]);
    }

    // Many epochs, many states: one epoch per state
    let states_ecef_series = bh::states_eci_to_ecef(&epochs, &states_eci[..6]).unwrap();
    println!("Per-epoch ECEF states: {}", states_ecef_series.len());

    // A sequence of epochs also vectorizes the rotation matrices
    let rotations = bh::rotations_eci_to_ecef(&epochs);
    println!("Rotation matrices: {}", rotations.len());
}
