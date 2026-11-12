//! Sample two numerical propagators on a regular grid and write them as an SP3-d file.

#[allow(unused_imports)]
use brahe as bh;
use bh::datafiles::sp3::{SP3, SP3SatelliteId, SP3Header};
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    let epoch = bh::Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh::TimeSystem::GPS);

    // Two LEO satellites in different planes
    let mut propagators = Vec::new();
    for raan in [0.0, 90.0] {
        let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 600e3, 0.001, 97.8, raan, 0.0, 0.0);
        let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
        propagators.push(
            bh::DNumericalOrbitPropagator::builder(
                epoch,
                na::DVector::from_column_slice(state.as_slice()),
                bh::ForceModelConfig::earth_gravity(),
            )
            .build()?,
        );
    }

    // The file is built from the trajectories the propagators store, so propagate first
    let end = epoch + 3600.0;
    for prop in propagators.iter_mut() {
        prop.propagate_to(end)?;
    }

    // Resample both trajectories every 60 s over the hour they cover
    let l01: SP3SatelliteId = "L01".parse()?;
    let l02: SP3SatelliteId = "L02".parse()?;
    let options = SP3Header { agency: "BRHE".to_string(), comments: vec!["Sampled from numerical propagators".to_string()], ..Default::default() };
    let sp3 = SP3::from_trajectories(
        &[
            (l01, propagators[0].trajectory()),
            (l02, propagators[1].trajectory()),
        ],
        Some(60.0),
        options,
    )?;

    println!(
        "Satellites: {}, epochs: {}",
        sp3.num_satellites(),
        sp3.num_epochs()
    );

    let text = sp3.to_string()?;
    println!("{}", text.lines().take(10).collect::<Vec<_>>().join("\n"));

    Ok(())
}
