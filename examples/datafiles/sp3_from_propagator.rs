//! Sample two numerical propagators on a regular grid and write them as an SP3-d file.

use brahe as bh;
use bh::datafiles::sp3::{SP3, SP3SatelliteId, SP3WriteOptions};
use bh::traits::DStatePropagator;
use bh::utils::DOrbitStateProvider;
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

    // The providers are sampled from their stored trajectories, so propagate first
    let end = epoch + 3600.0;
    for prop in propagators.iter_mut() {
        prop.propagate_to(end)?;
    }

    // Sample both satellites every 60 s for one hour
    let l01: SP3SatelliteId = "L01".parse()?;
    let l02: SP3SatelliteId = "L02".parse()?;
    let options = SP3WriteOptions::default()
        .with_agency("BRHE")
        .with_comments(vec!["Sampled from numerical propagators".to_string()]);
    let sp3 = SP3::from_propagators(
        &[
            (l01, &propagators[0] as &dyn DOrbitStateProvider),
            (l02, &propagators[1] as &dyn DOrbitStateProvider),
        ],
        epoch,
        end,
        60.0,
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
