//! Chain consecutive SP3 files into one continuous series of samples for each satellite.

#[allow(unused_imports)]
use brahe as bh;
use bh::datafiles::sp3::{SP3, SP3Satellite};
use bh::traits::Trajectory;
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = ["truncated", "after_drop", "after_no_drop"]
        .iter()
        .map(|name| SP3::from_file(format!("test_assets/sp3/gbm19500_{name}.sp3")))
        .collect::<Result<Vec<_>, _>>()?;
    for sp3 in &files {
        println!(
            "{} to {}: {} epochs",
            sp3.start_epoch().unwrap(),
            sp3.end_epoch().unwrap(),
            sp3.num_epochs()
        );
    }

    // The files may be given in any order; neighbors share a boundary epoch
    let satellites = SP3Satellite::from_files(&files)?;
    println!("\nChained satellites: {}", satellites.len());

    for sat in satellites.iter().take(3) {
        println!(
            "{}: {} samples from {} to {}",
            sat.get_name().unwrap(),
            sat.len(),
            sat.start_epoch(),
            sat.end_epoch()
        );
    }

    // Every satellite is one continuous run, so each converts to a single trajectory
    let trajectories = satellites[0].to_trajectories()?;
    let traj = &trajectories[0];
    println!(
        "\n{} trajectories: {}",
        satellites[0].get_name().unwrap(),
        trajectories.len()
    );
    println!("First trajectory: {} states in {}", traj.len(), traj.frame);

    // A file that starts more than 1.5 intervals after the previous one leaves a gap
    let later = SP3::from_file("test_assets/sp3/gbm19500_large_gap.sp3")?;
    let sat = SP3Satellite::from_files(&[files[0].clone(), later])?.remove(0);
    println!(
        "\n{} across a gap: {} samples",
        sat.get_name().unwrap(),
        sat.len()
    );
    for traj in sat.to_trajectories()? {
        println!(
            "  trajectory {} to {}",
            traj.start_epoch().unwrap(),
            traj.end_epoch().unwrap()
        );
    }

    Ok(())
}
