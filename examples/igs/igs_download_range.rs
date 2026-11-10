//! Download a short range of IGS final orbit products, chain them into continuous satellite trajectories, and save the files.
//!
//! Uses the Garner (SIO) mirror, which needs no credentials. Garner is served over
//! plain HTTP, so the downloaded files are not transport-secured.

// FLAGS = ["NETWORK"]

use brahe as bh;
use bh::datafiles::sp3::SP3Satellite;
use bh::igs::{
    IGSContentType, IGSFileFormat, IGSProductFilter, IGSProductSource, IGSProductsClient,
    IGSRangeOptions, IGSSolutionType,
};
use bh::time::{Epoch, TimeSystem};
use bh::traits::DOrbitStateProvider;
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    let start = Epoch::from_datetime(2024, 11, 11, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    let end = Epoch::from_datetime(2024, 11, 12, 0, 0, 0.0, 0.0, TimeSystem::GPS);

    // Daily CODE final orbit files covering the range, plus one day on each side
    let files = client.download_sp3_range(
        start,
        end,
        "COD",
        IGSSolutionType::Final,
        &IGSRangeOptions::default(),
    )?;
    println!("Downloaded {} daily SP3 files", files.len());
    for sp3 in &files {
        println!(
            "  {} to {}: {} epochs",
            sp3.start_epoch().unwrap(),
            sp3.end_epoch().unwrap(),
            sp3.num_epochs()
        );
    }

    // One continuous trajectory per satellite across all files
    let satellites = SP3Satellite::from_files(&files)?;
    let g05 = satellites
        .iter()
        .find(|s| s.get_name().as_deref() == Some("G05"))
        .unwrap();
    println!(
        "\nG05: {} samples from {} to {}",
        g05.len(),
        g05.start_epoch(),
        g05.end_epoch()
    );

    for hours in [0.0, 6.0, 12.0, 18.0] {
        let epoch = start + hours * 3600.0;
        let state = g05.state_eci(epoch)?;
        println!(
            "{}: r = [{:.1}, {:.1}, {:.1}] m, |v| = {:.1} m/s",
            epoch,
            state[0],
            state[1],
            state[2],
            state.fixed_rows::<3>(3).norm()
        );
    }

    // Save the files covering the range, decompressed, to a directory of choice
    let directory = std::env::temp_dir().join("brahe_igs_download_range");
    let filter = IGSProductFilter::new()
        .with_center("COD")
        .with_solution(IGSSolutionType::Final)
        .with_content(IGSContentType::Orbit)
        .with_format(IGSFileFormat::SP3);
    let paths = client.save_between(start, end, &filter, &directory, false)?;
    println!("\nSaved {} files:", paths.len());
    for path in &paths {
        println!("  {}", path.file_name().unwrap().to_string_lossy());
    }
    std::fs::remove_dir_all(&directory)?;

    Ok(())
}
