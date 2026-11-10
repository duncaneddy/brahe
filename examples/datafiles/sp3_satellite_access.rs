//! Compute ground station access windows directly from the satellites in an SP3 file.

#[allow(unused_imports)]
use brahe as bh;
use bh::datafiles::sp3::SP3;
use bh::utils::Identifiable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bh::initialize_eop()?;

    let sp3 = SP3::from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3")?;

    // One SP3Satellite per satellite with valid positions, named by its identifier
    let satellites = sp3.satellites()?;
    for sat in &satellites {
        println!(
            "{}: {} samples, gaps: {}",
            sat.get_name().unwrap(),
            sat.len(),
            if sat.has_gaps() { "True" } else { "False" }
        );
    }

    let (start, end) = (sp3.start_epoch().unwrap(), sp3.end_epoch().unwrap());
    let station = bh::PointLocation::new(-105.27, 40.01, 1655.0)?.with_name("Boulder");
    let constraint = bh::ElevationConstraint::new(Some(10.0), None)?;

    let windows = bh::location_accesses(
        &station,
        &satellites,
        start,
        end,
        &constraint,
        None,
        None,
    )?;

    println!(
        "\nAccess windows over {:.0} s: {}",
        end - start,
        windows.len()
    );
    for window in &windows {
        println!(
            "{}: {} to {} ({:.1} min)",
            window.satellite_name.as_deref().unwrap_or("?"),
            window.window_open,
            window.window_close,
            window.duration() / 60.0
        );
    }

    Ok(())
}
