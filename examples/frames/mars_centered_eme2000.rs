//! Loading a CENTER_NAME/REF_FRAME pair that names a non-Earth origin.
//!
//! FLAGS = ["NETWORK"]
//!
//! OEMExample4.txt declares CENTER_NAME = MARS with REF_FRAME = EME2000:
//! EME2000 orientation about Mars rather than Earth. OEM::to_trajectories
//! resolves that pair to CelestialFrame::Centered(EME2000, MARS) instead of
//! the Earth-centered EME2000 shorthand. Converting a sample to MCI only
//! rotates (no ephemeris needed, since both frames share Mars as their
//! center); converting to GCRF also translates by the Earth-Mars vector,
//! which requires the DE440s planetary ephemeris loaded below.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OEM;
use brahe::frames::{CelestialFrame, ReferenceFrame};
use brahe::traits::Trajectory;

fn main() {
    bh::initialize_eop().unwrap();
    bh::load_common_spice_kernels().unwrap();

    let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
    let seg = &oem.segments[0];
    println!(
        "REF_FRAME = {}, CENTER_NAME = {}",
        seg.metadata.ref_frame, seg.metadata.center_name
    );

    let traj = oem.to_trajectories().unwrap().remove(0);
    let ReferenceFrame::Celestial(traj_frame) = &traj.frame else {
        panic!("OEM trajectory frame is always celestial");
    };
    let traj_frame = *traj_frame;
    println!("\nTrajectory frame: {}", traj_frame);
    println!("  Axes:   {}", traj_frame.axes());
    println!("  Center: {}", traj_frame.center());

    // Same-center conversion: EME2000 to MCI is a rotation only, both
    // centered on Mars.
    let traj_mci = traj.to_frame(CelestialFrame::MCI).unwrap();
    let (epc, x_mci) = traj_mci.first().unwrap();
    println!("\nState at {} in MCI (Mars-centered):", epc);
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        x_mci[0] / 1e3,
        x_mci[1] / 1e3,
        x_mci[2] / 1e3
    );
    println!(
        "  Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        x_mci[3], x_mci[4], x_mci[5]
    );

    // Cross-center conversion: EME2000/Mars to GCRF/Earth adds the
    // Earth-Mars translation from the loaded SPK kernels.
    let traj_gcrf = traj.to_frame(CelestialFrame::GCRF).unwrap();
    let (_, x_gcrf) = traj_gcrf.first().unwrap();
    println!("\nState at {} in GCRF (Earth-centered):", epc);
    println!(
        "  Position (km): [{:.3e}, {:.3e}, {:.3e}]",
        x_gcrf[0] / 1e3,
        x_gcrf[1] / 1e3,
        x_gcrf[2] / 1e3
    );
    println!(
        "  Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        x_gcrf[3], x_gcrf[4], x_gcrf[5]
    );

    assert_eq!(traj_frame.center(), bh::NAIFId::Mars);
    assert_eq!(traj_frame.axes(), bh::FrameAxes::EME2000);
    println!("\nExample validated successfully!");
}
