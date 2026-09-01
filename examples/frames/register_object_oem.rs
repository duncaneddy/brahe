//! Register an OEM ephemeris as an object with `OEM::register_for`, then
//! query it through the object's RTN orbit-relative frame.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OEM;
use brahe::frames::{CelestialFrame, Frame};
use nalgebra::Vector6;

fn main() {
    bh::clear_object_registry();

    // OEM::register_for is a one-liner: it converts the ephemeris segment to
    // a trajectory, wraps it as a state provider, and registers it under a
    // name.
    let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
    oem.register_for("ISS").unwrap();
    println!("Registered objects: {:?}", bh::registered_objects());

    // The registered object anchors Frame::RTN("ISS"): its origin is the
    // object's GCRF position, interpolated from the OEM ephemeris.
    let epc = oem.segments[0].metadata.start_time + 300.0;
    let x_rtn_origin = bh::state_frame_to_frame(
        Frame::RTN("ISS"),
        CelestialFrame::GCRF,
        epc,
        Vector6::zeros(),
    )
    .unwrap();
    let p = x_rtn_origin.fixed_rows::<3>(0).into_owned() / 1e3;
    println!("\nISS position at {}: {:.3?} km", epc, p.as_slice());

    bh::clear_object_registry();
    println!("\nExample validated successfully!");
}
