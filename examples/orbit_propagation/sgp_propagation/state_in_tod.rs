//! Convert SGP4 TEME output into TOD with state_in_frame

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::SOrbitStateProvider;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    let epc = prop.epoch + 600.0;
    let state_tod = prop
        .state_in_frame(bh::CelestialFrame::TOD, epc)
        .unwrap();
    let state_gcrf = prop.state_gcrf(epc).unwrap();

    println!("Epoch: {}", epc);
    println!("TOD state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_tod[0], state_tod[1], state_tod[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_tod[3], state_tod[4], state_tod[5]);

    println!("GCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);

    let pos_tod = na::Vector3::new(state_tod[0], state_tod[1], state_tod[2]);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    let pos_diff = (pos_tod - pos_gcrf).norm();
    println!("Position difference norm: {:.3} m", pos_diff);
}
