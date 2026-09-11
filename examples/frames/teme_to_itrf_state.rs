//! Transform a TEME state vector from an SGP4 propagator into the ITRF

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::SStateProvider;

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    let epc = prop.epoch + 600.0;
    let state_teme = prop.state(epc).unwrap();
    println!("Epoch: {}", epc);
    println!("TEME state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_teme[0], state_teme[1], state_teme[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_teme[3], state_teme[4], state_teme[5]);

    let state_itrf = bh::state_teme_to_itrf(epc, state_teme);
    println!("ITRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_itrf[0], state_itrf[1], state_itrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_itrf[3], state_itrf[4], state_itrf[5]);

    let speed_teme = state_teme.fixed_rows::<3>(3).norm();
    let speed_itrf = state_itrf.fixed_rows::<3>(3).norm();
    println!("Inertial speed: {:.3} m/s, Earth-fixed speed: {:.3} m/s", speed_teme, speed_itrf);
}
