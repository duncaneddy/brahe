//! Access satellite state in PEF (Pseudo-Earth-Fixed) frame

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Get state in PEF frame (TEME rotated by GMST)
    let state_pef = prop.state_pef(prop.epoch);
    println!("PEF position: {:?}", state_pef.fixed_rows::<3>(0) / 1e3);
    // Expected output:
    // PEF position: [[-3953.2057482107907, 1427.514600436758, 5243.614536966578]]
}
