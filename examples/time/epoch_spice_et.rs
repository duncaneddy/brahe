//! Convert an Epoch to SPICE ephemeris time (ET) for kernel queries

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2025, 3, 15, 6, 30, 21.0, 0.0, bh::TimeSystem::UTC);

    // SPICE ephemeris time (ET) is TDB seconds past J2000. spk_position/velocity/state
    // and the *_de functions convert epochs this way internally.
    let et = epc.seconds_past_j2000_as_time_system(bh::TimeSystem::TDB);
    println!("Epoch: {}", epc);
    println!("SPICE ET (TDB seconds past J2000): {:.6}", et);

    // Other time systems are available the same way.
    let tt = epc.seconds_past_j2000_as_time_system(bh::TimeSystem::TT);
    println!("TT seconds past J2000: {:.6}", tt);
    println!("TDB - TT (s): {:.9}", et - tt);
}
