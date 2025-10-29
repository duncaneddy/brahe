//! Create Epoch instances from Modified Julian Date

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch from MJD
    let mjd = 61041.5; // 2024-01-01 12:00:00 UTC
    let epc2 = bh::Epoch::from_mjd(mjd, bh::TimeSystem::UTC);
    println!("MJD {}: {}", mjd, epc2);
    // MJD 61041.5: 2026-01-01 12:00:00.000 UTC

    // Verify round-trip conversion
    let mjd_out = epc2.mjd();
    println!("Round-trip MJD: {:.6}", mjd_out);
    // Round-trip MJD: 61041.500000
}
