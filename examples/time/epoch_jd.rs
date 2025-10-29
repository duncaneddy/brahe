//! Create Epoch instances from Julian Date

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch from JD
    let jd = 2460310.5;
    let epc = bh::Epoch::from_jd(jd, bh::TimeSystem::UTC);
    println!("JD {}: {}", jd, epc);
    // JD 2460310.5: 2024-01-01 00:00:00.000 UTC

    // Verify round-trip conversion
    let jd_out = epc.jd();
    println!("Round-trip JD: {:.10}", jd_out);
    // Round-trip JD: 2460310.5000000000
}
