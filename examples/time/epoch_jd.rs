//! Create Epoch instances from Julian Date

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch from JD
    let jd = 2460310.5;
    let epc = bh::Epoch::from_jd(jd, bh::TimeSystem::UTC);
    println!("JD {}: {}", jd, epc);

    // Verify round-trip conversion
    let jd_out = epc.jd();
    println!("Round-trip JD: {:.10}", jd_out);
}

