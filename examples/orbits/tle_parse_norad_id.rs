//! Parse NORAD IDs in different formats.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse NORAD IDs in different formats
    println!("Parsing NORAD IDs:");

    // Numeric format (standard)
    let norad_numeric = bh::parse_norad_id("25544").unwrap();
    println!("  '25544' -> {}", norad_numeric);

    // Alpha-5 format (for IDs >= 100000)
    let norad_alpha5 = bh::parse_norad_id("A0001").unwrap();
    println!("  'A0001' -> {}", norad_alpha5);

    // Expected output:
    // Parsing NORAD IDs:
    //   '25544' -> 25544
    //   'A0001' -> 100001
}
