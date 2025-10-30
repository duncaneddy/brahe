//! Convert between numeric and Alpha-5 NORAD ID formats.
//!
//! For NORAD catalog numbers >= 100000, TLEs use the Alpha-5 format which encodes
//! large numbers into 5 characters using letters A-Z (excluding I and O to avoid
//! confusion with 1 and 0).

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    println!("NORAD ID Format Conversions\n");

    // Parse NORAD IDs in different formats
    println!("Parsing NORAD IDs:");
    let norad_numeric = bh::parse_norad_id("25544").unwrap();  // Numeric format
    println!("  '25544' -> {}", norad_numeric);

    let norad_alpha5 = bh::parse_norad_id("A0001").unwrap();  // Alpha-5 format
    println!("  'A0001' -> {}", norad_alpha5);

    // Convert numeric to Alpha-5 (only works for IDs >= 100000)
    println!("\nNumeric to Alpha-5:");
    match bh::norad_id_numeric_to_alpha5(25544) {
        Ok(alpha5) => println!("  25544 -> {}", alpha5),
        Err(e) => println!("  25544 -> Error: {}", e),
    }

    let alpha5_high = bh::norad_id_numeric_to_alpha5(100000).unwrap();
    println!("  100000 -> {}", alpha5_high);

    let alpha5_higher = bh::norad_id_numeric_to_alpha5(123456).unwrap();
    println!("  123456 -> {}", alpha5_higher);

    // Convert Alpha-5 to numeric
    println!("\nAlpha-5 to Numeric:");
    let numeric_1 = bh::norad_id_alpha5_to_numeric("A0001").unwrap();
    println!("  'A0001' -> {}", numeric_1);

    let numeric_2 = bh::norad_id_alpha5_to_numeric("L0000").unwrap();
    println!("  'L0000' -> {}", numeric_2);

    // Round-trip conversion
    println!("\nRound-trip Conversion:");
    let original = 200000;
    let alpha5 = bh::norad_id_numeric_to_alpha5(original).unwrap();
    let back_to_numeric = bh::norad_id_alpha5_to_numeric(&alpha5).unwrap();
    println!("  {} -> '{}' -> {}", original, alpha5, back_to_numeric);
    println!("  Match: {}", original == back_to_numeric);

    // Expected output:
    // NORAD ID Format Conversions
    //
    // Parsing NORAD IDs:
    //   '25544' -> 25544
    //   'A0001' -> 100001
    //
    // Numeric to Alpha-5:
    //   25544 -> Error: NORAD ID 25544 is out of Alpha-5 range (100000-339999)
    //   100000 -> A0000
    //   123456 -> C3456
    //
    // Alpha-5 to Numeric:
    //   'A0001' -> 100001
    //   'L0000' -> 200000
    //
    // Round-trip Conversion:
    //   200000 -> 'L0000' -> 200000
    //   Match: true
}
