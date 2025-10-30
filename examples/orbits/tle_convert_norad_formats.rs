//! Convert between numeric and Alpha-5 NORAD ID formats.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    println!("NORAD ID Format Conversions\n");

    // Convert numeric to Alpha-5 (only works for IDs >= 100000)
    println!("Numeric to Alpha-5:");
    let alpha5_low = bh::norad_id_numeric_to_alpha5(25544).unwrap();
    println!("  25544 -> {}", alpha5_low);

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
    // Numeric to Alpha-5:
    //   25544 -> 25544
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
