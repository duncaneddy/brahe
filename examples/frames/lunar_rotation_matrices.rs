//! Demonstrate lunar reference frame rotation matrices

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Get constant bias matrix
    let b = bh::bias_moon_j2000();

    // Get rotation matrices (LCRF ↔ MOON_J2000)
    let r_lcrf_to_j2000 = bh::rotation_lcrf_to_moon_j2000();
    let r_j2000_to_lcrf = bh::rotation_moon_j2000_to_lcrf();

    // Using LCI alias
    let r_lci_to_j2000 = bh::rotation_lci_to_moon_j2000();  // Same as LCRF version

    println!("Bias matrix:");
    println!("{:.6}", b);
    println!("\nLCRF to MOON_J2000 rotation:");
    println!("{:.6}", r_lcrf_to_j2000);
    println!("\nMOON_J2000 to LCRF rotation:");
    println!("{:.6}", r_j2000_to_lcrf);
    println!("\nLCI to MOON_J2000 rotation (same as LCRF):");
    println!("{:.6}", r_lci_to_j2000);
}
