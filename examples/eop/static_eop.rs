//! Initialize Static EOP Providers

use brahe as bh;

fn main() {
    // Method 1: Static EOP Provider - All Zeros
    let eop_static_zeros = bh::eop::StaticEOPProvider::from_zero();
    bh::eop::set_global_eop_provider(eop_static_zeros);

    // Method 2: Static EOP Provider - Constant Values
    let eop_static_values = bh::eop::StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));
    bh::eop::set_global_eop_provider(eop_static_values);
}
