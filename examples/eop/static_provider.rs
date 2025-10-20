//! Using StaticEOPProvider for testing or offline environments.
//!
//! Demonstrates:
//! - Creating static EOP provider
//! - Setting global EOP provider
//! - Use cases for static EOP

use brahe::eop::*;
use std::path::Path;
fn main() {
    // Use built-in static data (all zeros)
    let provider = StaticEOPProvider::from_zero();
    set_global_eop_provider(provider);

    println!("Static EOP provider initialized");
    println!("Use case: Testing, offline environments, or when high precision not critical");
}
