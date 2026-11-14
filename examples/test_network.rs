//! Test file with NETWORK flag - should be skipped by default.
//!
//! This file tests that the FLAG system correctly skips NETWORK examples
//! unless --network is passed to the test command.

// FLAGS = ["NETWORK"]

// The example runner adds brahe as a dependency of every Rust example.
use brahe as _;

fn main() {
    println!("NETWORK flag test: This should only run with --network flag");
}
