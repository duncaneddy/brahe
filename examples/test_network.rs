//! Test file with NETWORK flag - should be skipped by default.
//!
//! This file tests that the FLAG system correctly skips NETWORK examples
//! unless --network is passed to the test command.

// FLAGS = ["NETWORK"]

fn main() {
    println!("NETWORK flag test: This should only run with --network flag");
}
