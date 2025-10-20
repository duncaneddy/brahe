//! Test file with IGNORE flag - should be skipped by default.
//!
//! This file tests that the FLAG system correctly skips IGNORE examples
//! unless --ignore is passed to the test command.
// FLAGS = ["IGNORE"]

fn main() {
    println!("IGNORE flag test: This should only run with --ignore flag");
    println!("If you see this during normal test runs, the flag system is broken!");
}
