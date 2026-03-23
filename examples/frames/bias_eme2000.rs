//! Get the EME2000 frame bias matrix and verify its relationship to rotation functions

use brahe as bh;

fn main() {
    // Get the EME2000 frame bias matrix
    let b = bh::bias_eme2000();

    println!("EME2000 frame bias matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", b[(0, 0)], b[(0, 1)], b[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", b[(1, 0)], b[(1, 1)], b[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", b[(2, 0)], b[(2, 1)], b[(2, 2)]);
}

