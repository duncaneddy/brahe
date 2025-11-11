//! Get the EME2000 frame bias matrix and verify its relationship to rotation functions

use brahe as bh;

fn main() {
    // Get the EME2000 frame bias matrix
    let b = bh::bias_eme2000();

    println!("EME2000 frame bias matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", b[(0, 0)], b[(0, 1)], b[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", b[(1, 0)], b[(1, 1)], b[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", b[(2, 0)], b[(2, 1)], b[(2, 2)]);
    // EME2000 frame bias matrix:
    //   [ 1.0000000000, -0.0000000708,  0.0000000806]
    //   [ 0.0000000708,  1.0000000000,  0.0000000331]
    //   [-0.0000000806, -0.0000000331,  1.0000000000]
}
