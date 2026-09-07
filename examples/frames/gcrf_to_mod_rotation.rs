//! Get the GCRF to MOD rotation matrix and show that it reduces to the frame bias at J2000

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    let r_gcrf_to_mod = bh::rotation_gcrf_to_mod(epc);

    println!("\nGCRF to MOD rotation matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_mod[(0, 0)], r_gcrf_to_mod[(0, 1)], r_gcrf_to_mod[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_mod[(1, 0)], r_gcrf_to_mod[(1, 1)], r_gcrf_to_mod[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", r_gcrf_to_mod[(2, 0)], r_gcrf_to_mod[(2, 1)], r_gcrf_to_mod[(2, 2)]);

    let epc_j2000 = bh::Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::TT);
    println!("J2000 epoch: {}", epc_j2000);

    let r_gcrf_to_mod_j2000 = bh::rotation_gcrf_to_mod(epc_j2000);

    println!("\nGCRF to MOD rotation matrix at J2000:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_mod_j2000[(0, 0)], r_gcrf_to_mod_j2000[(0, 1)], r_gcrf_to_mod_j2000[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_mod_j2000[(1, 0)], r_gcrf_to_mod_j2000[(1, 1)], r_gcrf_to_mod_j2000[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", r_gcrf_to_mod_j2000[(2, 0)], r_gcrf_to_mod_j2000[(2, 1)], r_gcrf_to_mod_j2000[(2, 2)]);

    let b = bh::bias_eme2000();
    let max_diff = (r_gcrf_to_mod_j2000 - b).abs().max();
    println!("Comparison with the EME2000 frame bias matrix at J2000:");
    println!("  Max absolute difference: {:.2e}", max_diff);

    println!("\nNote: at J2000 the IAU 2000 precession is identity, so MOD reduces");
    println!("to the constant frame bias between GCRF and EME2000.");
}
