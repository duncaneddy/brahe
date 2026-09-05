//! Get the MOD to TOD rotation matrix and compute the nutation angle it represents

use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    let r_mod_to_tod = bh::rotation_mod_to_tod(epc);

    println!("\nMOD to TOD rotation matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_mod_to_tod[(0, 0)], r_mod_to_tod[(0, 1)], r_mod_to_tod[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_mod_to_tod[(1, 0)], r_mod_to_tod[(1, 1)], r_mod_to_tod[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", r_mod_to_tod[(2, 0)], r_mod_to_tod[(2, 1)], r_mod_to_tod[(2, 2)]);

    let trace = r_mod_to_tod[(0, 0)] + r_mod_to_tod[(1, 1)] + r_mod_to_tod[(2, 2)];
    let nutation_angle = ((trace - 1.0) / 2.0).acos().to_degrees() * 3600.0;
    println!("Nutation angle (rotation angle of the MOD -> TOD matrix): {:.3} arcsec", nutation_angle);
}
