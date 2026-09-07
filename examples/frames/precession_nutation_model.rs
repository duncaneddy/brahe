//! Select the precession-nutation model and measure the difference between the two models

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    println!("Default precession-nutation model: {}", bh::get_precession_nutation_model());

    let epc = bh::Epoch::from_datetime(2040, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    let r_2006a = bh::rotation_gcrf_to_itrf(epc);

    bh::set_precession_nutation_model(bh::PrecessionNutationModel::IAU2000B);
    println!("Selected precession-nutation model: {}", bh::get_precession_nutation_model());

    let r_2000b = bh::rotation_gcrf_to_itrf(epc);

    // Rotation angle between the two matrices, in the form that keeps its precision
    // for the very small angles separating the two models
    let frobenius = (r_2006a - r_2000b).norm();
    let theta_mas = 2.0 * (frobenius / (2.0 * 2.0_f64.sqrt())).asin() * bh::RAD2AS * 1000.0;

    println!("\nGCRF to ITRF rotation difference between the two models:");
    println!("  Angle: {:.4} mas", theta_mas);

    let rc2i_2006a = bh::bias_precession_nutation_model(epc, bh::PrecessionNutationModel::IAU2006A);
    let rc2i_2000b = bh::bias_precession_nutation_model(epc, bh::PrecessionNutationModel::IAU2000B);

    println!("\nBias-precession-nutation matrices evaluated per call:");
    println!("  Max absolute difference: {:.2e}", (rc2i_2006a - rc2i_2000b).abs().max());

    bh::set_precession_nutation_model(bh::PrecessionNutationModel::IAU2006A);
    println!("\nRestored precession-nutation model: {}", bh::get_precession_nutation_model());
}
