//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! nalgebra = "0.33"
//! ```

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define satellite position and get Sun position
    let epc = bh::Epoch::from_date(2024, 1, 1, bh::TimeSystem::UTC);
    let r_sat = na::Vector3::new(bh::R_EARTH + 400e3, 0.0, 0.0);
    let r_sun = bh::sun_position(epc);

    // Check eclipse using conical model
    let nu_conical = bh::eclipse_conical(r_sat, r_sun);
    println!("Conical illumination fraction: {:.4}", nu_conical);

    // Check eclipse using cylindrical model
    let nu_cyl = bh::eclipse_cylindrical(r_sat, r_sun);
    println!("Cylindrical illumination: {:.1}", nu_cyl);

    if nu_conical == 0.0 {
        println!("Satellite in full shadow (umbra)");
    } else if nu_conical == 1.0 {
        println!("Satellite in full sunlight");
    } else {
        println!("Satellite in penumbra ({:.1}% illuminated)", nu_conical * 100.0);
    }
}

// Expected output:
// Conical illumination fraction: 1.0000
// Cylindrical illumination: 1.0
// Satellite in full sunlight
