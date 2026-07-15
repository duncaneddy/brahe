//! Load a GravityModel sourced from the ICGEM catalog.
//!
//! `GravityModelType::ICGEMModel { body, name }` is interchangeable with the
//! packaged `GravityModelType` variants. The .gfc file is downloaded on first
//! use and cached under `$BRAHE_CACHE/icgem/`.
//!
//! To pin a specific degree variant (when ICGEM publishes multiple), append
//! "-<DEGREE>" to the name, e.g. `name: "XGM2019e_2159-760".to_string()`.
//!
//! FLAGS = ["NETWORK"]

use brahe as bh;
use bh::GravityModelType;
use bh::datasets::icgem::ICGEMBody;

fn main() {
    // Earth — JGM3 (~70x70), small and stable
    let earth_type = GravityModelType::ICGEMModel {
        body: ICGEMBody::Earth,
        name: "JGM3".to_string(),
    };
    let earth_model = bh::GravityModel::from_model_type(&earth_type).unwrap();
    println!(
        "Loaded {} ({}x{}, GM={:.6e} m^3/s^2)",
        earth_model.model_name, earth_model.n_max, earth_model.m_max, earth_model.gm
    );

    // Inspect a coefficient — normalized C_{2,0} (J2 term) from the Earth model
    let (c20, _s20) = earth_model.get(2, 0).unwrap();
    println!("\nEarth normalized C(2,0) = {c20:.6e}");
}
