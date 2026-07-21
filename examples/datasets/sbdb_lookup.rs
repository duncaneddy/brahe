//! Resolve a small body in the JPL Small-Body Database (SBDB).
//!
//! Looks up a search string (name or designation) and returns its NAIF/SPK ID
//! and SI physical parameters. Responses are cached under $BRAHE_CACHE/sbdb
//! and reused for 30 days, so this hits the network only once per machine.
//!
//! FLAGS = ["NETWORK"]

use brahe as bh;
use bh::datasets::sbdb::SBDBClient;

fn main() {
    let client = SBDBClient::new();
    let ceres = client.lookup("Ceres").unwrap();

    println!("NAIF ID: {}", ceres.naif_id());
    println!("Full name: {}", ceres.full_name);
    println!("GM: {:.6e} m^3/s^2", ceres.gm.unwrap());
    println!("Radius: {:.1} m", ceres.radius.unwrap());
}
