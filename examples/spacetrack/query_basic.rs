//! Demonstrates basic SpaceTrack query construction for GP and SATCAT data.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass};

fn main() {
    // Build a GP query for the ISS by NORAD catalog ID
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544");

    let url_path = query.build();
    println!("GP query URL path:\n  {}", url_path);
    // GP query URL path:
    //   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/json

    // Build a SATCAT query for US-owned objects
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("COUNTRY", "US");

    let url_path = query.build();
    println!("\nSATCAT query URL path:\n  {}", url_path);
    // SATCAT query URL path:
    //   /basicspacedata/query/class/satcat/COUNTRY/US/format/json

    // The default controller is inferred from the request class
    let query = SpaceTrackQuery::new(RequestClass::CDMPublic);
    let url_path = query.build();
    println!("\nCDM query URL path (uses expandedspacedata controller):\n  {}", url_path);
    // CDM query URL path (uses expandedspacedata controller):
    //   /expandedspacedata/query/class/cdm_public/format/json
}

