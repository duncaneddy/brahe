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

    // Build a SATCAT query for US-owned objects
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("COUNTRY", "US");

    let url_path = query.build();
    println!("\nSATCAT query URL path:\n  {}", url_path);

    // The default controller is inferred from the request class
    let query = SpaceTrackQuery::new(RequestClass::CDMPublic);
    let url_path = query.build();
    println!("\nCDM query URL path (uses expandedspacedata controller):\n  {}", url_path);
}

