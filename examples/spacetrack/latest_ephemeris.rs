//! Demonstrates querying the most recent GP ephemeris for a single object.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};

fn main() {
    // Get the latest GP record for the ISS (NORAD 25544)
    // Order by EPOCH descending so the most recent is first, limit to 1
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", SortOrder::Desc)
        .limit(1);

    let url_path = query.build();
    println!("Latest GP for ISS:\n  {}", url_path);

    // Get the latest GP for a Starlink satellite (NORAD 48274)
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "48274")
        .order_by("EPOCH", SortOrder::Desc)
        .limit(1);

    let url_path = query.build();
    println!("\nLatest GP for Starlink-2541:\n  {}", url_path);
}

