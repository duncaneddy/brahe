//! Demonstrates using SpaceTrack operator functions to build filtered queries.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass};
use bh::spacetrack::operators;

fn main() {
    // Filter by NORAD ID range using inclusive_range
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", &operators::inclusive_range("25544", "25600"));
    println!("Range filter:\n  {}", query.build());

    // Filter for objects with low eccentricity using less_than
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("ECCENTRICITY", &operators::less_than("0.01"))
        .filter("OBJECT_TYPE", "PAYLOAD");
    println!("\nMultiple filters:\n  {}", query.build());

    // Filter for recently launched objects using greater_than + now_offset
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("LAUNCH", &operators::greater_than(operators::now_offset(-30)));
    println!("\nRecent launches (last 30 days):\n  {}", query.build());

    // Search by name pattern using like
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("SATNAME", &operators::like("STARLINK"));
    println!("\nName pattern match:\n  {}", query.build());

    // Filter for multiple NORAD IDs using or_list
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", &operators::or_list(&["25544", "48274", "54216"]));
    println!("\nMultiple IDs:\n  {}", query.build());
}

