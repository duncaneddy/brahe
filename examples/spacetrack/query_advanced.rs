//! Demonstrates advanced SpaceTrack query options: ordering, limits, and predicates.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};

fn main() {
    // Order results by epoch descending and limit to 5 records
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", SortOrder::Desc)
        .limit(5);
    println!("Ordered and limited:\n  {}", query.build());

    // Use limit with offset for pagination
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("OBJECT_TYPE", "PAYLOAD")
        .order_by("NORAD_CAT_ID", SortOrder::Asc)
        .limit_offset(10, 20);
    println!("\nPaginated results:\n  {}", query.build());

    // Select specific fields with predicates_filter
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .predicates_filter(&["OBJECT_NAME", "EPOCH", "INCLINATION", "PERIOD"]);
    println!("\nFiltered fields:\n  {}", query.build());

    // Enable metadata and distinct results
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("COUNTRY", "US")
        .distinct(true)
        .metadata(true);
    println!("\nDistinct with metadata:\n  {}", query.build());
}

