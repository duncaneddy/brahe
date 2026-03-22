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
    // Ordered and limited:
    //   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/orderby/EPOCH desc/limit/5/format/json

    // Use limit with offset for pagination
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("OBJECT_TYPE", "PAYLOAD")
        .order_by("NORAD_CAT_ID", SortOrder::Asc)
        .limit_offset(10, 20);
    println!("\nPaginated results:\n  {}", query.build());
    // Paginated results:
    //   /basicspacedata/query/class/gp/OBJECT_TYPE/PAYLOAD/orderby/NORAD_CAT_ID asc/limit/10,20/format/json

    // Select specific fields with predicates_filter
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .predicates_filter(&["OBJECT_NAME", "EPOCH", "INCLINATION", "PERIOD"]);
    println!("\nFiltered fields:\n  {}", query.build());
    // Filtered fields:
    //   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/predicates/OBJECT_NAME,EPOCH,INCLINATION,PERIOD/format/json

    // Enable metadata and distinct results
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("COUNTRY", "US")
        .distinct(true)
        .metadata(true);
    println!("\nDistinct with metadata:\n  {}", query.build());
    // Distinct with metadata:
    //   /basicspacedata/query/class/satcat/COUNTRY/US/metadata/true/distinct/true/format/json
}

