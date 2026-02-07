//! Demonstrates querying Conjunction Data Messages (CDMs) from Space-Track.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};
use bh::spacetrack::operators;

fn main() {
    // Query high-probability conjunction events (Pc > 1e-3)
    // CDMPublic uses the expandedspacedata controller automatically
    let query = SpaceTrackQuery::new(RequestClass::CDMPublic)
        .filter("PC", &operators::greater_than("1.0e-3"))
        .order_by("TCA", SortOrder::Desc)
        .limit(25);

    let url_path = query.build();
    println!("High-probability CDMs:\n  {}", url_path);
    // High-probability CDMs:
    //   /expandedspacedata/query/class/cdm_public/PC/>1.0e-3/orderby/TCA desc/limit/25/format/json

    // Query CDMs for a specific satellite (e.g., ISS, NORAD 25544)
    let query = SpaceTrackQuery::new(RequestClass::CDMPublic)
        .filter("SAT_1_ID", "25544")
        .order_by("TCA", SortOrder::Desc)
        .limit(10);

    let url_path = query.build();
    println!("\nCDMs involving ISS:\n  {}", url_path);
    // CDMs involving ISS:
    //   /expandedspacedata/query/class/cdm_public/SAT_1_ID/25544/orderby/TCA desc/limit/10/format/json

    // Query upcoming conjunctions within the next 7 days
    let query = SpaceTrackQuery::new(RequestClass::CDMPublic)
        .filter("TCA", &operators::inclusive_range(operators::now(), operators::now_offset(7)))
        .order_by("TCA", SortOrder::Asc);

    let url_path = query.build();
    println!("\nUpcoming conjunctions (next 7 days):\n  {}", url_path);
    // Upcoming conjunctions (next 7 days):
    //   /expandedspacedata/query/class/cdm_public/TCA/now--now+7/orderby/TCA asc/format/json
}
