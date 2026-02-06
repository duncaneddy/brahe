//! Demonstrates querying the latest GP data for all non-decayed (active) objects.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};
use bh::spacetrack::operators;

fn main() {
    // Get latest GP for all non-decayed objects
    // DECAY_DATE = null-val means the object has not decayed
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("DECAY_DATE", &operators::null_val())
        .order_by("NORAD_CAT_ID", SortOrder::Asc);

    let url_path = query.build();
    println!("All non-decayed objects:\n  {}", url_path);
    // All non-decayed objects:
    //   /basicspacedata/query/class/gp/DECAY_DATE/null-val/orderby/NORAD_CAT_ID asc/format/json

    // Filter to only active payloads (exclude debris and rocket bodies)
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("DECAY_DATE", &operators::null_val())
        .filter("OBJECT_TYPE", "PAYLOAD")
        .order_by("NORAD_CAT_ID", SortOrder::Asc);

    let url_path = query.build();
    println!("\nActive payloads only:\n  {}", url_path);
    // Active payloads only:
    //   /basicspacedata/query/class/gp/DECAY_DATE/null-val/OBJECT_TYPE/PAYLOAD/orderby/NORAD_CAT_ID asc/format/json

    // Filter to active objects in LEO (period under 128 minutes)
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("DECAY_DATE", &operators::null_val())
        .filter("PERIOD", &operators::less_than("128"))
        .order_by("NORAD_CAT_ID", SortOrder::Asc);

    let url_path = query.build();
    println!("\nActive LEO objects:\n  {}", url_path);
    // Active LEO objects:
    //   /basicspacedata/query/class/gp/DECAY_DATE/null-val/PERIOD/<128/orderby/NORAD_CAT_ID asc/format/json
}
