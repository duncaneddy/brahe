//! Demonstrates querying decay predictions for objects expected to reenter soon.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};
use bh::spacetrack::operators;

fn main() {
    // Get objects predicted to decay within the next 30 days
    // The Decay request class provides reentry predictions
    let query = SpaceTrackQuery::new(RequestClass::Decay)
        .filter("DECAY_EPOCH", &operators::inclusive_range(operators::now(), operators::now_offset(30)))
        .order_by("DECAY_EPOCH", SortOrder::Asc);

    let url_path = query.build();
    println!("Decaying within 30 days:\n  {}", url_path);
    // Decaying within 30 days:
    //   /basicspacedata/query/class/decay/DECAY_EPOCH/now--now+30/orderby/DECAY_EPOCH asc/format/json

    // Get recent actual decays from the past 7 days
    let query = SpaceTrackQuery::new(RequestClass::Decay)
        .filter("DECAY_EPOCH", &operators::inclusive_range(operators::now_offset(-7), operators::now()))
        .filter("MSG_TYPE", "Decay")
        .order_by("DECAY_EPOCH", SortOrder::Desc);

    let url_path = query.build();
    println!("\nRecent decays (last 7 days):\n  {}", url_path);
    // Recent decays (last 7 days):
    //   /basicspacedata/query/class/decay/DECAY_EPOCH/now-7--now/MSG_TYPE/Decay/orderby/DECAY_EPOCH desc/format/json
}
