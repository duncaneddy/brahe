//! Demonstrates setting different output formats on SpaceTrack queries.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{SpaceTrackQuery, RequestClass, OutputFormat};

fn main() {
    // Default format is JSON
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544");
    println!("Default (JSON):\n  {}", query.build());
    // Default (JSON):
    //   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/json

    // Request TLE format for direct TLE text output
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .format(OutputFormat::TLE);
    println!("\nTLE format:\n  {}", query.build());
    // TLE format:
    //   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/tle

    // Request CSV format for spreadsheet-compatible output
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("COUNTRY", "US")
        .limit(10)
        .format(OutputFormat::CSV);
    println!("\nCSV format:\n  {}", query.build());
    // CSV format:
    //   /basicspacedata/query/class/satcat/COUNTRY/US/limit/10/format/csv

    // Request KVN (CCSDS Keyword-Value Notation) format
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .format(OutputFormat::KVN);
    println!("\nKVN format:\n  {}", query.build());
    // KVN format:
    //   /basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/kvn
}

