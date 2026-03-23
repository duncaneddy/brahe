//! Access tile metadata properties after tessellation.
//! Demonstrates reading direction, dimensions, and identifiers from tiles.

#[allow(unused_imports)]
use brahe as bh;
use bh::access::constraints::AscDsc;
use bh::access::location::{AccessibleLocation, PointLocation};
use bh::access::tessellation::{OrbitGeometryTessellator, OrbitGeometryTessellatorConfig, Tessellator};
use bh::propagators::SGPPropagator;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator and tessellator
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    let epoch = prop.epoch;
    let config = OrbitGeometryTessellatorConfig::default()
        .with_asc_dsc(AscDsc::Ascending);
    let tess = OrbitGeometryTessellator::new(
        Box::new(prop), epoch, config, Some("ISS".to_string()),
    );

    // Tessellate a point and inspect properties
    let point = PointLocation::new(10.0, 30.0, 0.0);
    let tiles = tess.tessellate(&point).unwrap();
    let props = tiles[0].properties();

    // Along-track direction (unit vector in ECEF)
    let dir = props["tile_direction"].as_array().unwrap();
    let dx = dir[0].as_f64().unwrap();
    let dy = dir[1].as_f64().unwrap();
    let dz = dir[2].as_f64().unwrap();
    let mag = (dx * dx + dy * dy + dz * dz).sqrt();
    println!("tile_direction: [{:.4}, {:.4}, {:.4}]", dx, dy, dz);
    println!("  magnitude: {:.6}", mag);

    // Tile dimensions
    println!("tile_width: {:.0} m", props["tile_width"].as_f64().unwrap());
    println!("tile_length: {:.0} m", props["tile_length"].as_f64().unwrap());
    println!("tile_area: {:.0} m^2", props["tile_area"].as_f64().unwrap());

    // Group and spacecraft identifiers
    let group_id = props["tile_group_id"].as_str().unwrap();
    println!("tile_group_id: {}...", &group_id[..8]);
    println!("spacecraft_ids: {}", props["spacecraft_ids"]);
}

