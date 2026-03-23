//! Tessellate a point location into rectangular tiles aligned with satellite ground tracks.
//! Demonstrates point tessellation with an ascending-only configuration.

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

    // Tessellate a point
    let point = PointLocation::new(10.0, 30.0, 0.0);
    let tiles = tess.tessellate(&point).unwrap();

    println!("Number of tiles: {}", tiles.len());
    for (i, tile) in tiles.iter().enumerate() {
        let center = tile.center_geodetic();
        let props = tile.properties();
        println!("Tile {}: center=({:.4}, {:.4})", i, center[0], center[1]);
        println!("  width={:.0} m, length={:.0} m",
            props["tile_width"].as_f64().unwrap(),
            props["tile_length"].as_f64().unwrap(),
        );
    }

}

