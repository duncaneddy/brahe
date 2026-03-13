//! Tessellate a polygon location into rectangular tiles aligned with satellite ground tracks.
//! Demonstrates polygon tessellation with an ascending-only configuration.

#[allow(unused_imports)]
use brahe as bh;
use bh::access::constraints::AscDsc;
use bh::access::location::{AccessibleLocation, PolygonLocation};
use bh::access::tessellation::{OrbitGeometryTessellator, OrbitGeometryTessellatorConfig, Tessellator};
use bh::propagators::SGPPropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Create propagator and tessellator
    let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    let epoch = prop.epoch();
    let config = OrbitGeometryTessellatorConfig::default()
        .with_asc_dsc(AscDsc::Ascending);
    let tess = OrbitGeometryTessellator::new(
        Box::new(prop), epoch, config, Some("ISS".to_string()),
    );

    // Define a small polygon (approximately 0.1 deg x 0.1 deg)
    let vertices = vec![
        na::SVector::<f64, 3>::new(10.0, 30.0, 0.0),
        na::SVector::<f64, 3>::new(10.1, 30.0, 0.0),
        na::SVector::<f64, 3>::new(10.1, 30.1, 0.0),
        na::SVector::<f64, 3>::new(10.0, 30.1, 0.0),
    ];
    let polygon = PolygonLocation::new(vertices).unwrap();

    // Tessellate the polygon
    let tiles = tess.tessellate(&polygon).unwrap();

    println!("Number of tiles: {}", tiles.len());
    for (i, tile) in tiles.iter().enumerate() {
        let props = tile.properties();
        let group_id = props["tile_group_id"].as_str().unwrap();
        println!("Tile {}: group_id={}... length={:.0} m",
            i, &group_id[..8],
            props["tile_length"].as_f64().unwrap(),
        );
    }
}
