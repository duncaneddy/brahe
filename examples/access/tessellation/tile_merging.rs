//! Merge tessellation tiles from multiple spacecraft with similar ground-track directions.
//! Demonstrates the tile_merge_orbit_geometry function.

#[allow(unused_imports)]
use brahe as bh;
use bh::access::constraints::AscDsc;
use bh::access::location::{AccessibleLocation, PointLocation};
use bh::access::tessellation::{
    OrbitGeometryTessellator, OrbitGeometryTessellatorConfig, Tessellator,
    tile_merge_orbit_geometry,
};
use bh::propagators::SGPPropagator;

fn main() {
    bh::initialize_eop().unwrap();

    // SC-1 and SC-2 TLEs with slightly different inclinations (~1.4 degree offset)
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2_sc1 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let line2_sc2 = "2 25544  53.0000 247.4627 0006703 130.5360 325.0288 15.72125391563532";

    // Create two tessellators with different spacecraft IDs
    let config = OrbitGeometryTessellatorConfig::default()
        .with_asc_dsc(AscDsc::Ascending);

    let prop1 = SGPPropagator::from_tle(line1, line2_sc1, 60.0).unwrap();
    let epoch1 = prop1.epoch;
    let tess1 = OrbitGeometryTessellator::new(
        Box::new(prop1), epoch1, config.clone(), Some("SC-1".to_string()),
    );

    let prop2 = SGPPropagator::from_tle(line1, line2_sc2, 60.0).unwrap();
    let epoch2 = prop2.epoch;
    let tess2 = OrbitGeometryTessellator::new(
        Box::new(prop2), epoch2, config, Some("SC-2".to_string()),
    );

    // Tessellate the same point with both spacecraft
    let point = PointLocation::new(10.0, 30.0, 0.0);
    let tiles_sc1 = tess1.tessellate(&point).unwrap();
    let tiles_sc2 = tess2.tessellate(&point).unwrap();
    let mut all_tiles = tiles_sc1;
    all_tiles.extend(tiles_sc2);

    println!("Before merge: {} tiles", all_tiles.len());

    // Merge tiles with similar directions
    let merged = tile_merge_orbit_geometry(&all_tiles, 200.0, 200.0, 2.0);

    println!("After merge: {} tiles", merged.len());
    for tile in &merged {
        println!("  spacecraft_ids: {}", tile.properties()["spacecraft_ids"]);
    }
}

