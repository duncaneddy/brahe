//! Create a PolygonLocation from a list of vertices.
//! Demonstrates polygon construction and center/vertex access.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use bh::AccessibleLocation;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define polygon vertices (lon, lat, alt in degrees and meters)
    let vertices = vec![
        na::SVector::<f64, 3>::new(-122.5, 37.7, 0.0),
        na::SVector::<f64, 3>::new(-122.35, 37.7, 0.0),
        na::SVector::<f64, 3>::new(-122.35, 37.8, 0.0),
        na::SVector::<f64, 3>::new(-122.5, 37.8, 0.0),
        na::SVector::<f64, 3>::new(-122.5, 37.7, 0.0),
    ];

    let polygon = bh::PolygonLocation::new(vertices).unwrap()
        .with_name("SF Region");

    let center = polygon.center_geodetic();
    println!("Name: {}", polygon.get_name().unwrap_or_default());
    println!("Vertices: {}", polygon.num_vertices());
    println!("Center: ({:.4}, {:.4})", center[0], center[1]);

    // Expected output:
    // Name: SF Region
    // Vertices: 4
    // Center: (-122.4250, 37.7500)
}
