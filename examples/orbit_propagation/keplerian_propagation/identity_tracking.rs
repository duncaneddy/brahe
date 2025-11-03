//! Track KeplerianPropagator with names and IDs for multi-satellite scenarios

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);

    // Create propagator with identity (builder pattern)
    let prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    ).with_name("Satellite-A").with_id(12345);

    println!("Name: {:?}", prop.get_name());
    // Name: Some("Satellite-A")
    println!("ID: {:?}", prop.get_id());
    // ID: Some(12345)
    println!("UUID: {:?}", prop.get_uuid());
    // UUID: None (because not set)
}
