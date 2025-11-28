//! Complete overview of ForceModelConfig showing all configuration fields.
//! This example demonstrates every configurable option for force modeling.

use brahe as bh;
use bh::GravityModelType;

fn main() {

    // Create a fully-configured force model
    let force_config = bh::ForceModelConfig {
        // Gravity: Spherical harmonic model (EGM2008, 20x20 degree/order)
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
        },
        // Atmospheric drag: Harris-Priester model with parameter indices
        drag: Some(bh::DragConfiguration {
            model: bh::AtmosphericModel::HarrisPriester,
            area: bh::ParameterSource::ParameterIndex(1), // Index into parameter vector
            cd: bh::ParameterSource::ParameterIndex(2),
        }),
        // Solar radiation pressure: Conical eclipse model
        srp: Some(bh::SolarRadiationPressureConfiguration {
            area: bh::ParameterSource::ParameterIndex(3),
            cr: bh::ParameterSource::ParameterIndex(4),
            eclipse_model: bh::EclipseModel::Conical,
        }),
        // Third-body: Sun and Moon with DE440s ephemeris
        third_body: Some(bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::DE440s,
            bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
        }),
        // General relativistic corrections
        relativity: true,
        // Spacecraft mass (can also use ParameterIndex for estimation)
        mass: Some(bh::ParameterSource::Value(1000.0)), // kg
    };

    println!("Gravity: {:?}", force_config.gravity);
    println!("Drag: {:?}", force_config.drag);
    println!("SRP: {:?}", force_config.srp);
    println!("Third-body: {:?}", force_config.third_body);
    println!("Relativity: {:?}", force_config.relativity);
    println!("Mass: {:?}", force_config.mass);
    // Gravity: SphericalHarmonic { source: ModelType(EGM2008_360), degree: 20, order: 20 }
    // Drag: Some(DragConfiguration { model: HarrisPriester, area: ParameterIndex(1), cd: ParameterIndex(2) })
    // SRP: Some(SolarRadiationPressureConfiguration { area: ParameterIndex(3), cr: ParameterIndex(4), eclipse_model: Conical })
    // Third-body: Some(ThirdBodyConfiguration { ephemeris_source: DE440s, bodies: [Sun, Moon] })
    // Relativity: true
    // Mass: Some(Value(1000.0))
}
