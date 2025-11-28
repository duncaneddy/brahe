//! Using ParameterSource::ParameterIndex for parameter vector values.
//! Parameters that can be varied or estimated during propagation.

use brahe as bh;

fn main() {
    // ParameterSource::ParameterIndex references a value in the parameter vector
    // Use when parameters may change or need to be estimated

    // Default parameter layout:
    // Index 0: mass (kg)
    // Index 1: drag_area (m^2)
    // Index 2: Cd (dimensionless)
    // Index 3: srp_area (m^2)
    // Index 4: Cr (dimensionless)

    let _drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(1), // params[1] = drag_area
        cd: bh::ParameterSource::ParameterIndex(2),   // params[2] = Cd
    };

    let _srp_config = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(3), // params[3] = srp_area
        cr: bh::ParameterSource::ParameterIndex(4),   // params[4] = Cr
        eclipse_model: bh::EclipseModel::Conical,
    };

    // Custom parameter layout example
    println!("\nCustom parameter layout example:");
    println!("  You can map parameters to any indices:");
    let _custom_drag = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(5),  // Custom index
        cd: bh::ParameterSource::ParameterIndex(10),   // Custom index
    };
}
