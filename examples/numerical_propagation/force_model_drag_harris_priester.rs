//! Configuring atmospheric drag with Harris-Priester model.
//! Fast atmospheric model accounting for diurnal density variations.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // Harris-Priester atmospheric drag configuration
    // - Valid for altitudes 100-1000 km
    // - Accounts for latitude-dependent diurnal bulge
    // - Does not require space weather data (F10.7, Ap)
    
    // Using parameter indices (default layout)
    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(1), // drag_area from params[1]
        cd: bh::ParameterSource::ParameterIndex(2),   // Cd from params[2]
    };
}
