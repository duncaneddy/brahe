//! Configuring atmospheric drag with NRLMSISE-00 model.
//! High-fidelity atmospheric model for precision applications.

use brahe as bh;

fn main() {
    // Initialize space weather data provider
    bh::initialize_sw().unwrap();

    // NRLMSISE-00 atmospheric drag configuration
    // - Naval Research Laboratory Mass Spectrometer and Incoherent Scatter Radar
    // - High-fidelity empirical model
    // - Valid from ground to thermospheric heights
    // - Uses space weather data (F10.7, Ap) when available
    // - More computationally expensive than Harris-Priester

    let _drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::NRLMSISE00,
        area: bh::ParameterSource::ParameterIndex(1), // drag_area from params[1]
        cd: bh::ParameterSource::ParameterIndex(2),   // Cd from params[2]
    };
}
