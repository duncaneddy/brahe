/*!
Module for the third body perturbations. Also provides low-precession models for the Sun and Moon
ephemerides.
 */

use nalgebra::Vector3;

use crate::{GM_MOON, GM_SUN};
use crate::ephemerides::{moon_position, sun_position};
use crate::orbit_dynamics::gravity::acceleration_point_mass_gravity;
use crate::time::Epoch;

/// Calculate the acceleration due to the Sun on an object at a given epoch.
/// The calculation is performed using the point-mass gravity model and the
/// low-precision analytical ephemerides for the Sun position implemented in
/// the `ephemerides` module.
///
/// Should a more accurate calculation be required, you can utilize the
/// point-mass gravity model and a higher-precision ephemerides for the Sun.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Sun's position
/// * `r_object` - Position of the object in the GCRF frame. Units: [m]
///
/// # Returns
///
/// * `a` - Acceleration due to the Sun. Units: [m/s^2]
///
/// # Example
///
/// ```
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::third_body_sun;
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = third_body_sun(epc, &r_object);
/// ```
pub fn third_body_sun(epc: Epoch, r_object: &Vector3<f64>) -> Vector3<f64> {
    acceleration_point_mass_gravity(r_object, &sun_position(epc), GM_SUN)
}

/// Calculate the acceleration due to the Moon on an object at a given epoch.
/// The calculation is performed using the point-mass gravity model and the
/// low-precision analytical ephemerides for the Moon position implemented in
/// the `ephemerides` module.
///
/// Should a more accurate calculation be required, you can utilize the
/// point-mass gravity model and a higher-precision ephemerides for the Moon.
///
/// # Arguments
///
/// - `epc` - Epoch at which to calculate the Moon's position
/// - `r_object` - Position of the object in the GCRF frame. Units: [m]
///
/// # Returns
///
/// - `a` - Acceleration due to the Moon. Units: [m/s^2]
///
pub fn third_body_moon(epc: Epoch, r_object: &Vector3<f64>) -> Vector3<f64> {
    acceleration_point_mass_gravity(r_object, &moon_position(epc), GM_MOON)
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector6;

    use crate::constants::R_EARTH;
    use crate::coordinates::*;
    use crate::TimeSystem;

    use super::*;

    #[test]
    fn test_third_body_sun() {
        let epc = Epoch::from_date(2023, 1, 1, TimeSystem::UTC);

        let oe = Vector6::new(
            R_EARTH + 500e3,
            0.01,
            97.3,
            15.0,
            30.0,
            45.0,
        );
        let r_object = state_osculating_to_cartesian(oe, true).xyz();

        let a = third_body_sun(epc, &r_object);

        // TODO: Do better validation of the implementation and the expected results
        assert!(a.norm() < 1.0e-1);
    }

    #[test]
    fn test_third_body_moon() {
        let epc = Epoch::from_date(2023, 1, 1, TimeSystem::UTC);

        let oe = Vector6::new(
            R_EARTH + 500e3,
            0.01,
            97.3,
            15.0,
            30.0,
            45.0,
        );
        let r_object = state_osculating_to_cartesian(oe, true).xyz();

        let a = third_body_moon(epc, &r_object);

        // TODO: Do better validation of the implementation and the expected results
        assert!(a.norm() < 1.0e-4);
    }
}