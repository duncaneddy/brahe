/*!
 Acceleration due to non-Newtionian gravitational effects. Specifically, special and general
relativity.
 */

use nalgebra::{Vector3, Vector6};

use crate::constants::{C_LIGHT, GM_EARTH};

/// Calculate the acceleration due to special and general relativity for an Earth orbiting object.
///
/// # Arguments
///
/// - `x_object`: State vector of the object in the ECI frame.
///
/// # Returns
///
/// - `a_relativity` : Acceleration due to special and general relativity.
///
/// # Examples
///
/// ```
/// use brahe::coordinates::state_osculating_to_cartesian;
/// use brahe::orbit_dynamics::acceleration_relativity;
/// use nalgebra::Vector6;
/// use brahe::R_EARTH;
///
/// let x_object = Vector6::new(R_EARTH + 500.0e3, 0.0, 0.0, 0.0, 0.0, 0.0);
/// let a_relativity = acceleration_relativity(x_object);
/// ```
pub fn acceleration_relativity(x_object: Vector6<f64>) -> Vector3<f64> {
    // Extract state variables
    let r = x_object.fixed_rows::<3>(0);
    let v = x_object.fixed_rows::<3>(3);

    // Intermediate computations
    let norm_r = r.norm();
    let r2 = norm_r.powi(2);
    let norm_v = v.norm();
    let v2 = norm_v.powi(2);
    let c2 = C_LIGHT.powi(2);

    // Compute unit vectors
    let er = r / norm_r;
    let ev = v / norm_v;

    // Compute perturbation acceleration and return
    GM_EARTH / r2 * ((4.0 * GM_EARTH / (c2 * norm_r) - v2 / c2) * er + 4.0 * v2 / c2 * er.dot(&ev) * ev)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_acceleration_relativity() {
        // TODO: Add tests
    }
}