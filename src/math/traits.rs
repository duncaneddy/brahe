/*!
 * Traits for mathematical operations and type conversions.
 */

use nalgebra as na;

/// Trait for types that can provide a 3D position vector.
///
/// This trait enables functions to accept either a 3D position vector (`SVector<f64, 3>`)
/// or a 6D state vector (`SVector<f64, 6>`) without requiring manual slicing at call sites.
///
/// # Examples
///
/// Using with a position vector:
/// ```
/// use brahe::math::traits::IntoPosition;
/// use nalgebra as na;
///
/// fn compute_distance<P: IntoPosition>(r: P) -> f64 {
///     let pos = r.position();
///     pos.norm()
/// }
///
/// let r = na::Vector3::new(7000e3, 0.0, 0.0);
/// let distance = compute_distance(r);
/// assert!((distance - 7000e3).abs() < 1e-6);
/// ```
///
/// Using with a state vector:
/// ```
/// use brahe::math::traits::IntoPosition;
/// use nalgebra as na;
///
/// fn compute_distance<P: IntoPosition>(r: P) -> f64 {
///     let pos = r.position();
///     pos.norm()
/// }
///
/// let x = na::SVector::<f64, 6>::new(7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0);
/// let distance = compute_distance(x);
/// assert!((distance - 7000e3).abs() < 1e-6);
/// ```
pub trait IntoPosition {
    /// Extract the position component as a 3D vector.
    ///
    /// # Returns
    ///
    /// * `Vector3<f64>` - The 3D position vector in meters.
    fn position(&self) -> na::Vector3<f64>;
}

impl IntoPosition for na::Vector3<f64> {
    fn position(&self) -> na::Vector3<f64> {
        *self
    }
}

impl IntoPosition for na::SVector<f64, 6> {
    fn position(&self) -> na::Vector3<f64> {
        self.fixed_rows::<3>(0).into()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_into_position_vector3() {
        let r = na::Vector3::new(1.0, 2.0, 3.0);
        let pos = r.position();
        assert_eq!(pos, na::Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_into_position_vector6() {
        let x = na::SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let pos = x.position();
        assert_eq!(pos, na::Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_into_position_generic_function() {
        fn get_norm<P: IntoPosition>(r: P) -> f64 {
            r.position().norm()
        }

        let r3 = na::Vector3::new(3.0, 4.0, 0.0);
        assert!((get_norm(r3) - 5.0).abs() < 1e-10);

        let r6 = na::SVector::<f64, 6>::new(3.0, 4.0, 0.0, 10.0, 20.0, 30.0);
        assert!((get_norm(r6) - 5.0).abs() < 1e-10);
    }
}
