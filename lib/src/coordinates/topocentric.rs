/*!
 * Provides topocentric coordiante transformations.
 */

use nalgebra::Vector3;
use std::f64::consts::PI;

use crate::constants;
use crate::utils::math::{from_degrees, to_degrees};
use crate::coordinates::types::EllipsoidalConversionType;



#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

}