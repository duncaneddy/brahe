/*!
The `rotation_matrix` module provides the `RotationMatrix` struct, which represents an attitude transformation in the form of a
3x3 rotation matrix.
*/

use std::{fmt, ops};
use nalgebra::{Matrix3, Vector3};

use crate::constants::{DEG2RAD, RAD2DEG};
use crate::attitude::ToAttitude;
use crate::attitude::attitude_types::{Quaternion, EulerAngle, EulerAxis, EulerAngleOrder, RotationMatrix};
use crate::FromAttitude;

impl RotationMatrix {
    pub fn new(matrix: Matrix3<f64>) -> Self {
        Self { data: matrix }
    }

    pub fn from_matrix(matrix: Matrix3<f64>) -> Self {
        Self::new(matrix.clone())
    }

    pub fn to_matrix(&self) -> Matrix3<f64> {
        self.data.clone()
    }

    #[allow(non_snake_case)]
    pub fn Rx(angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        let data = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, angle.cos(), angle.sin(),
            0.0, -angle.sin(), angle.cos()
        );
        Self::new(data)
    }

    #[allow(non_snake_case)]
    pub fn Ry(angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        let data = Matrix3::new(
            angle.cos(), 0.0, -angle.sin(),
            0.0, 1.0, 0.0,
            angle.sin(), 0.0, angle.cos()
        );
        Self::new(data)
    }

    #[allow(non_snake_case)]
    pub fn Rz(angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        let data = Matrix3::new(
            angle.cos(), angle.sin(), 0.0,
            -angle.sin(), angle.cos(), 0.0,
            0.0, 0.0, 1.0
        );
        Self::new(data)
    }
}

impl fmt::Display for RotationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(f, "RotationMatrix: \n{}", self.data)
    }
}

impl fmt::Debug for RotationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RotationMatrix: \n{:?}", self.data)
    }
}

impl PartialEq for RotationMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }

    fn ne(&self, other: &Self) -> bool {
        self.data != other.data
    }
}


impl std::ops::Mul<RotationMatrix> for RotationMatrix {
    type Output = RotationMatrix;

    fn mul(self, rhs: RotationMatrix) -> RotationMatrix {
        RotationMatrix::from_matrix(self.data * rhs.data)
    }
}

impl std::ops::MulAssign<RotationMatrix> for RotationMatrix {
    fn mul_assign(&mut self, rhs: RotationMatrix) {
        self.data *= rhs.data;
    }
}

impl std::ops::Mul<Vector3<f64>> for RotationMatrix {
    type Output = Vector3<f64>;

    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.data * rhs
    }
}

impl std::ops::Index<(usize, usize)> for RotationMatrix {
    type Output = f64;

    fn index(&self, idx: (usize, usize)) -> &f64 {
        &self.data[(idx.0, idx.1)]
    }
}

impl FromAttitude for RotationMatrix {
    fn from_quaternion(q: Quaternion) -> Self {
        todo!()
    }

    fn from_euler_axis(e: EulerAxis) -> Self {
        todo!()
    }

    fn from_euler_angle(e: EulerAngle) -> Self {
        todo!()
    }

    fn from_rotation_matrix(r: RotationMatrix) -> Self {
        todo!()
    }
}

impl ToAttitude for RotationMatrix {
    fn to_quaternion(&self) -> Quaternion {
        todo!()
    }

    fn to_euler_axis(&self) -> EulerAxis {
        todo!()
    }

    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        todo!()
    }

    fn to_rotation_matrix(&self) -> RotationMatrix {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Matrix3;

}