/*!
The `euler_axis` module provides the `EulerAxis` struct, which represents an attitude transformation in the form of a
single rotation about an arbitrary axis.
*/

use std::{ops, fmt};
use nalgebra::{Vector3, Vector4};

use crate::constants::{DEG2RAD, RAD2DEG};
use crate::attitude::ToAttitude;
use crate::attitude::attitude_types::{Quaternion, EulerAngle, RotationMatrix, EulerAngleOrder, EulerAxis};

impl EulerAxis {
    pub fn new(axis: Vector3<f64>, angle: f64) -> Self {
        Self { axis, angle }
    }

    pub fn from_vector(vector: Vector4<f64>, as_degrees: bool, vector_first: bool) -> Self {
        let (angle, axis) = if vector_first {
            (vector[3], Vector3::new(vector[0], vector[1], vector[2]))
        } else {
            (vector[0], Vector3::new(vector[1], vector[2], vector[3]))
        };

        if as_degrees {
            Self::new(axis, angle * DEG2RAD)
        } else {
            Self::new(axis, angle)
        }
    }

    pub fn to_vector(&self, as_degrees: bool, vector_first: bool) -> Vector4<f64> {
        if as_degrees {
            if vector_first {
                Vector4::new(self.angle * RAD2DEG, self.axis.x, self.axis.y, self.axis.z)
            } else {
                Vector4::new(self.axis.x, self.axis.y, self.axis.z, self.angle * RAD2DEG)
            }
        } else {
            if vector_first {
                Vector4::new(self.angle, self.axis.x, self.axis.y, self.axis.z)
            } else {
                Vector4::new(self.axis.x, self.axis.y, self.axis.z, self.angle)
            }
        }
    }
}

impl fmt::Display for EulerAxis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(f, "EulerAxis: [axis: ({}, {}, {}), angle: {}]", self.axis.x, self.axis.y, self.axis.z, self.angle * RAD2DEG)
    }
}

impl fmt::Debug for EulerAxis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EulerAxis<Axis<{},{},{}>,Angle<{}>>", self.axis.x, self.axis.y, self.axis.z, self.angle * RAD2DEG)
    }
}

impl ops::Index<usize> for EulerAxis {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        &self.axis[idx]
    }
}

impl PartialEq for EulerAxis {
    fn eq(&self, other: &Self) -> bool {
        self.axis == other.axis && self.angle == other.angle
    }

    fn ne(&self, other: &Self) -> bool {
        self.axis != other.axis || self.angle != other.angle
    }
}

impl ToAttitude for EulerAxis {
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
}