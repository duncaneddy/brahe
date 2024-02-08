/*!
The `euler_axis` module provides the `EulerAxis` struct, which represents an attitude transformation in the form of a
single rotation about an arbitrary axis.
*/

use nalgebra::{Vector3, Vector4};
use std::{fmt, ops};

use crate::attitude::attitude_types::{
    EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix,
};
use crate::attitude::ToAttitude;
use crate::constants::{DEG2RAD, RAD2DEG};
use crate::FromAttitude;

impl EulerAxis {
    /// Create a new `EulerAxis` struct from an axis and angle. The angle can be in either degrees or radians, and
    /// the `as_degrees` flag is used to specify the input angle units.
    ///
    /// # Arguments
    ///
    /// - `axis` - A `Vector3<f64>` representing the axis of rotation.
    /// - `angle` - A `f64` representing the angle of rotation.
    /// - `as_degrees` - A `bool` flag indicating if the input angle is in degrees.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let axis = Vector3::new(1.0, 1.0, 1.0);
    /// let angle = 45.0;
    ///
    /// let e = EulerAxis::new(axis, angle, true);
    /// ```
    pub fn new(axis: Vector3<f64>, angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        Self { axis, angle }
    }

    /// Create a new `EulerAxis` struct from individual axis and angle values. The angle can be in either degrees or radians, and
    /// the `as_degrees` flag is used to specify the input angle units.
    ///
    /// # Arguments
    ///
    /// - `x` - A `f64` representing the x-component of the axis of rotation.
    /// - `y` - A `f64` representing the y-component of the axis of rotation.
    /// - `z` - A `f64` representing the z-component of the axis of rotation.
    /// - `angle` - A `f64` representing the angle of rotation.
    /// - `as_degrees` - A `bool` flag indicating if the input angle is in degrees. Set to `true` if the input angle is in degrees.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// ```
    pub fn from_values(x: f64, y: f64, z: f64, angle: f64, as_degrees: bool) -> Self {
        Self::new(Vector3::new(x, y, z), angle, as_degrees)
    }

    /// Create a new `EulerAxis` struct from a `Vector4<f64>`. The `as_degrees` flag is used to specify the input angle units.
    /// The angle can be either the first or last component of the vector, and the `vector_first` flag is used to specify
    /// the location of the angle in the vector.
    ///
    /// # Arguments
    ///
    /// - `vector` - A `Vector4<f64>` representing the axis and angle of rotation.
    /// - `as_degrees` - A `bool` flag indicating if the input angle is in degrees. Set to `true` if the input angle is in degrees.
    /// - `vector_first` - A `bool` flag indicating if the angle is the first component of the vector. Set to `true` if the angle is the first component.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector4;
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let vector = Vector4::new(1.0, 1.0, 1.0, 45.0);
    /// let e = EulerAxis::from_vector(vector, true, false);
    /// ```
    pub fn from_vector(vector: Vector4<f64>, as_degrees: bool, vector_first: bool) -> Self {
        let (angle, axis) = if vector_first {
            (vector[3], Vector3::new(vector[0], vector[1], vector[2]))
        } else {
            (vector[0], Vector3::new(vector[1], vector[2], vector[3]))
        };

        Self::new(axis, angle, as_degrees)
    }

    /// Convert the `EulerAxis` struct to a `Vector4<f64>`. The `as_degrees` flag is used to specify the output angle units.
    /// The angle can be either the first or last component of the vector, and the `vector_first` flag is used to specify
    /// the location of the angle in the vector.
    ///
    /// # Arguments
    ///
    /// - `as_degrees` - A `bool` flag indicating if the output angle is in degrees. Set to `true` if the output angle is in degrees.
    /// - `vector_first` - A `bool` flag indicating if the angle is the first component of the vector. Set to `true` if the angle is the first component.
    ///
    /// # Returns
    ///
    /// - A `Vector4<f64>` representing the axis and angle of rotation.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector4;
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// // Create angle-first vector
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// let vector = e.to_vector(true, false);
    ///
    /// assert_eq!(vector, Vector4::new(45.0, 1.0, 1.0, 1.0));
    ///
    /// // Create angle-last vector
    ///
    /// let vector = e.to_vector(true, true);
    ///
    /// assert_eq!(vector, Vector4::new(1.0, 1.0, 1.0, 45.0));
    /// ```
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
        write!(
            f,
            "EulerAxis: [axis: ({}, {}, {}), angle: {}]",
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.angle * RAD2DEG
        )
    }
}

impl fmt::Debug for EulerAxis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "EulerAxis<Axis<{},{},{}>,Angle<{}>>",
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.angle * RAD2DEG
        )
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

impl From<Quaternion> for EulerAxis {
    fn from(q: Quaternion) -> Self {
        q.to_euler_axis()
    }
}

impl From<RotationMatrix> for EulerAxis {
    fn from(r: RotationMatrix) -> Self {
        r.to_euler_axis()
    }
}

impl From<EulerAngle> for EulerAxis {
    fn from(e: EulerAngle) -> Self {
        e.to_euler_axis()
    }
}

impl FromAttitude for EulerAxis {
    /// Create a new `EulerAxis` struct from a `Quaternion`.
    ///
    /// # Arguments
    ///
    /// - `q` - A `Quaternion` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{Quaternion, EulerAxis};
    ///
    /// let q = Quaternion::from_values(0.0, 0.0, 0.0, 1.0);
    /// let e = EulerAxis::from_quaternion(q);
    /// ```
    fn from_quaternion(q: Quaternion) -> Self {
        q.to_euler_axis()
    }

    /// Create a new `EulerAxis` struct from an `EulerAxis`. This is equivalent to cloning the input `EulerAxis`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAxis` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// let e2 = EulerAxis::from_euler_axis(e);
    /// ```
    fn from_euler_axis(e: EulerAxis) -> Self {
        EulerAxis::new(e.axis.clone(), e.angle, false)
    }

    /// Create a new `EulerAxis` struct from an `EulerAngle`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAngle` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAxis};
    ///
    /// let e = EulerAngle::from_values(45.0, 45.0, 45.0, true, true, true);
    /// let e2 = EulerAxis::from_euler_angle(e);
    /// ```
    fn from_euler_angle(e: EulerAngle) -> Self {
        e.to_euler_axis()
    }

    /// Create a new `EulerAxis` struct from a `RotationMatrix`.
    ///
    /// # Arguments
    ///
    /// - `r` - A `RotationMatrix` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{RotationMatrix, EulerAxis};
    ///
    /// let r = RotationMatrix::from_values(
    ///    1.0, 0.0, 0.0,
    ///    0.0, 0.70710678, -0.70710678,
    ///    0.0, 0.70710678, 0.70710678
    /// );
    fn from_rotation_matrix(r: RotationMatrix) -> Self {
        // Convert to quaternion and then to euler axis
        r.to_quaternion().to_euler_axis()
    }
}

impl ToAttitude for EulerAxis {
    /// Convert the `EulerAxis` struct to a `Quaternion`.
    ///
    /// # Returns
    ///
    /// - A `Quaternion` representing the attitude transformation.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// let q = e.to_quaternion();
    /// ```
    fn to_quaternion(&self) -> Quaternion {
        Quaternion::from_euler_axis(*self)
    }

    /// Convert the `EulerAxis` struct to an `EulerAxis`. This is equivalent to cloning the input `EulerAxis`.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// let e2 = e.to_euler_axis();
    /// ```
    fn to_euler_axis(&self) -> EulerAxis {
        self.clone()
    }

    /// Convert the `EulerAxis` struct to an `EulerAngle`. The `order` field is used to specify the order of the Euler angles.
    /// The `order` field is specified by the `EulerAngleOrder` enum.
    ///
    /// # Arguments
    ///
    /// - `order` - An `EulerAngleOrder` enum specifying the order of the Euler angles.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAxis, EulerAngleOrder};
    ///
    /// let euler_axis = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// let euler_angle = e.to_euler_angle(EulerAngleOrder::XYZ);
    /// ```
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        EulerAngle::from_euler_axis(*self, order)
    }

    /// Convert the `EulerAxis` struct to a `RotationMatrix`.
    ///
    /// # Returns
    ///
    /// - A new `RotationMatrix` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, true);
    /// let r = e.to_rotation_matrix();
    /// ```
    fn to_rotation_matrix(&self) -> RotationMatrix {
        self.to_quaternion().to_rotation_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        todo!()
    }

    #[test]
    fn from_values() {
        todo!()
    }

    #[test]
    fn from_vector() {
        todo!()
    }

    #[test]
    fn to_vector() {
        todo!()
    }

    #[test]
    fn from_quaternion() {
        todo!()
    }

    #[test]
    fn from_euler_axis() {
        todo!()
    }

    #[test]
    fn from_euler_angle() {
        todo!()
    }

    #[test]
    fn from_rotation_matrix() {
        todo!()
    }

    #[test]
    fn to_quaternion() {
        todo!()
    }

    #[test]
    fn to_euler_axis() {
        todo!()
    }

    #[test]
    fn to_euler_angle() {
        todo!()
    }

    #[test]
    fn to_rotation_matrix() {
        todo!()
    }
}
