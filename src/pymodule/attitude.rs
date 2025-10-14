/// Helper function to parse strings into appropriate EOPExtrapolation enumerations
fn string_to_euler_angle_order(s: &str) -> Result<attitude::EulerAngleOrder, BraheError> {
    match s {
        "XYX" => Ok(attitude::EulerAngleOrder::XYX),
        "XYZ" => Ok(attitude::EulerAngleOrder::XYZ),
        "XZX" => Ok(attitude::EulerAngleOrder::XZX),
        "XZY" => Ok(attitude::EulerAngleOrder::XZY),
        "YXY" => Ok(attitude::EulerAngleOrder::YXY),
        "YXZ" => Ok(attitude::EulerAngleOrder::YXZ),
        "YZX" => Ok(attitude::EulerAngleOrder::YZX),
        "YZY" => Ok(attitude::EulerAngleOrder::YZY),
        "ZXY" => Ok(attitude::EulerAngleOrder::ZXY),
        "ZXZ" => Ok(attitude::EulerAngleOrder::ZXZ),
        "ZYX" => Ok(attitude::EulerAngleOrder::ZYX),
        "ZYZ" => Ok(attitude::EulerAngleOrder::ZYZ),
        _ => Err(BraheError::Error(format!(
            "Unknown EulerAngleOrder \"{}\"",
            s
        ))),
    }
}

/// Helper function to convert EOPExtrapolation enumerations into representative string
fn euler_angle_order_to_string(order: attitude::EulerAngleOrder) -> String {
    match order {
        attitude::EulerAngleOrder::XYX => String::from("XYX"),
        attitude::EulerAngleOrder::XYZ => String::from("XYZ"),
        attitude::EulerAngleOrder::XZX => String::from("XZX"),
        attitude::EulerAngleOrder::XZY => String::from("XZY"),
        attitude::EulerAngleOrder::YXY => String::from("YXY"),
        attitude::EulerAngleOrder::YXZ => String::from("YXZ"),
        attitude::EulerAngleOrder::YZX => String::from("YZX"),
        attitude::EulerAngleOrder::YZY => String::from("YZY"),
        attitude::EulerAngleOrder::ZXY => String::from("ZXY"),
        attitude::EulerAngleOrder::ZXZ => String::from("ZXZ"),
        attitude::EulerAngleOrder::ZYX => String::from("ZYX"),
        attitude::EulerAngleOrder::ZYZ => String::from("ZYZ"),
    }
}

#[pyclass]
#[pyo3(name = "Quaternion")]
#[derive(Clone)]
/// Represents a quaternion for 3D rotations.
///
/// Quaternions provide a compact, singularity-free representation of rotations.
/// The quaternion is stored as [w, x, y, z] where w is the scalar part and
/// [x, y, z] is the vector part.
///
/// Args:
///     w (float): Scalar component
///     x (float): X component of vector part
///     y (float): Y component of vector part
///     z (float): Z component of vector part
///
/// Example:
///     >>> q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
///     >>> q.normalize()
///     >>> print(q.norm())
struct PyQuaternion {
    obj: attitude::Quaternion,
}

#[pymethods]
impl PyQuaternion {
    #[new]
    #[pyo3(text_signature = "(w, x, y, z)")]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> PyQuaternion {
        PyQuaternion {
            obj: attitude::Quaternion::new(w, x, y, z),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(v, scalar_first)")]
    /// Create a quaternion from a numpy array.
    ///
    /// Args:
    ///     v (numpy.ndarray): 4-element array containing quaternion components
    ///     scalar_first (bool): If True, array is [w, x, y, z], else [x, y, z, w]
    ///
    /// Returns:
    ///     Quaternion: New quaternion instance
    pub fn from_vector(
        _cls: &Bound<'_, PyType>,
        v: &Bound<'_, PyArray<f64, Ix1>>,
        scalar_first: bool,
    ) -> PyQuaternion {
        PyQuaternion {
            obj: attitude::Quaternion::from_vector(numpy_to_vector!(v, 4, f64), scalar_first),
        }
    }

    #[pyo3(text_signature = "($self, scalar_first)")]
    /// Convert quaternion to a numpy array.
    ///
    /// Args:
    ///     scalar_first (bool): If True, returns [w, x, y, z], else [x, y, z, w]
    ///
    /// Returns:
    ///     numpy.ndarray: 4-element array containing quaternion components
    pub unsafe fn to_vector<'py>(&self, py: Python<'py>, scalar_first: bool) -> Bound<'py, PyArray<f64, Ix1>> {
        vector_to_numpy!(py, self.obj.to_vector(scalar_first), 4, f64)
    }

    #[pyo3(text_signature = "($self)")]
    /// Normalize the quaternion in-place to unit length.
    pub fn normalize(&mut self) {
        self.obj.normalize();
    }

    #[pyo3(text_signature = "($self)")]
    /// Calculate the norm (magnitude) of the quaternion.
    ///
    /// Returns:
    ///     float: Euclidean norm of the quaternion
    pub fn norm(&self) -> f64 {
        self.obj.norm()
    }

    #[pyo3(text_signature = "($self)")]
    /// Compute the conjugate of the quaternion.
    ///
    /// Returns:
    ///     Quaternion: Conjugate quaternion with negated vector part
    pub fn conjugate(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.conjugate()
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Compute the inverse of the quaternion.
    ///
    /// Returns:
    ///     Quaternion: Inverse quaternion
    pub fn inverse(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.inverse()
        }
    }

    #[pyo3(text_signature = "($self, other, t)")]
    /// Perform spherical linear interpolation (SLERP) between two quaternions.
    ///
    /// Args:
    ///     other (Quaternion): Target quaternion
    ///     t (float): Interpolation parameter in [0, 1]
    ///
    /// Returns:
    ///     Quaternion: Interpolated quaternion
    pub fn slerp(&self, other: PyQuaternion, t: f64) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.slerp(other.obj, t)
        }
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    pub fn __add__(&self, other: PyQuaternion) -> PyResult<PyQuaternion> {
        Ok(PyQuaternion {
            obj: self.obj + other.obj,
        })
    }
    pub fn __iadd__(&mut self, other: PyQuaternion) {
        self.obj += other.obj;
    }

    pub fn __sub__(&self, other: &PyQuaternion) -> PyResult<PyQuaternion> {
        Ok(PyQuaternion {
            obj: self.obj - other.obj
        })
    }

    pub fn __mul__(&self, other: &PyQuaternion) -> PyResult<PyQuaternion> {
        Ok(PyQuaternion {
            obj: self.obj * other.obj
        })
    }

    pub fn __imul__(&mut self, other: PyQuaternion) {
        self.obj *= other.obj;
    }

    pub fn __isub__(&mut self, other: PyQuaternion) {
        self.obj -= other.obj;
    }

    pub fn __eq__(&self, other: &PyQuaternion) -> bool {
        self.obj == other.obj
    }

    pub fn __ne__(&self, other: &PyQuaternion) -> bool {
        self.obj != other.obj
    }

    pub fn __getitem__(&self, index: usize) -> f64 {
        self.obj[index]
    }

    #[getter]
    /// Get the quaternion components as a numpy array [w, x, y, z].
    ///
    /// Returns:
    ///     numpy.ndarray: 4-element array containing quaternion components
    pub unsafe fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        vector_to_numpy!(py, self.obj.data, 4, f64)
    }

    #[classmethod]
    #[pyo3(text_signature = "(q)")]
    /// Create a quaternion from another quaternion (copy constructor).
    ///
    /// Args:
    ///     q (Quaternion): Source quaternion
    ///
    /// Returns:
    ///     Quaternion: New quaternion instance
    pub fn from_quaternion(_cls: &Bound<'_, PyType>, q: &PyQuaternion) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_quaternion(q.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e)")]
    /// Create a quaternion from an Euler axis representation.
    ///
    /// Args:
    ///     e (EulerAxis): Euler axis representation
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    pub fn from_euler_axis(_cls: &Bound<'_, PyType>, e: &PyEulerAxis) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_euler_axis(e.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e)")]
    /// Create a quaternion from an Euler angle representation.
    ///
    /// Args:
    ///     e (EulerAngle): Euler angle representation
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    pub fn from_euler_angle(_cls: &Bound<'_, PyType>, e: &PyEulerAngle) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_euler_angle(e.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(r)")]
    /// Create a quaternion from a rotation matrix.
    ///
    /// Args:
    ///     r (RotationMatrix): Rotation matrix
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    pub fn from_rotation_matrix(_cls: &Bound<'_, PyType>, r: &PyRotationMatrix) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_rotation_matrix(r.obj)
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to quaternion representation (returns self).
    ///
    /// Returns:
    ///     Quaternion: This quaternion
    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to Euler axis representation.
    ///
    /// Returns:
    ///     EulerAxis: Equivalent Euler axis
    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    #[pyo3(text_signature = "($self, order)")]
    /// Convert to Euler angle representation.
    ///
    /// Args:
    ///     order (str): Rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles
    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation.
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass]
#[pyo3(name = "EulerAngle")]
#[derive(Clone)]
/// Represents a rotation using Euler angles.
///
/// Euler angles describe rotations as a sequence of three rotations about
/// specified axes. The rotation sequence is specified by the order parameter
/// (e.g., "XYZ", "ZYX").
///
/// Args:
///     order (str): Rotation sequence (e.g., "XYZ", "ZYX", "ZXZ")
///     phi (float): First rotation angle in radians or degrees
///     theta (float): Second rotation angle in radians or degrees
///     psi (float): Third rotation angle in radians or degrees
///     angle_format (AngleFormat): Units of input angles (RADIANS or DEGREES)
///
/// Example:
///     >>> from brahe import EulerAngle, AngleFormat
///     >>> e = EulerAngle("XYZ", 0.1, 0.2, 0.3, AngleFormat.RADIANS)
///     >>> print(e.phi, e.theta, e.psi)
struct PyEulerAngle {
    obj: attitude::EulerAngle,
}

#[pymethods]
impl PyEulerAngle {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    pub fn __eq__(&self, other: &PyEulerAngle) -> bool {
        self.obj == other.obj
    }

    pub fn __ne__(&self, other: &PyEulerAngle) -> bool {
        self.obj != other.obj
    }

    #[getter]
    /// Get the first rotation angle (phi) in radians.
    ///
    /// Returns:
    ///     float: First rotation angle in radians
    pub fn phi(&self) -> f64 {
        self.obj.phi
    }

    #[getter]
    /// Get the second rotation angle (theta) in radians.
    ///
    /// Returns:
    ///     float: Second rotation angle in radians
    pub fn theta(&self) -> f64 {
        self.obj.theta
    }

    #[getter]
    /// Get the third rotation angle (psi) in radians.
    ///
    /// Returns:
    ///     float: Third rotation angle in radians
    pub fn psi(&self) -> f64 {
        self.obj.psi
    }

    #[getter]
    /// Get the rotation sequence order.
    ///
    /// Returns:
    ///     str: Rotation sequence (e.g., "XYZ", "ZYX")
    pub fn order(&self) -> String {
        euler_angle_order_to_string(self.obj.order)
    }

    #[new]
    #[pyo3(text_signature = "(order, phi, theta, psi, angle_format)")]
    pub fn new(order: &str, phi: f64, theta: f64, psi: f64, angle_format: &PyAngleFormat) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::new(string_to_euler_angle_order(order).unwrap(), phi, theta, psi, angle_format.value),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(v, order, angle_format)")]
    /// Create Euler angles from a numpy array.
    ///
    /// Args:
    ///     v (numpy.ndarray): 3-element array [phi, theta, psi]
    ///     order (str): Rotation sequence (e.g., "XYZ", "ZYX")
    ///     angle_format (AngleFormat): Units of input angles (RADIANS or DEGREES)
    ///
    /// Returns:
    ///     EulerAngle: New Euler angle instance
    pub fn from_vector(_cls: &Bound<'_, PyType>, v: &Bound<'_, PyArray<f64, Ix1>>, order: &str, angle_format: &PyAngleFormat) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_vector(numpy_to_vector!(v, 3, f64), string_to_euler_angle_order(order).unwrap(), angle_format.value),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(q, order)")]
    /// Create Euler angles from a quaternion.
    ///
    /// Args:
    ///     q (Quaternion): Source quaternion
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles
    pub fn from_quaternion(_cls: &Bound<'_, PyType>, q: &PyQuaternion, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_quaternion(q.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e, order)")]
    /// Create Euler angles from an Euler axis representation.
    ///
    /// Args:
    ///     e (EulerAxis): Euler axis representation
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles
    pub fn from_euler_axis(_cls: &Bound<'_, PyType>, e: &PyEulerAxis, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_euler_axis(e.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e, order)")]
    /// Create Euler angles from another Euler angle with different order.
    ///
    /// Args:
    ///     e (EulerAngle): Source Euler angles
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles with new order
    pub fn from_euler_angle(_cls: &Bound<'_, PyType>, e: &PyEulerAngle, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_euler_angle(e.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(r, order)")]
    /// Create Euler angles from a rotation matrix.
    ///
    /// Args:
    ///     r (RotationMatrix): Rotation matrix
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles
    pub fn from_rotation_matrix(_cls: &Bound<'_, PyType>, r: &PyRotationMatrix, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_rotation_matrix(r.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to quaternion representation.
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to Euler axis representation.
    ///
    /// Returns:
    ///     EulerAxis: Equivalent Euler axis
    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    #[pyo3(text_signature = "($self, order)")]
    /// Convert to Euler angles with different rotation sequence.
    ///
    /// Args:
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles with new order
    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation.
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass]
#[pyo3(name = "EulerAxis")]
#[derive(Clone)]
/// Represents a rotation using Euler axis-angle representation.
///
/// The Euler axis-angle representation describes a rotation as a single rotation
/// about a specified axis by a given angle. This is also known as the axis-angle
/// or rotation vector representation.
///
/// Args:
///     axis (numpy.ndarray): 3-element unit vector specifying rotation axis
///     angle (float): Rotation angle in radians or degrees
///     angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)
///
/// Example:
///     >>> import numpy as np
///     >>> from brahe import EulerAxis, AngleFormat
///     >>> axis = np.array([0.0, 0.0, 1.0])
///     >>> e = EulerAxis(axis, np.pi/2, AngleFormat.RADIANS)
///     >>> print(e.angle)
struct PyEulerAxis {
    obj: attitude::EulerAxis,
}

#[pymethods]
impl PyEulerAxis {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    pub fn __eq__(&self, other: &PyEulerAxis) -> bool {
        self.obj == other.obj
    }

    pub fn __ne__(&self, other: &PyEulerAxis) -> bool {
        self.obj != other.obj
    }

    pub fn __getitem__(&self, index: usize) -> f64 {
        self.obj[index]
    }

    #[getter]
    /// Get the rotation angle in radians.
    ///
    /// Returns:
    ///     float: Rotation angle in radians
    pub fn angle(&self) -> f64 {
        self.obj.angle
    }

    #[getter]
    /// Get the rotation axis as a numpy array.
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element unit vector specifying rotation axis
    pub unsafe fn axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        vector_to_numpy!(py, self.obj.axis, 3, f64)
    }

    #[new]
    #[pyo3(text_signature = "(axis, angle, angle_format)")]
    pub fn new(axis: &Bound<'_, PyArray<f64, Ix1>>, angle: f64, angle_format: &PyAngleFormat) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::new(numpy_to_vector!(axis, 3, f64), angle, angle_format.value),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(x, y, z, angle, angle_format)")]
    /// Create an Euler axis from individual axis components and angle.
    ///
    /// Args:
    ///     x (float): X component of rotation axis
    ///     y (float): Y component of rotation axis
    ///     z (float): Z component of rotation axis
    ///     angle (float): Rotation angle in radians or degrees
    ///     angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)
    ///
    /// Returns:
    ///     EulerAxis: New Euler axis instance
    pub fn from_values(_cls: &Bound<'_, PyType>, x: f64, y: f64, z: f64, angle: f64, angle_format: &PyAngleFormat) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_values(x, y, z, angle, angle_format.value),
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(v, angle_format, vector_first)")]
    /// Create an Euler axis from a numpy array.
    ///
    /// Args:
    ///     v (numpy.ndarray): 4-element array containing axis and angle
    ///     angle_format (AngleFormat): Units of angle (RADIANS or DEGREES)
    ///     vector_first (bool): If True, array is [x, y, z, angle], else [angle, x, y, z]
    ///
    /// Returns:
    ///     EulerAxis: New Euler axis instance
    pub fn from_vector(_cls: &Bound<'_, PyType>, v: &Bound<'_, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat, vector_first: bool) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_vector(numpy_to_vector!(v, 4, f64), angle_format.value, vector_first),
        }
    }

    #[pyo3(text_signature = "($self, angle_format, vector_first)")]
    /// Convert Euler axis to a numpy array.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Units for output angle (RADIANS or DEGREES)
    ///     vector_first (bool): If True, returns [x, y, z, angle], else [angle, x, y, z]
    ///
    /// Returns:
    ///     numpy.ndarray: 4-element array containing axis and angle
    pub unsafe fn to_vector<'py>(&self, py: Python<'py>, angle_format: &PyAngleFormat, vector_first: bool) -> Bound<'py, PyArray<f64, Ix1>> {
        vector_to_numpy!(py, self.obj.to_vector(angle_format.value, vector_first), 4, f64)
    }

    #[classmethod]
    #[pyo3(text_signature = "(q)")]
    /// Create an Euler axis from a quaternion.
    ///
    /// Args:
    ///     q (Quaternion): Source quaternion
    ///
    /// Returns:
    ///     EulerAxis: Equivalent Euler axis
    pub fn from_quaternion(_cls: &Bound<'_, PyType>, q: &PyQuaternion) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_quaternion(q.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e)")]
    /// Create an Euler axis from another Euler axis (copy constructor).
    ///
    /// Args:
    ///     e (EulerAxis): Source Euler axis
    ///
    /// Returns:
    ///     EulerAxis: New Euler axis instance
    pub fn from_euler_axis(_cls: &Bound<'_, PyType>, e: &PyEulerAxis) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_euler_axis(e.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e)")]
    /// Create an Euler axis from Euler angles.
    ///
    /// Args:
    ///     e (EulerAngle): Euler angle representation
    ///
    /// Returns:
    ///     EulerAxis: Equivalent Euler axis
    pub fn from_euler_angle(_cls: &Bound<'_, PyType>, e: &PyEulerAngle) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_euler_angle(e.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(r)")]
    /// Create an Euler axis from a rotation matrix.
    ///
    /// Args:
    ///     r (RotationMatrix): Rotation matrix
    ///
    /// Returns:
    ///     EulerAxis: Equivalent Euler axis
    pub fn from_rotation_matrix(_cls: &Bound<'_, PyType>, r: &PyRotationMatrix) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_rotation_matrix(r.obj)
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to quaternion representation.
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to Euler axis representation (returns self).
    ///
    /// Returns:
    ///     EulerAxis: This Euler axis
    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    #[pyo3(text_signature = "($self, order)")]
    /// Convert to Euler angle representation.
    ///
    /// Args:
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles
    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation.
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass]
#[pyo3(name = "RotationMatrix")]
#[derive(Clone)]
/// Represents a rotation using a 3x3 rotation matrix (Direction Cosine Matrix).
///
/// A rotation matrix is an orthogonal 3x3 matrix with determinant +1 that
/// represents rotation in 3D space. Also known as a Direction Cosine Matrix (DCM).
///
/// Args:
///     r11 (float): Element at row 1, column 1
///     r12 (float): Element at row 1, column 2
///     r13 (float): Element at row 1, column 3
///     r21 (float): Element at row 2, column 1
///     r22 (float): Element at row 2, column 2
///     r23 (float): Element at row 2, column 3
///     r31 (float): Element at row 3, column 1
///     r32 (float): Element at row 3, column 2
///     r33 (float): Element at row 3, column 3
///
/// Raises:
///     BraheError: If the matrix is not a valid rotation matrix
///
/// Example:
///     >>> from brahe import RotationMatrix
///     >>> r = RotationMatrix(1, 0, 0, 0, 1, 0, 0, 0, 1)  # Identity
///     >>> print(r)
struct PyRotationMatrix {
    obj: attitude::RotationMatrix,
}

#[pymethods]
impl PyRotationMatrix {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    pub fn __eq__(&self, other: &PyRotationMatrix) -> bool {
        self.obj == other.obj
    }

    pub fn __ne__(&self, other: &PyRotationMatrix) -> bool {
        self.obj != other.obj
    }

    pub fn __getitem__(&self, index: (usize, usize)) -> f64 {
        self.obj[index]
    }

    pub fn __mul__(&self, other: &PyRotationMatrix) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj * other.obj
        }
    }

    pub fn __imul__(&mut self, other: PyRotationMatrix) {
        self.obj *= other.obj;
    }

    #[new]
    #[pyo3(text_signature = "(r11, r12, r13, r21, r22, r23, r31, r32, r33)")]
    #[allow(clippy::too_many_arguments)]
    pub fn new(r11: f64, r12: f64, r13: f64, r21: f64, r22: f64, r23: f64, r31: f64, r32: f64, r33: f64) -> Result<PyRotationMatrix, BraheError> {
        Ok(PyRotationMatrix {
            obj: attitude::RotationMatrix::new(r11, r12, r13, r21, r22, r23, r31, r32, r33)?,
        })
    }

    #[classmethod]
    #[pyo3(text_signature = "(m)")]
    /// Create a rotation matrix from a 3x3 numpy array.
    ///
    /// Args:
    ///     m (numpy.ndarray): 3x3 rotation matrix
    ///
    /// Returns:
    ///     RotationMatrix: New rotation matrix instance
    ///
    /// Raises:
    ///     BraheError: If the matrix is not a valid rotation matrix
    pub fn from_matrix(_cls: &Bound<'_, PyType>, m: &Bound<'_, PyArray<f64, Ix2>>) -> Result<PyRotationMatrix, BraheError> {
        Ok(PyRotationMatrix {
            obj: attitude::RotationMatrix::from_matrix(numpy_to_matrix!(m, 3, 3, f64))?,
        })
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert rotation matrix to a 3x3 numpy array.
    ///
    /// Returns:
    ///     numpy.ndarray: 3x3 rotation matrix
    pub unsafe fn to_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        matrix_to_numpy!(py, self.obj.to_matrix(), 3, 3, f64)
    }

    #[classmethod]
    #[allow(non_snake_case)]
    #[pyo3(text_signature = "(angle, angle_format)")]
    /// Create a rotation matrix for rotation about the X axis.
    ///
    /// Args:
    ///     angle (float): Rotation angle in radians or degrees
    ///     angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)
    ///
    /// Returns:
    ///     RotationMatrix: Rotation matrix for X-axis rotation
    pub fn Rx(_cls: &Bound<'_, PyType>, angle: f64, angle_format: &PyAngleFormat) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::Rx(angle, angle_format.value)
        }
    }

    #[classmethod]
    #[allow(non_snake_case)]
    #[pyo3(text_signature = "(angle, angle_format)")]
    /// Create a rotation matrix for rotation about the Y axis.
    ///
    /// Args:
    ///     angle (float): Rotation angle in radians or degrees
    ///     angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)
    ///
    /// Returns:
    ///     RotationMatrix: Rotation matrix for Y-axis rotation
    pub fn Ry(_cls: &Bound<'_, PyType>, angle: f64, angle_format: &PyAngleFormat) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::Ry(angle, angle_format.value)
        }
    }

    #[classmethod]
    #[allow(non_snake_case)]
    #[pyo3(text_signature = "(angle, angle_format)")]
    /// Create a rotation matrix for rotation about the Z axis.
    ///
    /// Args:
    ///     angle (float): Rotation angle in radians or degrees
    ///     angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)
    ///
    /// Returns:
    ///     RotationMatrix: Rotation matrix for Z-axis rotation
    pub fn Rz(_cls: &Bound<'_, PyType>, angle: f64, angle_format: &PyAngleFormat) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::Rz(angle, angle_format.value)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(q)")]
    /// Create a rotation matrix from a quaternion.
    ///
    /// Args:
    ///     q (Quaternion): Source quaternion
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    pub fn from_quaternion(_cls: &Bound<'_, PyType>, q: &PyQuaternion) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_quaternion(q.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e)")]
    /// Create a rotation matrix from an Euler axis.
    ///
    /// Args:
    ///     e (EulerAxis): Euler axis representation
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    pub fn from_euler_axis(_cls: &Bound<'_, PyType>, e: &PyEulerAxis) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_euler_axis(e.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(e)")]
    /// Create a rotation matrix from Euler angles.
    ///
    /// Args:
    ///     e (EulerAngle): Euler angle representation
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    pub fn from_euler_angle(_cls: &Bound<'_, PyType>, e: &PyEulerAngle) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_euler_angle(e.obj)
        }
    }

    #[classmethod]
    #[pyo3(text_signature = "(r)")]
    /// Create a rotation matrix from another rotation matrix (copy constructor).
    ///
    /// Args:
    ///     r (RotationMatrix): Source rotation matrix
    ///
    /// Returns:
    ///     RotationMatrix: New rotation matrix instance
    pub fn from_rotation_matrix(_cls: &Bound<'_, PyType>, r: &PyRotationMatrix) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_rotation_matrix(r.obj)
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to quaternion representation.
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to Euler axis representation.
    ///
    /// Returns:
    ///     EulerAxis: Equivalent Euler axis
    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    #[pyo3(text_signature = "($self, order)")]
    /// Convert to Euler angle representation.
    ///
    /// Args:
    ///     order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")
    ///
    /// Returns:
    ///     EulerAngle: Equivalent Euler angles
    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation (returns self).
    ///
    /// Returns:
    ///     RotationMatrix: This rotation matrix
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}