
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

/// Python wrapper for EulerAngleOrder enum
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EulerAngleOrder")]
#[derive(Clone)]
pub struct PyEulerAngleOrder {
    pub(crate) value: attitude::EulerAngleOrder,
}

#[pymethods]
impl PyEulerAngleOrder {
    #[classattr]
    #[allow(non_snake_case)]
    fn XYX() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::XYX }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn XYZ() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::XYZ }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn XZX() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::XZX }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn XZY() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::XZY }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn YXY() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::YXY }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn YXZ() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::YXZ }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn YZX() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::YZX }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn YZY() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::YZY }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ZXY() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::ZXY }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ZXZ() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::ZXZ }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ZYX() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::ZYX }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ZYZ() -> Self {
        PyEulerAngleOrder { value: attitude::EulerAngleOrder::ZYZ }
    }

    fn __str__(&self) -> String {
        euler_angle_order_to_string(self.value)
    }

    fn __repr__(&self) -> String {
        format!("EulerAngleOrder.{}", euler_angle_order_to_string(self.value))
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

#[pyclass(module = "brahe._brahe")]
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
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create identity quaternion
///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
///     print(f"Norm: {q.norm()}")
///
///     # Create from array
///     q_vec = np.array([1.0, 0.0, 0.0, 0.0])
///     q2 = bh.Quaternion.from_vector(q_vec, scalar_first=True)
///
///     # Convert to rotation matrix
///     dcm = q.to_rotation_matrix()
///
///     # Quaternion multiplication
///     q3 = q * q2
///
///     # Normalize
///     q3.normalize()
///     ```
pub struct PyQuaternion {
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     v = np.array([1.0, 0.0, 0.0, 0.0])
    ///     q = bh.Quaternion.from_vector(v, scalar_first=True)
    ///     ```
    pub fn from_vector(
        _cls: &Bound<'_, PyType>,
        v: &Bound<'_, PyAny>,
        scalar_first: bool,
    ) -> PyResult<PyQuaternion> {
        let vec = pyany_to_f64_array1(v, Some(4))?;
        Ok(PyQuaternion {
            obj: attitude::Quaternion::from_vector(na::SVector::<f64, 4>::from_vec(vec), scalar_first),
        })
    }

    #[pyo3(text_signature = "($self, scalar_first)")]
    /// Convert quaternion to a numpy array.
    ///
    /// Args:
    ///     scalar_first (bool): If True, returns [w, x, y, z], else [x, y, z, w]
    ///
    /// Returns:
    ///     numpy.ndarray: 4-element array containing quaternion components
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     v = q.to_vector(scalar_first=True)
    ///     ```
    pub unsafe fn to_vector<'py>(&self, py: Python<'py>, scalar_first: bool) -> Bound<'py, PyArray<f64, Ix1>> {
        vector_to_numpy!(py, self.obj.to_vector(scalar_first), 4, f64)
    }

    #[pyo3(text_signature = "($self)")]
    /// Normalize the quaternion in-place to unit length.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(2.0, 0.0, 0.0, 0.0)
    ///     q.normalize()
    ///     ```
    pub fn normalize(&mut self) {
        self.obj.normalize();
    }

    #[pyo3(text_signature = "($self)")]
    /// Calculate the norm (magnitude) of the quaternion.
    ///
    /// Returns:
    ///     float: Euclidean norm of the quaternion
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     norm = q.norm()
    ///     ```
    pub fn norm(&self) -> f64 {
        self.obj.norm()
    }

    #[pyo3(text_signature = "($self)")]
    /// Compute the conjugate of the quaternion.
    ///
    /// Returns:
    ///     Quaternion: Conjugate quaternion with negated vector part
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     q_conj = q.conjugate()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     q_inv = q.inverse()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q1 = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     q2 = bh.Quaternion(0.707, 0.707, 0.0, 0.0)
    ///     q_mid = q1.slerp(q2, 0.5)
    ///     ```
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

    #[getter]
    /// Get the scalar (w) component of the quaternion.
    ///
    /// Returns:
    ///     float: Scalar component
    pub fn w(&self) -> f64 {
        self.obj[0]
    }

    #[getter]
    /// Get the x component of the quaternion's vector part.
    ///
    /// Returns:
    ///     float: X component
    pub fn x(&self) -> f64 {
        self.obj[1]
    }

    #[getter]
    /// Get the y component of the quaternion's vector part.
    ///
    /// Returns:
    ///     float: Y component
    pub fn y(&self) -> f64 {
        self.obj[2]
    }

    #[getter]
    /// Get the z component of the quaternion's vector part.
    ///
    /// Returns:
    ///     float: Z component
    pub fn z(&self) -> f64 {
        self.obj[3]
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q1 = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     q2 = bh.Quaternion.from_quaternion(q1)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     q = bh.Quaternion.from_euler_axis(ea)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     euler = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     q = bh.Quaternion.from_euler_angle(euler)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     mat = np.eye(3)
    ///     rm = bh.RotationMatrix.from_matrix(mat)
    ///     q = bh.Quaternion.from_rotation_matrix(rm)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     euler = q.to_euler_angle("XYZ")
    ///     ```
    pub fn to_euler_angle(&self, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(order.value)
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
///     ```python
///     import brahe as bh
///
///     # Create Euler angle rotation (roll, pitch, yaw in ZYX order)
///     e = bh.EulerAngle("ZYX", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
///     print(f"Roll={e.phi}, Pitch={e.theta}, Yaw={e.psi}")
///
///     # Convert to quaternion
///     q = e.to_quaternion()
///
///     # Convert to rotation matrix
///     dcm = e.to_rotation_matrix()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EulerAngle")]
#[derive(Clone)]
pub struct PyEulerAngle {
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     print(f"Phi: {e.phi}")
    ///     ```
    pub fn phi(&self) -> f64 {
        self.obj.phi
    }

    #[getter]
    /// Get the second rotation angle (theta) in radians.
    ///
    /// Returns:
    ///     float: Second rotation angle in radians
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     print(f"Theta: {e.theta}")
    ///     ```
    pub fn theta(&self) -> f64 {
        self.obj.theta
    }

    #[getter]
    /// Get the third rotation angle (psi) in radians.
    ///
    /// Returns:
    ///     float: Third rotation angle in radians
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     print(f"Psi: {e.psi}")
    ///     ```
    pub fn psi(&self) -> f64 {
        self.obj.psi
    }

    #[getter]
    /// Get the rotation sequence order.
    ///
    /// Returns:
    ///     EulerAngleOrder: Rotation sequence enum value
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle(bh.EulerAngleOrder.XYZ, 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     print(f"Order: {e.order}")
    ///     ```
    pub fn order(&self) -> PyEulerAngleOrder {
        PyEulerAngleOrder {
            value: self.obj.order
        }
    }

    #[new]
    #[pyo3(text_signature = "(order, phi, theta, psi, angle_format)")]
    pub fn new(order: &PyEulerAngleOrder, phi: f64, theta: f64, psi: f64, angle_format: &PyAngleFormat) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::new(order.value, phi, theta, psi, angle_format.value),
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     v = np.array([0.1, 0.2, 0.3])
    ///     euler = bh.EulerAngle.from_vector(v, "XYZ", bh.AngleFormat.RADIANS)
    ///     ```
    pub fn from_vector(_cls: &Bound<'_, PyType>, v: &Bound<'_, PyAny>, order: &PyEulerAngleOrder, angle_format: &PyAngleFormat) -> PyResult<PyEulerAngle> {
        let vec = pyany_to_f64_array1(v, Some(3))?;
        Ok(PyEulerAngle {
            obj: attitude::EulerAngle::from_vector(na::SVector::<f64, 3>::from_vec(vec), order.value, angle_format.value),
        })
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     e = bh.EulerAngle.from_quaternion(q, "XYZ")
    ///     ```
    pub fn from_quaternion(_cls: &Bound<'_, PyType>, q: &PyQuaternion, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_quaternion(q.obj, order.value),
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     e = bh.EulerAngle.from_euler_axis(ea, "XYZ")
    ///     ```
    pub fn from_euler_axis(_cls: &Bound<'_, PyType>, e: &PyEulerAxis, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_euler_axis(e.obj, order.value),
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e1 = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     e2 = bh.EulerAngle.from_euler_angle(e1, "ZYX")
    ///     ```
    pub fn from_euler_angle(_cls: &Bound<'_, PyType>, e: &PyEulerAngle, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_euler_angle(e.obj, order.value),
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r = bh.RotationMatrix.from_array(np.eye(3))
    ///     e = bh.EulerAngle.from_rotation_matrix(r, "XYZ")
    ///     ```
    pub fn from_rotation_matrix(_cls: &Bound<'_, PyType>, r: &PyRotationMatrix, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_rotation_matrix(r.obj, order.value),
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to quaternion representation.
    ///
    /// Returns:
    ///     Quaternion: Equivalent quaternion
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     q = e.to_quaternion()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     ea = e.to_euler_axis()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e1 = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     e2 = e1.to_euler_angle("ZYX")
    ///     ```
    pub fn to_euler_angle(&self, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(order.value)
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation.
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     r = e.to_rotation_matrix()
    ///     ```
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass(module = "brahe._brahe")]
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
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Rotation of 90 degrees about z-axis
///     axis = np.array([0.0, 0.0, 1.0])
///     e = bh.EulerAxis(axis, np.pi/2, bh.AngleFormat.RADIANS)
///     print(f"Angle: {e.angle} rad")
///
///     # Convert to quaternion
///     q = e.to_quaternion()
///     ```
pub struct PyEulerAxis {
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     print(f"Angle: {e.angle}")
    ///     ```
    pub fn angle(&self) -> f64 {
        self.obj.angle
    }

    #[getter]
    /// Get the rotation axis as a numpy array.
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element unit vector specifying rotation axis
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     print(f"Axis: {e.axis}")
    ///     ```
    pub unsafe fn axis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        vector_to_numpy!(py, self.obj.axis, 3, f64)
    }

    #[new]
    #[pyo3(text_signature = "(axis, angle, angle_format)")]
    pub fn new(axis: &Bound<'_, PyAny>, angle: f64, angle_format: &PyAngleFormat) -> PyResult<PyEulerAxis> {
        let axis_vec = pyany_to_f64_array1(axis, Some(3))?;
        Ok(PyEulerAxis {
            obj: attitude::EulerAxis::new(na::SVector::<f64, 3>::from_vec(axis_vec), angle, angle_format.value),
        })
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     e = bh.EulerAxis.from_values(0.0, 0.0, 1.0, 1.5708, bh.AngleFormat.RADIANS)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     v = np.array([0.0, 0.0, 1.0, 1.5708])
    ///     e = bh.EulerAxis.from_vector(v, bh.AngleFormat.RADIANS, True)
    ///     ```
    pub fn from_vector(_cls: &Bound<'_, PyType>, v: &Bound<'_, PyAny>, angle_format: &PyAngleFormat, vector_first: bool) -> PyResult<PyEulerAxis> {
        let vec = pyany_to_f64_array1(v, Some(4))?;
        Ok(PyEulerAxis {
            obj: attitude::EulerAxis::from_vector(na::SVector::<f64, 4>::from_vec(vec), angle_format.value, vector_first),
        })
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     e = bh.EulerAxis.from_quaternion(q)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     e1 = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     e2 = bh.EulerAxis.from_euler_axis(e1)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     euler = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     e = bh.EulerAxis.from_euler_angle(euler)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r = bh.RotationMatrix.from_array(np.eye(3))
    ///     e = bh.EulerAxis.from_rotation_matrix(r)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     q = e.to_quaternion()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     e1 = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     e2 = e1.to_euler_axis()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     e = ea.to_euler_angle("XYZ")
    ///     ```
    pub fn to_euler_angle(&self, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(order.value)
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation.
    ///
    /// Returns:
    ///     RotationMatrix: Equivalent rotation matrix
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     r = e.to_rotation_matrix()
    ///     ```
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass(module = "brahe._brahe")]
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
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create identity rotation
///     dcm = bh.RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
///
///     # Create from numpy array
///     R = np.eye(3)
///     dcm2 = bh.RotationMatrix.from_matrix(R)
///
///     # Convert to quaternion
///     q = dcm.to_quaternion()
///
///     # Rotate a vector
///     v = np.array([1.0, 0.0, 0.0])
///     v_rot = dcm.rotate_vector(v)
///     ```
pub struct PyRotationMatrix {
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     mat = np.eye(3)
    ///     r = bh.RotationMatrix.from_matrix(mat)
    ///     ```
    pub fn from_matrix(_cls: &Bound<'_, PyType>, m: &Bound<'_, PyAny>) -> Result<PyRotationMatrix, BraheError> {
        let mat_vec = pyany_to_f64_array2(m, Some((3, 3)))
            .map_err(|e| BraheError::ParseError(format!("Invalid matrix input: {}", e)))?;

        // Convert Vec<Vec<f64>> (row-major) to flat Vec<f64> (column-major) for nalgebra
        // Iterate over columns, then for each column iterate over rows to get row[col]
        let flat: Vec<f64> = (0..3)
            .flat_map(|col| mat_vec.iter().map(move |row| row[col]))
            .collect();

        Ok(PyRotationMatrix {
            obj: attitude::RotationMatrix::from_matrix(na::SMatrix::<f64, 3, 3>::from_vec(flat))?,
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

    #[getter]
    /// Get element (1,1) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 1, column 1
    pub fn r11(&self) -> f64 {
        self.obj[(0, 0)]
    }

    #[getter]
    /// Get element (1,2) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 1, column 2
    pub fn r12(&self) -> f64 {
        self.obj[(0, 1)]
    }

    #[getter]
    /// Get element (1,3) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 1, column 3
    pub fn r13(&self) -> f64 {
        self.obj[(0, 2)]
    }

    #[getter]
    /// Get element (2,1) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 2, column 1
    pub fn r21(&self) -> f64 {
        self.obj[(1, 0)]
    }

    #[getter]
    /// Get element (2,2) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 2, column 2
    pub fn r22(&self) -> f64 {
        self.obj[(1, 1)]
    }

    #[getter]
    /// Get element (2,3) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 2, column 3
    pub fn r23(&self) -> f64 {
        self.obj[(1, 2)]
    }

    #[getter]
    /// Get element (3,1) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 3, column 1
    pub fn r31(&self) -> f64 {
        self.obj[(2, 0)]
    }

    #[getter]
    /// Get element (3,2) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 3, column 2
    pub fn r32(&self) -> f64 {
        self.obj[(2, 1)]
    }

    #[getter]
    /// Get element (3,3) of the rotation matrix.
    ///
    /// Returns:
    ///     float: Matrix element at row 3, column 3
    pub fn r33(&self) -> f64 {
        self.obj[(2, 2)]
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     r = bh.RotationMatrix.Rz(1.5708, bh.AngleFormat.RADIANS)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
    ///     r = bh.RotationMatrix.from_quaternion(q)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     axis = np.array([0.0, 0.0, 1.0])
    ///     ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
    ///     r = bh.RotationMatrix.from_euler_axis(ea)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     euler = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
    ///     r = bh.RotationMatrix.from_euler_angle(euler)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r1 = bh.RotationMatrix.from_array(np.eye(3))
    ///     r2 = bh.RotationMatrix.from_rotation_matrix(r1)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r = bh.RotationMatrix.from_array(np.eye(3))
    ///     q = r.to_quaternion()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r = bh.RotationMatrix.from_array(np.eye(3))
    ///     e = r.to_euler_axis()
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r = bh.RotationMatrix.from_array(np.eye(3))
    ///     euler = r.to_euler_angle("XYZ")
    ///     ```
    pub fn to_euler_angle(&self, order: &PyEulerAngleOrder) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(order.value)
        }
    }

    #[pyo3(text_signature = "($self)")]
    /// Convert to rotation matrix representation (returns self).
    ///
    /// Returns:
    ///     RotationMatrix: This rotation matrix
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     r1 = bh.RotationMatrix.from_array(np.eye(3))
    ///     r2 = r1.to_rotation_matrix()
    ///     ```
    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}