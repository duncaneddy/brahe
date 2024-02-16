
/// Helper function to parse strings into appropriate EOPExtrapolation enumerations
fn string_to_euler_angle_order(s: &str) -> Result<attitude::EulerAngleOrder, BraheError> {
    match s.as_ref() {
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
struct PyQuaternion {
    obj: attitude::Quaternion,
}

#[pymethods]
impl PyQuaternion {

    #[new]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> PyQuaternion {
        PyQuaternion {
            obj: attitude::Quaternion::new(w, x, y, z),
        }
    }

    #[classmethod]
    pub fn from_vector(
        _cls: &PyType,
        v: &PyArray<f64, Ix1>,
        scalar_first: bool,
    ) -> PyQuaternion {
        PyQuaternion {
            obj: attitude::Quaternion::from_vector(numpy_to_vector!(v, 4, f64), scalar_first),
        }
    }

    pub unsafe fn to_vector<'py>(&self, py: Python<'py>, scalar_first: bool) -> &'py PyArray<f64, Ix1> {
        vector_to_numpy!(py, self.obj.to_vector(scalar_first), 4, f64)
    }

    pub fn normalize(&mut self) {
        self.obj.normalize();
    }

    pub fn norm(&self) -> f64 {
        self.obj.norm()
    }

    pub fn conjugate(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.conjugate()
        }
    }

    pub fn inverse(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.inverse()
        }
    }

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
    pub fn __iadd__(&mut self, other: PyQuaternion) -> () {
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

    pub fn __imul__(&mut self, other: PyQuaternion) -> () {
        self.obj *= other.obj;
    }

    pub fn __isub__(&mut self, other: PyQuaternion) -> () {
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
    pub unsafe fn data<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix1> {
        vector_to_numpy!(py, self.obj.data, 4, f64)
    }

    #[classmethod]
    pub fn from_quaternion(_cls: &PyType, q: &PyQuaternion) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_quaternion(q.obj)
        }
    }

    #[classmethod]
    pub fn from_euler_axis(_cls: &PyType, e: &PyEulerAxis) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_euler_axis(e.obj)
        }
    }

    #[classmethod]
    pub fn from_euler_angle(_cls: &PyType, e: &PyEulerAngle) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_euler_angle(e.obj)
        }
    }

    #[classmethod]
    pub fn from_rotation_matrix(_cls: &PyType, r: &PyRotationMatrix) -> PyQuaternion {
        PyQuaternion {
            obj: Quaternion::from_rotation_matrix(r.obj)
        }
    }

    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass]
#[pyo3(name = "EulerAngle")]
#[derive(Clone)]
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
    pub fn phi(&self) -> f64 {
        self.obj.phi
    }

    #[getter]
    pub fn theta(&self) -> f64 {
        self.obj.theta
    }

    #[getter]
    pub fn psi(&self) -> f64 {
        self.obj.psi
    }

    #[getter]
    pub fn order(&self) -> String {
        euler_angle_order_to_string(self.obj.order)
    }

    #[new]
    pub fn new(order: &str, phi: f64, theta: f64, psi: f64, as_degrees: bool) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::new(string_to_euler_angle_order(order).unwrap(), phi, theta, psi, as_degrees),
        }
    }

    #[classmethod]
    pub fn from_vector(_cls: &PyType, v: &PyArray<f64, Ix1>, order: &str, as_degrees: bool) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_vector(numpy_to_vector!(v, 3, f64), string_to_euler_angle_order(order).unwrap(), as_degrees),
        }
    }

    #[classmethod]
    pub fn from_quaternion(_cls: &PyType, q: &PyQuaternion, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_quaternion(q.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[classmethod]
    pub fn from_euler_axis(_cls: &PyType, e: &PyEulerAxis, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_euler_axis(e.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[classmethod]
    pub fn from_euler_angle(_cls: &PyType, e: &PyEulerAngle, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_euler_angle(e.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    #[classmethod]
    pub fn from_rotation_matrix(_cls: &PyType, r: &PyRotationMatrix, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: attitude::EulerAngle::from_rotation_matrix(r.obj, string_to_euler_angle_order(order).unwrap()),
        }
    }

    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass]
#[pyo3(name = "EulerAxis")]
#[derive(Clone)]
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
    pub fn angle(&self) -> f64 {
        self.obj.angle
    }

    #[getter]
    pub unsafe fn axis<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix1> {
        vector_to_numpy!(py, self.obj.axis, 3, f64)
    }

    #[new]
    pub fn new(axis: &PyArray<f64, Ix1>, angle: f64, as_degrees: bool) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::new(numpy_to_vector!(axis, 3, f64), angle, as_degrees),
        }
    }

    #[classmethod]
    pub fn from_values(_cls: &PyType, x: f64, y: f64, z: f64, angle: f64, as_degrees: bool) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_values(x, y, z, angle, as_degrees),
        }
    }

    #[classmethod]
    pub fn from_vector(_cls: &PyType, v: &PyArray<f64, Ix1>, as_degrees: bool, vector_first: bool) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_vector(numpy_to_vector!(v, 4, f64), as_degrees, vector_first),
        }
    }

    pub unsafe fn to_vector<'py>(&self, py: Python<'py>, as_degrees: bool, vector_first: bool) -> &'py PyArray<f64, Ix1> {
        vector_to_numpy!(py, self.obj.to_vector(as_degrees, vector_first), 4, f64)
    }

    #[classmethod]
    pub fn from_quaternion(_cls: &PyType, q: &PyQuaternion) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_quaternion(q.obj)
        }
    }

    #[classmethod]
    pub fn from_euler_axis(_cls: &PyType, e: &PyEulerAxis) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_euler_axis(e.obj)
        }
    }

    #[classmethod]
    pub fn from_euler_angle(_cls: &PyType, e: &PyEulerAngle) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_euler_angle(e.obj)
        }
    }

    #[classmethod]
    pub fn from_rotation_matrix(_cls: &PyType, r: &PyRotationMatrix) -> PyEulerAxis {
        PyEulerAxis {
            obj: attitude::EulerAxis::from_rotation_matrix(r.obj)
        }
    }

    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}

#[pyclass]
#[pyo3(name = "RotationMatrix")]
#[derive(Clone)]
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

    pub fn __imul__(&mut self, other: PyRotationMatrix) -> () {
        self.obj *= other.obj;
    }

    #[new]
    pub fn new(r11: f64, r12: f64, r13: f64, r21: f64, r22: f64, r23: f64, r31: f64, r32: f64, r33: f64) -> Result<PyRotationMatrix, BraheError> {
        Ok(PyRotationMatrix {
            obj: attitude::RotationMatrix::new(r11, r12, r13, r21, r22, r23, r31, r32, r33)?,
        })
    }

    #[classmethod]
    pub fn from_matrix(_cls: &PyType, m: &PyArray<f64, Ix2>) -> Result<PyRotationMatrix, BraheError> {
        Ok(PyRotationMatrix {
            obj: attitude::RotationMatrix::from_matrix(numpy_to_matrix!(m, 3, 3, f64))?,
        })
    }

    pub unsafe fn to_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix2> {
        matrix_to_numpy!(py, self.obj.to_matrix(), 3, 3, f64)
    }

    #[classmethod]
    #[allow(non_snake_case)]
    pub fn Rx(_cls: &PyType, angle: f64, as_degrees: bool) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::Rx(angle, as_degrees)
        }
    }

    #[classmethod]
    #[allow(non_snake_case)]
    pub fn Ry(_cls: &PyType, angle: f64, as_degrees: bool) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::Ry(angle, as_degrees)
        }
    }

    #[classmethod]
    #[allow(non_snake_case)]
    pub fn Rz(_cls: &PyType, angle: f64, as_degrees: bool) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::Rz(angle, as_degrees)
        }
    }

    #[classmethod]
    pub fn from_quaternion(_cls: &PyType, q: &PyQuaternion) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_quaternion(q.obj)
        }
    }

    #[classmethod]
    pub fn from_euler_axis(_cls: &PyType, e: &PyEulerAxis) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_euler_axis(e.obj)
        }
    }

    #[classmethod]
    pub fn from_euler_angle(_cls: &PyType, e: &PyEulerAngle) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_euler_angle(e.obj)
        }
    }

    #[classmethod]
    pub fn from_rotation_matrix(_cls: &PyType, r: &PyRotationMatrix) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: attitude::RotationMatrix::from_rotation_matrix(r.obj)
        }
    }

    pub fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            obj: self.obj.to_quaternion()
        }
    }

    pub fn to_euler_axis(&self) -> PyEulerAxis {
        PyEulerAxis {
            obj: self.obj.to_euler_axis()
        }
    }

    pub fn to_euler_angle(&self, order: &str) -> PyEulerAngle {
        PyEulerAngle {
            obj: self.obj.to_euler_angle(string_to_euler_angle_order(order).unwrap())
        }
    }

    pub fn to_rotation_matrix(&self) -> PyRotationMatrix {
        PyRotationMatrix {
            obj: self.obj.to_rotation_matrix()
        }
    }
}