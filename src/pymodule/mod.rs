/*!
 * This module defines the Python module for the Brahe library. It aggregates
 * all the Python bindings for the core library into a single module.
 */

use std::path::Path;

use nalgebra as na;
use nalgebra::{SMatrix, SVector};
use numpy::{
    IntoPyArray, Ix1, Ix2, PyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods, ToPyArray, ndarray,
};

use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDateAccess, PyDateTime, PyString, PyTimeAccess, PyTuple, PyType};
use pyo3::{IntoPyObjectExt, exceptions, wrap_pyfunction};

use crate::traits::*;
use crate::*;

// NOTE: While it would be better if all bindings were in separate files,
// currently pyo3 does not support this easily. This is a known issue where
// classes defined in different rust modules cannot be passed between functions
// in the same module. This is a known issue and is being worked on.
//
// See: https://github.com/PyO3/pyo3/issues/1444

// Helper functions

macro_rules! matrix_to_numpy {
    ($py:expr,$mat:expr,$r:expr,$c:expr,$typ:ty) => {{
        let flat_vec: Vec<$typ> = (0..$r)
            .flat_map(|i| (0..$c).map(move |j| $mat[(i, j)]))
            .collect();
        flat_vec.into_pyarray($py).reshape([$r, $c]).unwrap()
    }};
}

macro_rules! vector_to_numpy {
    ($py:expr,$vec:expr,$l:expr,$typ:ty) => {{
        let flat_vec: Vec<$typ> = (0..$l).map(|i| $vec[i]).collect();
        flat_vec.into_pyarray($py)
    }};
}

/// Convert a Python object to a 1D f64 array, automatically handling dtype conversion.
///
/// This function accepts any numpy array-like object and converts it to Vec<f64>,
/// automatically converting integer arrays to float64 if needed.
///
/// # Arguments
///
/// * `arr` - Python object that should be a numpy array
/// * `expected_len` - Optional expected length for validation
///
/// # Returns
///
/// * `PyResult<Vec<f64>>` - Converted vector or error
fn pyany_to_f64_array1(arr: &Bound<'_, PyAny>, expected_len: Option<usize>) -> PyResult<Vec<f64>> {
    // Try to extract as Vec<f64> first (handles Python lists)
    if let Ok(vec) = arr.extract::<Vec<f64>>() {
        // Validate length if expected
        if let Some(len) = expected_len
            && vec.len() != len
        {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected array or list of length {}, got {}",
                len,
                vec.len()
            )));
        }
        return Ok(vec);
    }

    // Fallback to numpy array conversion (handles int arrays and other numpy dtypes)
    let py = arr.py();

    // Import numpy
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;

    // Get float64 dtype
    let float64_dtype = np
        .getattr("float64")
        .map_err(|_| exceptions::PyAttributeError::new_err("Failed to get numpy.float64"))?;

    // Convert to float64 dtype - this handles int arrays gracefully
    let arr_f64 = arr
        .call_method1("astype", (float64_dtype,))
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a numpy array or Python list"))?;

    // Downcast to PyArray<f64, Ix1>
    let pyarray = arr_f64
        .cast::<PyArray<f64, Ix1>>()
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a 1-D numpy array or list"))?;

    // Convert to vector
    let vec = pyarray
        .to_vec()
        .map_err(|_| exceptions::PyValueError::new_err("Failed to convert array to vector"))?;

    // Validate length if expected
    if let Some(len) = expected_len
        && vec.len() != len
    {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected array or list of length {}, got {}",
            len,
            vec.len()
        )));
    }

    Ok(vec)
}

/// Convert a Python object to a 2D f64 array, automatically handling dtype conversion.
///
/// This function accepts any numpy array-like object and converts it to Vec<Vec<f64>>,
/// automatically converting integer arrays to float64 if needed.
///
/// # Arguments
///
/// * `arr` - Python object that should be a 2D numpy array
/// * `expected_shape` - Optional expected shape (rows, cols) for validation
///
/// # Returns
///
/// * `PyResult<Vec<Vec<f64>>>` - Converted 2D vector or error
fn pyany_to_f64_array2(
    arr: &Bound<'_, PyAny>,
    expected_shape: Option<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    // Try to extract as Vec<Vec<f64>> first (handles nested Python lists)
    if let Ok(mat_vec) = arr.extract::<Vec<Vec<f64>>>() {
        // Validate shape if expected
        if let Some((exp_rows, exp_cols)) = expected_shape {
            if mat_vec.len() != exp_rows {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Expected matrix or list with {} rows, got {}",
                    exp_rows,
                    mat_vec.len()
                )));
            }
            for (i, row) in mat_vec.iter().enumerate() {
                if row.len() != exp_cols {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Expected {} columns in row {}, got {}",
                        exp_cols,
                        i,
                        row.len()
                    )));
                }
            }
        }
        return Ok(mat_vec);
    }

    // Fallback to numpy array conversion (handles int arrays and other numpy dtypes)
    let py = arr.py();

    // Import numpy
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;

    // Get float64 dtype
    let float64_dtype = np
        .getattr("float64")
        .map_err(|_| exceptions::PyAttributeError::new_err("Failed to get numpy.float64"))?;

    // Convert to float64 dtype
    let arr_f64 = arr.call_method1("astype", (float64_dtype,)).map_err(|_| {
        exceptions::PyTypeError::new_err("Expected a 2D numpy array or nested Python list")
    })?;

    // Downcast to PyArray<f64, Ix2>
    let pyarray = arr_f64.cast::<PyArray<f64, Ix2>>().map_err(|_| {
        exceptions::PyTypeError::new_err("Expected a 2-D numpy array or nested list")
    })?;

    // Get shape
    let shape = pyarray.shape();
    if shape.len() != 2 {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected 2-D array, got {}-D",
            shape.len()
        )));
    }
    let rows = shape[0];
    let cols = shape[1];

    // Validate shape if expected
    if let Some((exp_rows, exp_cols)) = expected_shape
        && (rows != exp_rows || cols != exp_cols)
    {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected array or list of shape ({}, {}), got ({}, {})",
            exp_rows, exp_cols, rows, cols
        )));
    }

    // Convert to Vec<Vec<f64>>
    let flat_vec = pyarray
        .to_vec()
        .map_err(|_| exceptions::PyValueError::new_err("Failed to convert array to vector"))?;

    // Reshape into 2D Vec (row-major order)
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = flat_vec[i * cols..(i + 1) * cols].to_vec();
        result.push(row);
    }

    Ok(result)
}

/// Convert a Python object to a statically-sized nalgebra vector with automatic dtype conversion.
///
/// This function accepts numpy arrays or Python lists and converts them to `SVector<f64, N>`,
/// automatically converting integer arrays/lists to float64 if needed.
///
/// # Type Parameters
///
/// * `N` - The compile-time size of the vector
///
/// # Arguments
///
/// * `arr` - Python object that should be a numpy array or list
///
/// # Returns
///
/// * `PyResult<na::SVector<f64, N>>` - Static vector with compile-time size checking
///
/// # Examples
///
/// ```rust
/// // Accept size-3 vector (position)
/// let pos = pyany_to_svector::<3>(arr)?;
///
/// // Accept size-6 vector (state)
/// let state = pyany_to_svector::<6>(arr)?;
/// ```
fn pyany_to_svector<const N: usize>(arr: &Bound<'_, PyAny>) -> PyResult<na::SVector<f64, N>> {
    // Try to extract as Vec<f64> first (handles Python lists)
    if let Ok(vec) = arr.extract::<Vec<f64>>() {
        if vec.len() != N {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected array or list of length {}, got {}",
                N,
                vec.len()
            )));
        }
        return Ok(na::SVector::<f64, N>::from_vec(vec));
    }

    // Fallback to numpy array conversion (handles int arrays and other numpy dtypes)
    let py = arr.py();

    // Import numpy
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;

    // Get float64 dtype
    let float64_dtype = np
        .getattr("float64")
        .map_err(|_| exceptions::PyAttributeError::new_err("Failed to get numpy.float64"))?;

    // Convert to float64 dtype - this handles int arrays gracefully
    let arr_f64 = arr
        .call_method1("astype", (float64_dtype,))
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a numpy array or Python list"))?;

    // Downcast to PyArray<f64, Ix1>
    let pyarray = arr_f64
        .cast::<PyArray<f64, Ix1>>()
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a 1-D numpy array or list"))?;

    // Convert to vector
    let vec = pyarray
        .to_vec()
        .map_err(|_| exceptions::PyValueError::new_err("Failed to convert array to vector"))?;

    // Validate length
    if vec.len() != N {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected array or list of length {}, got {}",
            N,
            vec.len()
        )));
    }

    Ok(na::SVector::<f64, N>::from_vec(vec))
}

/// Convert a Python object to a statically-sized nalgebra matrix with automatic dtype conversion.
///
/// This function accepts numpy 2D arrays or nested Python lists and converts them to `SMatrix<f64, R, C>`,
/// automatically converting integer arrays/lists to float64 and handling row-major to column-major conversion.
///
/// # Type Parameters
///
/// * `R` - The compile-time number of rows
/// * `C` - The compile-time number of columns
///
/// # Arguments
///
/// * `arr` - Python object that should be a 2D numpy array or nested list
///
/// # Returns
///
/// * `PyResult<na::SMatrix<f64, R, C>>` - Static matrix with compile-time size checking
///
/// # Examples
///
/// ```rust
/// // Accept 3x3 matrix (rotation matrix)
/// let rot = pyany_to_smatrix::<3, 3>(arr)?;
/// ```
#[allow(dead_code)]
fn pyany_to_smatrix<const R: usize, const C: usize>(
    arr: &Bound<'_, PyAny>,
) -> PyResult<na::SMatrix<f64, R, C>> {
    // Try to extract as Vec<Vec<f64>> first (handles nested Python lists)
    if let Ok(mat_vec) = arr.extract::<Vec<Vec<f64>>>() {
        // Validate shape
        if mat_vec.len() != R {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected matrix or list with {} rows, got {}",
                R,
                mat_vec.len()
            )));
        }
        for (i, row) in mat_vec.iter().enumerate() {
            if row.len() != C {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Expected {} columns in row {}, got {}",
                    C,
                    i,
                    row.len()
                )));
            }
        }

        // Convert Vec<Vec<f64>> (row-major) to flat Vec<f64> (column-major) for nalgebra
        let flat: Vec<f64> = (0..C)
            .flat_map(|col| mat_vec.iter().map(move |row| row[col]))
            .collect();

        return Ok(na::SMatrix::<f64, R, C>::from_vec(flat));
    }

    // Fallback to numpy array conversion (handles int arrays and other numpy dtypes)
    let py = arr.py();

    // Import numpy
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;

    // Get float64 dtype
    let float64_dtype = np
        .getattr("float64")
        .map_err(|_| exceptions::PyAttributeError::new_err("Failed to get numpy.float64"))?;

    // Convert to float64 dtype
    let arr_f64 = arr.call_method1("astype", (float64_dtype,)).map_err(|_| {
        exceptions::PyTypeError::new_err("Expected a 2D numpy array or nested Python list")
    })?;

    // Downcast to PyArray<f64, Ix2>
    let pyarray = arr_f64.cast::<PyArray<f64, Ix2>>().map_err(|_| {
        exceptions::PyTypeError::new_err("Expected a 2-D numpy array or nested list")
    })?;

    // Get shape and validate
    let shape = pyarray.shape();
    if shape.len() != 2 {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected 2-D array, got {}-D",
            shape.len()
        )));
    }
    let rows = shape[0];
    let cols = shape[1];

    if rows != R || cols != C {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected array or list of shape ({}, {}), got ({}, {})",
            R, C, rows, cols
        )));
    }

    // Convert to flat Vec (row-major from numpy)
    let flat_vec = pyarray
        .to_vec()
        .map_err(|_| exceptions::PyValueError::new_err("Failed to convert array to vector"))?;

    // Reshape into Vec<Vec<f64>> (row-major)
    let mut mat_vec = Vec::with_capacity(R);
    for i in 0..R {
        let row = flat_vec[i * C..(i + 1) * C].to_vec();
        mat_vec.push(row);
    }

    // Convert to column-major for nalgebra
    let flat: Vec<f64> = (0..C)
        .flat_map(|col| mat_vec.iter().map(move |row| row[col]))
        .collect();

    Ok(na::SMatrix::<f64, R, C>::from_vec(flat))
}

/// Python wrapper for AngleFormat enum
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AngleFormat")]
#[derive(Clone)]
pub struct PyAngleFormat {
    pub(crate) value: constants::AngleFormat,
}

#[pymethods]
impl PyAngleFormat {
    #[classattr]
    #[allow(non_snake_case)]
    fn RADIANS() -> Self {
        PyAngleFormat {
            value: constants::AngleFormat::Radians,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn DEGREES() -> Self {
        PyAngleFormat {
            value: constants::AngleFormat::Degrees,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("AngleFormat.{:?}", self.value)
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

// We directly include the contents of the module-definition files as they need to be part of a
// single module for the Python bindings to work correctly.

include!("datasets.rs");
include!("eop.rs");
include!("time.rs");
include!("frames.rs");
include!("coordinates.rs");
include!("orbits.rs");
include!("propagators.rs");
include!("attitude.rs");
include!("trajectories.rs");
include!("access.rs");
include!("relative_motion.rs");
include!("math.rs");
include!("integrators.rs");
include!("utils.rs");
include!("orbit_dynamics.rs");

// Define Module

#[cfg(feature = "python")] // Gate this definition behind a feature flag so it doesn't interfere with non-python builds
#[pymodule(name = "_brahe")] // See: https://www.maturin.rs/project_layout#import-rust-as-a-submodule-of-your-project
pub fn _brahe(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Re-export PanicException so Python tests can catch Rust panics
    module.add("PanicException", py.get_type::<PanicException>())?;

    // Add version from Cargo.toml
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;

    //* Constants *//
    module.add("DEG2RAD", constants::DEG2RAD)?;
    module.add("RAD2DEG", constants::RAD2DEG)?;
    module.add("AS2RAD", constants::AS2RAD)?;
    module.add("RAD2AS", constants::RAD2AS)?;
    module.add("MJD_ZERO", constants::MJD_ZERO)?;
    module.add("MJD2000", constants::MJD2000)?;
    module.add("GPS_TAI", constants::GPS_TAI)?;
    module.add("TAI_GPS", constants::TAI_GPS)?;
    module.add("TT_TAI", constants::TT_TAI)?;
    module.add("TAI_TT", constants::TAI_TT)?;
    module.add("GPS_TT", constants::GPS_TT)?;
    module.add("TT_GPS", constants::TT_GPS)?;
    module.add("GPS_ZERO", constants::GPS_ZERO)?;
    module.add("C_LIGHT", constants::C_LIGHT)?;
    module.add("AU", constants::AU)?;
    module.add("R_EARTH", constants::R_EARTH)?;
    module.add("WGS84_A", constants::WGS84_A)?;
    module.add("WGS84_F", constants::WGS84_F)?;
    module.add("GM_EARTH", constants::GM_EARTH)?;
    module.add("ECC_EARTH", constants::ECC_EARTH)?;
    module.add("J2_EARTH", constants::J2_EARTH)?;
    module.add("OMEGA_EARTH", constants::OMEGA_EARTH)?;
    module.add("GM_SUN", constants::GM_SUN)?;
    module.add("R_SUN", constants::R_SUN)?;
    module.add("P_SUN", constants::P_SUN)?;
    module.add("R_MOON", constants::R_MOON)?;
    module.add("GM_MOON", constants::GM_MOON)?;
    module.add("GM_MERCURY", constants::GM_MERCURY)?;
    module.add("GM_VENUS", constants::GM_VENUS)?;
    module.add("GM_MARS", constants::GM_MARS)?;
    module.add("GM_JUPITER", constants::GM_JUPITER)?;
    module.add("GM_SATURN", constants::GM_SATURN)?;
    module.add("GM_URANUS", constants::GM_URANUS)?;
    module.add("GM_NEPTUNE", constants::GM_NEPTUNE)?;
    module.add("GM_PLUTO", constants::GM_PLUTO)?;

    //* EOP *//

    // Download
    module.add_function(wrap_pyfunction!(py_download_c04_eop_file, module)?)?;
    module.add_function(wrap_pyfunction!(py_download_standard_eop_file, module)?)?;

    // Static Provider
    module.add_class::<PyStaticEOPProvider>()?;

    // File Provider
    module.add_class::<PyFileEOPProvider>()?;

    // Caching Provider
    module.add_class::<PyCachingEOPProvider>()?;

    // Global
    module.add_function(wrap_pyfunction!(py_set_global_eop_provider, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_set_global_eop_provider_from_static_provider,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_set_global_eop_provider_from_file_provider,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_set_global_eop_provider_from_caching_provider,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_get_global_ut1_utc, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_pm, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_dxdy, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_lod, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_initialization, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_len, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_type, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_extrapolation, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_interpolation, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_mjd_min, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_mjd_max, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_mjd_last_lod, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_global_eop_mjd_last_dxdy, module)?)?;
    module.add_function(wrap_pyfunction!(py_initialize_eop, module)?)?;

    //* Time *//

    module.add_class::<PyTimeSystem>()?;
    module.add_class::<PyEpoch>()?;
    module.add_class::<PyTimeRange>()?;
    module.add_function(wrap_pyfunction!(py_mjd_to_datetime, module)?)?;
    module.add_function(wrap_pyfunction!(py_datetime_to_mjd, module)?)?;
    module.add_function(wrap_pyfunction!(py_jd_to_datetime, module)?)?;
    module.add_function(wrap_pyfunction!(py_datetime_to_jd, module)?)?;
    module.add_function(wrap_pyfunction!(py_time_system_offset_for_mjd, module)?)?;
    module.add_function(wrap_pyfunction!(py_time_system_offset_for_jd, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_time_system_offset_for_datetime,
        module
    )?)?;

    // Top-level time system constants
    module.add(
        "GPS",
        PyTimeSystem {
            ts: time::TimeSystem::GPS,
        },
    )?;
    module.add(
        "TAI",
        PyTimeSystem {
            ts: time::TimeSystem::TAI,
        },
    )?;
    module.add(
        "TT",
        PyTimeSystem {
            ts: time::TimeSystem::TT,
        },
    )?;
    module.add(
        "UTC",
        PyTimeSystem {
            ts: time::TimeSystem::UTC,
        },
    )?;
    module.add(
        "UT1",
        PyTimeSystem {
            ts: time::TimeSystem::UT1,
        },
    )?;

    //* Frames *//
    module.add_function(wrap_pyfunction!(py_bias_precession_nutation, module)?)?;
    module.add_function(wrap_pyfunction!(py_earth_rotation, module)?)?;
    module.add_function(wrap_pyfunction!(py_polar_motion, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_gcrf_to_itrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_itrf_to_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_eci_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_ecef_to_eci, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_gcrf_to_itrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_itrf_to_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_eci_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_ecef_to_eci, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_gcrf_to_itrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_itrf_to_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_eci_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_ecef_to_eci, module)?)?;
    module.add_function(wrap_pyfunction!(py_bias_eme2000, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_gcrf_to_eme2000, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_eme2000_to_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_gcrf_to_eme2000, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_eme2000_to_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_gcrf_to_eme2000, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_eme2000_to_gcrf, module)?)?;

    //* Coordinates *//

    // Coordinate Types
    module.add_class::<PyEllipsoidalConversionType>()?;

    // Cartesian
    module.add_function(wrap_pyfunction!(py_state_osculating_to_cartesian, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_cartesian_to_osculating, module)?)?;

    // Geocentric
    module.add_function(wrap_pyfunction!(py_position_geocentric_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_ecef_to_geocentric, module)?)?;

    // Geodetic
    module.add_function(wrap_pyfunction!(py_position_geodetic_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_ecef_to_geodetic, module)?)?;

    // Topocentric
    module.add_function(wrap_pyfunction!(py_rotation_ellipsoid_to_enz, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_enz_to_ellipsoid, module)?)?;
    module.add_function(wrap_pyfunction!(py_relative_position_ecef_to_enz, module)?)?;
    module.add_function(wrap_pyfunction!(py_relative_position_enz_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_ellipsoid_to_sez, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_sez_to_ellipsoid, module)?)?;
    module.add_function(wrap_pyfunction!(py_relative_position_ecef_to_sez, module)?)?;
    module.add_function(wrap_pyfunction!(py_relative_position_sez_to_ecef, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_enz_to_azel, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_sez_to_azel, module)?)?;

    //* Orbits *//
    module.add_function(wrap_pyfunction!(py_orbital_period, module)?)?;
    module.add_function(wrap_pyfunction!(py_orbital_period_general, module)?)?;
    module.add_function(wrap_pyfunction!(py_orbital_period_from_state, module)?)?;
    module.add_function(wrap_pyfunction!(py_mean_motion, module)?)?;
    module.add_function(wrap_pyfunction!(py_mean_motion_general, module)?)?;
    module.add_function(wrap_pyfunction!(py_semimajor_axis, module)?)?;
    module.add_function(wrap_pyfunction!(py_semimajor_axis_general, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_semimajor_axis_from_orbital_period,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py_semimajor_axis_from_orbital_period_general,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_perigee_velocity, module)?)?;
    module.add_function(wrap_pyfunction!(py_periapsis_velocity, module)?)?;
    module.add_function(wrap_pyfunction!(py_periapsis_distance, module)?)?;
    module.add_function(wrap_pyfunction!(py_apogee_velocity, module)?)?;
    module.add_function(wrap_pyfunction!(py_apoapsis_velocity, module)?)?;
    module.add_function(wrap_pyfunction!(py_apoapsis_distance, module)?)?;
    module.add_function(wrap_pyfunction!(py_periapsis_altitude, module)?)?;
    module.add_function(wrap_pyfunction!(py_perigee_altitude, module)?)?;
    module.add_function(wrap_pyfunction!(py_apoapsis_altitude, module)?)?;
    module.add_function(wrap_pyfunction!(py_apogee_altitude, module)?)?;
    module.add_function(wrap_pyfunction!(py_sun_synchronous_inclination, module)?)?;
    module.add_function(wrap_pyfunction!(py_anomaly_eccentric_to_mean, module)?)?;
    module.add_function(wrap_pyfunction!(py_anomaly_mean_to_eccentric, module)?)?;
    module.add_function(wrap_pyfunction!(py_anomaly_true_to_eccentric, module)?)?;
    module.add_function(wrap_pyfunction!(py_anomaly_eccentric_to_true, module)?)?;
    module.add_function(wrap_pyfunction!(py_anomaly_true_to_mean, module)?)?;
    module.add_function(wrap_pyfunction!(py_anomaly_mean_to_true, module)?)?;

    // Propagator Support
    module.add_class::<PySGPPropagator>()?;
    module.add_class::<PyKeplerianPropagator>()?;
    module.add_function(wrap_pyfunction!(py_par_propagate_to, module)?)?;

    // TLE Support
    module.add_function(wrap_pyfunction!(py_validate_tle_lines, module)?)?;
    module.add_function(wrap_pyfunction!(py_validate_tle_line, module)?)?;
    module.add_function(wrap_pyfunction!(py_calculate_tle_line_checksum, module)?)?;
    module.add_function(wrap_pyfunction!(py_parse_norad_id, module)?)?;
    module.add_function(wrap_pyfunction!(py_norad_id_numeric_to_alpha5, module)?)?;
    module.add_function(wrap_pyfunction!(py_norad_id_alpha5_to_numeric, module)?)?;

    // New TLE conversion functions
    module.add_function(wrap_pyfunction!(py_keplerian_elements_from_tle, module)?)?;
    module.add_function(wrap_pyfunction!(py_keplerian_elements_to_tle, module)?)?;
    module.add_function(wrap_pyfunction!(py_create_tle_lines, module)?)?;
    module.add_function(wrap_pyfunction!(py_epoch_from_tle, module)?)?;

    //* Relative Motion *//
    module.add_function(wrap_pyfunction!(py_rotation_rtn_to_eci, module)?)?;
    module.add_function(wrap_pyfunction!(py_rotation_eci_to_rtn, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_eci_to_rtn, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_rtn_to_eci, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_oe_to_roe, module)?)?;
    module.add_function(wrap_pyfunction!(py_state_roe_to_oe, module)?)?;

    //* Trajectories *//
    module.add_class::<PyOrbitFrame>()?;
    module.add_class::<PyOrbitRepresentation>()?;
    module.add_class::<PyAngleFormat>()?;
    module.add_class::<PyInterpolationMethod>()?;
    module.add_class::<PyCovarianceInterpolationMethod>()?;
    module.add_class::<PyOrbitalTrajectory>()?;
    module.add_class::<PyTrajectory>()?;
    module.add_class::<PySTrajectory6>()?;

    //* Attitude *//
    module.add_class::<PyQuaternion>()?;
    module.add_class::<PyEulerAxis>()?;
    module.add_class::<PyEulerAngle>()?;
    module.add_class::<PyEulerAngleOrder>()?;
    module.add_class::<PyRotationMatrix>()?;

    //* Datasets *//
    module.add_function(wrap_pyfunction!(py_celestrak_get_tles, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_celestrak_get_tles_as_propagators,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_celestrak_download_tles, module)?)?;
    module.add_function(wrap_pyfunction!(py_celestrak_get_tle_by_id, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_celestrak_get_tle_by_id_as_propagator,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_celestrak_get_tle_by_name, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_celestrak_get_tle_by_name_as_propagator,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_groundstations_load, module)?)?;
    module.add_function(wrap_pyfunction!(py_groundstations_load_from_file, module)?)?;
    module.add_function(wrap_pyfunction!(py_groundstations_load_all, module)?)?;
    module.add_function(wrap_pyfunction!(py_groundstations_list_providers, module)?)?;
    module.add_function(wrap_pyfunction!(py_naif_download_de_kernel, module)?)?;

    //* Orbit Dynamics - Ephemerides *//
    module.add_function(wrap_pyfunction!(py_sun_position, module)?)?;
    module.add_function(wrap_pyfunction!(py_moon_position, module)?)?;
    module.add_function(wrap_pyfunction!(py_sun_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_moon_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_mercury_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_venus_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_mars_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_jupiter_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_saturn_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_uranus_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_neptune_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_solar_system_barycenter_position_de440s,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_ssb_position_de440s, module)?)?;
    module.add_function(wrap_pyfunction!(py_initialize_ephemeris, module)?)?;

    //* Access *//

    // Enums
    module.add_class::<PyLookDirection>()?;
    module.add_class::<PyAscDsc>()?;

    // Constraints
    module.add_class::<PyElevationConstraint>()?;
    module.add_class::<PyElevationMaskConstraint>()?;
    module.add_class::<PyOffNadirConstraint>()?;
    module.add_class::<PyLocalTimeConstraint>()?;
    module.add_class::<PyLookDirectionConstraint>()?;
    module.add_class::<PyAscDscConstraint>()?;

    // Constraint Composition
    module.add_class::<PyConstraintAll>()?;
    module.add_class::<PyConstraintAny>()?;
    module.add_class::<PyConstraintNot>()?;

    // Locations
    module.add_class::<PyPointLocation>()?;
    module.add_class::<PyPolygonLocation>()?;
    module.add_class::<PyPropertiesDict>()?;

    // Access Properties
    module.add_class::<PyAccessWindow>()?;
    module.add_class::<PyAccessProperties>()?;
    module.add_class::<PyAccessSearchConfig>()?;
    module.add_class::<PyAdditionalPropertiesDict>()?;
    module.add_class::<PySamplingConfig>()?;
    module.add_class::<PyDopplerComputer>()?;
    module.add_class::<PyRangeComputer>()?;
    module.add_class::<PyRangeRateComputer>()?;
    module.add_class::<PyAccessPropertyComputer>()?;
    module.add_class::<PyAccessConstraintComputer>()?;

    // Access Computation
    module.add_function(wrap_pyfunction!(py_location_accesses, module)?)?;

    //* Utils *//
    // Cache Management
    module.add_function(wrap_pyfunction!(py_get_brahe_cache_dir, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_get_brahe_cache_dir_with_subdir,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_get_eop_cache_dir, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_celestrak_cache_dir, module)?)?;

    // Threading
    module.add_function(wrap_pyfunction!(py_set_num_threads, module)?)?;
    module.add_function(wrap_pyfunction!(py_set_max_threads, module)?)?;
    module.add_function(wrap_pyfunction!(py_set_ludicrous_speed, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_max_threads, module)?)?;

    // Formatting
    module.add_function(wrap_pyfunction!(py_format_time_string, module)?)?;

    //* Jacobian *//
    module.add_class::<PyDifferenceMethod>()?;
    module.add_class::<PyPerturbationStrategy>()?;
    module.add_class::<PyDNumericalJacobian>()?;
    module.add_class::<PyDAnalyticJacobian>()?;

    //* Integrators *//
    module.add_class::<PyIntegratorConfig>()?;
    module.add_class::<PyAdaptiveStepDResult>()?;
    module.add_class::<PyRK4DIntegrator>()?;
    module.add_class::<PyRKF45DIntegrator>()?;
    module.add_class::<PyDP54DIntegrator>()?;
    module.add_class::<PyRKN1210DIntegrator>()?;

    Ok(())
}
