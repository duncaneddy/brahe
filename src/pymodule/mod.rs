/*!
 * This module defines the Python module for the Brahe library. It aggregates
 * all of the Python bindings for the core library into a single module.
 */

use nalgebra as na;
use numpy;
use numpy::{Ix1, Ix2, PyArray};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::PyType;
use pyo3::{exceptions, wrap_pyfunction};
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
        let arr = numpy::PyArray2::<$typ>::new($py, [$r, $c], false);

        for i in 0..$r {
            for j in 0..$c {
                arr.uget_raw([i, j]).write($mat[(i, j)])
            }
        }

        arr
    }};
}

macro_rules! vector_to_numpy {
    ($py:expr,$vec:expr,$l:expr,$typ:ty) => {{
        let arr = numpy::PyArray1::<$typ>::new($py, [$l], false);

        for i in 0..$l {
            arr.uget_raw([i]).write($vec[i])
        }

        arr
    }};
}

macro_rules! numpy_to_matrix {
    ($py:expr,$mat:expr,$r:expr,$c:expr,$typ:ty) => {{
        na::SMatrix::<$typ, $r, $c>::from_vec($mat.to_vec().unwrap())
    }};
}

macro_rules! numpy_to_vector {
    ($vec:expr,$l:expr,$typ:ty) => {{
        na::SVector::<$typ, $l>::from_vec($vec.to_vec().unwrap())
    }};
}

// Direct import files


// Define Module

#[cfg(feature = "python")] // Gate this definition behind a feature flag so it doesn't interfere with non-python builds
#[pymodule]
#[pyo3(name = "_brahe")] // See: https://www.maturin.rs/project_layout#import-rust-as-a-submodule-of-your-project
pub fn module(_py: Python, module: &PyModule) -> PyResult<()> {
    // Constants
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

    Ok(())
}