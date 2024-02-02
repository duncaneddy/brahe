/*!
 * Brahe crate error
 */

use std::fmt;
use std::io;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyOSError;

#[derive(Debug, PartialEq)]
pub enum BraheError {
    /// General-purpose error
    Error(String),
    /// IO error - typically from file read/write
    IoError(String),
    /// Earth Orientation Data Erro
    EOPError(String),
    /// Out of bounds error
    OutOfBoundsError(String),
}

impl fmt::Display for BraheError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BraheError::Error(e) => write!(f, "{}", e),
            BraheError::IoError(e) => write!(f, "{}", e),
            BraheError::EOPError(e) => write!(f, "{}", e),
            BraheError::OutOfBoundsError(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for BraheError {}

impl From<io::Error> for BraheError {
    fn from(error: io::Error) -> Self {
        BraheError::IoError(error.to_string())
    }
}

impl From<String> for BraheError {
    fn from(msg: String) -> Self {
        BraheError::Error(msg)
    }
}

#[cfg(feature = "python")]
impl From<BraheError> for PyErr {
    fn from(error: BraheError) -> PyErr {
        PyOSError::new_err(error.to_string())
    }
}