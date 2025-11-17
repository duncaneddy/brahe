/*!
 * Brahe crate error
 */

use std::fmt;
use std::io;
use std::num::{ParseFloatError, ParseIntError};

#[cfg(feature = "python")]
use pyo3::exceptions::PyOSError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Brahe library error types for consistent error handling across modules.
#[derive(Debug, PartialEq)]
pub enum BraheError {
    /// General-purpose error for situations not covered by other specific error types.
    /// Use when no other error category applies. Contains descriptive error message.
    Error(String),
    /// IO error - typically from file read/write operations. Includes file access errors,
    /// missing files, permission issues, or data download failures.
    IoError(String),
    /// Earth Orientation Data errors. Includes EOP provider initialization failures,
    /// data file parsing errors, out-of-range date requests, or missing EOP data.
    EOPError(String),
    /// Out of bounds error for array/vector access or invalid indices. Includes trajectory
    /// time range violations, invalid orbital element indices, or matrix dimension mismatches.
    OutOfBoundsError(String),
    /// Parse error from string-to-type conversion failures. Includes TLE parsing errors,
    /// invalid time string formats, malformed numeric values, or configuration file errors.
    ParseError(String),
    /// Initialization error when setting up library components. Includes EOP provider setup
    /// failures, gravity model loading errors, or global state initialization problems.
    InitializationError(String),
    /// Propagator error during orbit propagation. Includes SGP4 failures, numerical integrator
    /// divergence, timestep violations, or invalid initial state vectors.
    PropagatorError(String),
    /// Numerical error from mathematical operations. Includes matrix singularities,
    /// convergence failures in iterative algorithms, or numerical instability.
    NumericalError(String),
}

impl fmt::Display for BraheError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BraheError::Error(e) => write!(f, "{}", e),
            BraheError::IoError(e) => write!(f, "{}", e),
            BraheError::EOPError(e) => write!(f, "{}", e),
            BraheError::OutOfBoundsError(e) => write!(f, "{}", e),
            BraheError::ParseError(e) => write!(f, "{}", e),
            BraheError::InitializationError(e) => write!(f, "{}", e),
            BraheError::PropagatorError(e) => write!(f, "{}", e),
            BraheError::NumericalError(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for BraheError {}

impl From<io::Error> for BraheError {
    fn from(error: io::Error) -> Self {
        BraheError::IoError(error.to_string())
    }
}

impl From<ParseFloatError> for BraheError {
    fn from(error: ParseFloatError) -> Self {
        BraheError::ParseError(error.to_string())
    }
}

impl From<ParseIntError> for BraheError {
    fn from(error: ParseIntError) -> Self {
        BraheError::ParseError(error.to_string())
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_brahe_error() {
        let e = BraheError::Error("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_io_error() {
        let e = BraheError::IoError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_eop_error() {
        let e = BraheError::EOPError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_out_of_bounds_error() {
        let e = BraheError::OutOfBoundsError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_parse_error() {
        let e = BraheError::ParseError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_initialization_error() {
        let e = BraheError::InitializationError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_propagator_error() {
        let e = BraheError::PropagatorError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_numerical_error() {
        let e = BraheError::NumericalError("Test error".to_string());
        assert_eq!(e.to_string(), "Test error");
    }

    #[test]
    fn test_from_io_error() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let brahe_error = BraheError::from(io_error);
        assert!(brahe_error.to_string().contains("File not found"));
        assert!(matches!(brahe_error, BraheError::IoError(_)));
    }

    #[test]
    fn test_from_parse_float_error() {
        let result = "not_a_number".parse::<f64>();
        assert!(result.is_err());
        let parse_error = result.unwrap_err();
        let brahe_error = BraheError::from(parse_error);
        assert!(matches!(brahe_error, BraheError::ParseError(_)));
    }

    #[test]
    fn test_from_parse_int_error() {
        let result = "not_an_int".parse::<i32>();
        assert!(result.is_err());
        let parse_error = result.unwrap_err();
        let brahe_error = BraheError::from(parse_error);
        assert!(matches!(brahe_error, BraheError::ParseError(_)));
    }

    #[test]
    fn test_from_string() {
        let msg = "Test error message".to_string();
        let brahe_error = BraheError::from(msg.clone());
        assert_eq!(brahe_error.to_string(), msg);
        assert!(matches!(brahe_error, BraheError::Error(_)));
    }

    #[test]
    fn test_error_trait() {
        // Test that BraheError implements std::error::Error
        let e = BraheError::Error("Test".to_string());
        let _: &dyn std::error::Error = &e; // Verify trait is implemented
    }

    #[test]
    fn test_debug_format() {
        let e = BraheError::Error("Debug test".to_string());
        let debug_str = format!("{:?}", e);
        assert!(debug_str.contains("Error"));
        assert!(debug_str.contains("Debug test"));
    }

    #[test]
    fn test_partial_eq() {
        let e1 = BraheError::Error("Test".to_string());
        let e2 = BraheError::Error("Test".to_string());
        let e3 = BraheError::Error("Different".to_string());
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }
}
