//! Python interoperability helpers.
//!
//! Defines the `BraheError` Python exception type and the `From` conversion that
//! lets core `Result<T, BraheError>` values propagate into PyO3 return types.
//! Living in the core `brahe` crate (rather than the bindings crate) satisfies
//! Rust's orphan rule: both the trait (`From`) and the foreign type (`PyErr`)
//! sit outside `brahe-py`, so the impl must co-locate with `BraheError`.

#![allow(missing_docs)]

use pyo3::create_exception;

use super::errors::BraheError as RustBraheError;

create_exception!(brahe._brahe, BraheError, pyo3::exceptions::PyException);

impl From<RustBraheError> for pyo3::PyErr {
    fn from(error: RustBraheError) -> pyo3::PyErr {
        BraheError::new_err(error.to_string())
    }
}
