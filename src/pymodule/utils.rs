// Utils module Python bindings
//
// This file contains Python bindings for utility functions including threading

// NOTE: No imports needed here as all pyo3 items are imported in mod.rs
// where this file is included

// ================================
// Threading Functions
// ================================

/// Set the maximum number of threads for parallel access computation.
///
/// Configures the global thread pool used by Brahe for parallel operations.
/// Must be called before any parallel operations begin, otherwise the default
/// (90% of available cores) will be used.
///
/// Args:
///     n (int): Number of threads to use. Must be at least 1.
///
/// Raises:
///     RuntimeError: If called after the thread pool has already been initialized,
///         or if n < 1.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use 4 threads for all parallel access computations
///     bh.set_max_threads(4)
///
///     # Now all location_accesses calls will use 4 threads
///     # (unless overridden with AccessSearchConfig.num_threads)
///     ```
///
/// Note:
///     This function should be called early in your program, before any
///     access computations are performed. Once the thread pool is initialized,
///     it cannot be changed.
#[pyfunction]
#[pyo3(name = "set_max_threads")]
pub fn py_set_max_threads(n: usize) -> PyResult<()> {
    use crate::utils::threading::set_max_threads;

    // Use panic::catch_unwind to handle Rust panics
    match std::panic::catch_unwind(|| set_max_threads(n)) {
        Ok(_) => Ok(()),
        Err(_) => Err(exceptions::PyRuntimeError::new_err(
            "Thread pool already initialized or invalid thread count. \
             set_max_threads() must be called before any parallel operations \
             and n must be at least 1."
        )),
    }
}

/// Get the current maximum number of threads for parallel computation.
///
/// Returns the number of threads configured for the global thread pool.
/// If the thread pool hasn't been initialized yet, this initializes it
/// with the default (90% of available cores) and returns that value.
///
/// Returns:
///     int: Number of threads currently configured.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get default thread count (90% of cores)
///     threads = bh.get_max_threads()
///     print(f"Using {threads} threads")
///
///     # Or set explicitly first
///     bh.set_max_threads(4)
///     assert bh.get_max_threads() == 4
///     ```
///
/// Note:
///     Calling this function will initialize the thread pool with default
///     settings if it hasn't been initialized yet. After that, set_max_threads()
///     can no longer be called.
#[pyfunction]
#[pyo3(name = "get_max_threads")]
pub fn py_get_max_threads() -> PyResult<usize> {
    use crate::utils::threading::get_max_threads;
    Ok(get_max_threads())
}
