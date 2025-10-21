// Utils module Python bindings
//
// This file contains Python bindings for utility functions including threading and cache management

// NOTE: No imports needed here as all pyo3 items are imported in mod.rs
// where this file is included

// ================================
// Cache Management Functions
// ================================

/// Get the brahe cache directory path.
///
/// The cache directory is determined by the `BRAHE_CACHE` environment variable.
/// If not set, defaults to `~/.cache/brahe`.
///
/// The directory is created if it doesn't exist.
///
/// Returns:
///     str: The full path to the cache directory.
///
/// Raises:
///     IOError: If the cache directory cannot be created or accessed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     cache_dir = bh.get_brahe_cache_dir()
///     print(f"Cache directory: {cache_dir}")
///
///     # You can also override with environment variable
///     import os
///     os.environ['BRAHE_CACHE'] = '/custom/cache/path'
///     cache_dir = bh.get_brahe_cache_dir()
///     ```
///
/// Note:
///     The directory will be created on first access if it doesn't exist.
#[pyfunction]
#[pyo3(name = "get_brahe_cache_dir")]
pub fn py_get_brahe_cache_dir() -> PyResult<String> {
    use crate::utils::cache::get_brahe_cache_dir;

    get_brahe_cache_dir().map_err(|e| exceptions::PyIOError::new_err(format!("{}", e)))
}

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
