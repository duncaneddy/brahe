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

/// Get the brahe cache directory path with an optional subdirectory.
///
/// The cache directory is determined by the `BRAHE_CACHE` environment variable.
/// If not set, defaults to `~/.cache/brahe`. If a subdirectory is provided,
/// it is appended to the cache path.
///
/// The directory is created if it doesn't exist.
///
/// Args:
///     subdirectory (str or None): Optional subdirectory name to append to cache path.
///
/// Returns:
///     str: The full path to the cache directory (with subdirectory if provided).
///
/// Raises:
///     IOError: If the cache directory cannot be created or accessed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get main cache directory
///     cache_dir = bh.get_brahe_cache_dir_with_subdir(None)
///     print(f"Cache: {cache_dir}")
///
///     # Get custom subdirectory
///     custom_cache = bh.get_brahe_cache_dir_with_subdir("my_data")
///     print(f"Custom cache: {custom_cache}")
///     ```
///
/// Note:
///     The directory (and subdirectory) will be created on first access if it doesn't exist.
#[pyfunction]
#[pyo3(name = "get_brahe_cache_dir_with_subdir")]
pub fn py_get_brahe_cache_dir_with_subdir(subdirectory: Option<&str>) -> PyResult<String> {
    use crate::utils::cache::get_brahe_cache_dir_with_subdir;

    get_brahe_cache_dir_with_subdir(subdirectory).map_err(|e| exceptions::PyIOError::new_err(format!("{}", e)))
}

/// Get the EOP cache directory path.
///
/// Returns the path to the EOP (Earth Orientation Parameters) cache subdirectory.
/// Defaults to `~/.cache/brahe/eop` (or `$BRAHE_CACHE/eop` if environment variable is set).
///
/// The directory is created if it doesn't exist.
///
/// Returns:
///     str: The full path to the EOP cache directory.
///
/// Raises:
///     IOError: If the cache directory cannot be created or accessed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     eop_cache = bh.get_eop_cache_dir()
///     print(f"EOP cache: {eop_cache}")
///     ```
///
/// Note:
///     The directory will be created on first access if it doesn't exist.
#[pyfunction]
#[pyo3(name = "get_eop_cache_dir")]
pub fn py_get_eop_cache_dir() -> PyResult<String> {
    use crate::utils::cache::get_eop_cache_dir;

    get_eop_cache_dir().map_err(|e| exceptions::PyIOError::new_err(format!("{}", e)))
}

/// Get the CelesTrak cache directory path.
///
/// Returns the path to the CelesTrak cache subdirectory used for storing downloaded
/// TLE data. Defaults to `~/.cache/brahe/celestrak` (or `$BRAHE_CACHE/celestrak`
/// if environment variable is set).
///
/// The directory is created if it doesn't exist.
///
/// Returns:
///     str: The full path to the CelesTrak cache directory.
///
/// Raises:
///     IOError: If the cache directory cannot be created or accessed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     celestrak_cache = bh.get_celestrak_cache_dir()
///     print(f"CelesTrak cache: {celestrak_cache}")
///     ```
///
/// Note:
///     The directory will be created on first access if it doesn't exist.
#[pyfunction]
#[pyo3(name = "get_celestrak_cache_dir")]
pub fn py_get_celestrak_cache_dir() -> PyResult<String> {
    use crate::utils::cache::get_celestrak_cache_dir;

    get_celestrak_cache_dir().map_err(|e| exceptions::PyIOError::new_err(format!("{}", e)))
}

// ================================
// Threading Functions
// ================================

/// Set the number of threads for parallel computation.
///
/// Configures the global thread pool used by Brahe for parallel operations such as
/// access computations. This function can be called multiple times to dynamically
/// change the thread pool configuration - each call will reinitialize the pool with
/// the new thread count.
///
/// Args:
///     n (int): Number of threads to use. Must be at least 1.
///
/// Raises:
///     ValueError: If n < 1.
///     RuntimeError: If thread pool fails to build.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Set to 4 threads initially
///     bh.set_num_threads(4)
///     print(f"Threads: {bh.get_max_threads()}")  # Output: 4
///
///     # Reinitialize with 8 threads - no error!
///     bh.set_num_threads(8)
///     print(f"Threads: {bh.get_max_threads()}")  # Output: 8
///
///     # All parallel operations (e.g., location_accesses) will now use
///     # 8 threads unless overridden with AccessSearchConfig.num_threads
///     ```
///
/// Note:
///     Unlike earlier versions, this function no longer raises an error if the
///     thread pool has already been initialized. You can safely call it at any
///     time to reconfigure the thread pool.
#[pyfunction]
#[pyo3(name = "set_num_threads")]
pub fn py_set_num_threads(n: usize) -> PyResult<()> {
    use crate::utils::threading::set_num_threads;

    if n == 0 {
        return Err(exceptions::PyValueError::new_err(
            "Number of threads must be at least 1"
        ));
    }

    // Use panic::catch_unwind only for thread pool build failures
    match std::panic::catch_unwind(|| set_num_threads(n)) {
        Ok(_) => Ok(()),
        Err(_) => Err(exceptions::PyRuntimeError::new_err(
            "Failed to build thread pool"
        )),
    }
}

/// Set the thread pool to use all available CPU cores.
///
/// This is a convenience function that sets the number of threads to 100%
/// of available CPU cores. Can be called multiple times to reinitialize the
/// thread pool dynamically.
///
/// Raises:
///     RuntimeError: If thread pool fails to build.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use all available CPU cores
///     bh.set_max_threads()
///     print(f"Using all {bh.get_max_threads()} cores")
///
///     # Switch to 2 threads
///     bh.set_num_threads(2)
///
///     # Switch back to max - no error!
///     bh.set_max_threads()
///     print(f"Back to {bh.get_max_threads()} cores")
///     ```
///
/// Note:
///     This function can be called at any time, even after the thread pool
///     has been initialized with a different configuration.
#[pyfunction]
#[pyo3(name = "set_max_threads")]
pub fn py_set_max_threads() -> PyResult<()> {
    use crate::utils::threading::set_max_threads;

    // Use panic::catch_unwind only for thread pool build failures
    match std::panic::catch_unwind(set_max_threads) {
        Ok(_) => Ok(()),
        Err(_) => Err(exceptions::PyRuntimeError::new_err(
            "Failed to build thread pool"
        )),
    }
}

/// LUDICROUS SPEED! GO!
///
/// Set the thread pool to use all available CPU cores (alias for `set_max_threads`).
///
/// This is a fun alias for `set_max_threads()` that sets the number of threads
/// to 100% of available CPU cores for maximum performance. Can be called multiple
/// times to dynamically reinitialize the thread pool.
///
/// Raises:
///     RuntimeError: If thread pool fails to build.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # MAXIMUM POWER! Use all available CPU cores
///     bh.set_ludicrous_speed()
///     print(f"Going ludicrous with {bh.get_max_threads()} threads!")
///
///     # Throttle down for testing
///     bh.set_num_threads(1)
///
///     # ENGAGE LUDICROUS SPEED again - no error!
///     bh.set_ludicrous_speed()
///     ```
///
/// Note:
///     This function can be called at any time to reconfigure the thread pool
///     to use maximum available cores, regardless of previous configuration.
#[pyfunction]
#[pyo3(name = "set_ludicrous_speed")]
pub fn py_set_ludicrous_speed() -> PyResult<()> {
    // Just call set_max_threads - it's an alias
    py_set_max_threads()
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
///     # Get default thread count (90% of cores, initialized on first call)
///     threads = bh.get_max_threads()
///     print(f"Default: {threads} threads")
///
///     # Set to specific value and verify
///     bh.set_num_threads(4)
///     assert bh.get_max_threads() == 4
///
///     # Reconfigure and verify again
///     bh.set_num_threads(8)
///     assert bh.get_max_threads() == 8
///
///     # Switch to max cores
///     bh.set_max_threads()
///     print(f"Max cores: {bh.get_max_threads()}")
///     ```
///
/// Note:
///     Calling this function will initialize the thread pool with default
///     settings (90% of cores) if it hasn't been configured yet. After
///     initialization, you can still reconfigure it using set_num_threads()
///     or set_max_threads().
#[pyfunction]
#[pyo3(name = "get_max_threads")]
pub fn py_get_max_threads() -> PyResult<usize> {
    use crate::utils::threading::get_max_threads;
    Ok(get_max_threads())
}

// ================================
// Formatting Functions
// ================================

/// Format a time duration in seconds to a human-readable string.
///
/// Converts a duration in seconds to either a long format (e.g., "6 minutes and 2.00 seconds")
/// or a short format (e.g., "6m 2s").
///
/// Args:
///     seconds (float): Time duration in seconds
///     short (bool): If True, use short format; otherwise use long format (default: False)
///
/// Returns:
///     str: Human-readable string representation of the time duration
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Long format (default)
///     print(bh.format_time_string(90.0))
///     # Output: "1 minutes and 30.00 seconds"
///
///     print(bh.format_time_string(3665.0))
///     # Output: "1 hours, 1 minutes, and 5.00 seconds"
///
///     # Short format
///     print(bh.format_time_string(90.0, short=True))
///     # Output: "1m 30s"
///
///     print(bh.format_time_string(3665.0, short=True))
///     # Output: "1h 1m 5s"
///     ```
#[pyfunction]
#[pyo3(name = "format_time_string", signature = (seconds, short=false))]
pub fn py_format_time_string(seconds: f64, short: bool) -> PyResult<String> {
    use crate::utils::formatting::format_time_string;
    Ok(format_time_string(seconds, short))
}
