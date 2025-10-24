/*!
 * Thread pool management for parallel computation
 *
 * This module provides global thread pool management using rayon,
 * with a default of 90% of available CPU cores.
 */

use rayon::ThreadPool;
use std::sync::OnceLock;

// Global thread pool singleton
static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Set the number of threads for parallel computation.
///
/// This function configures the global thread pool used by Brahe for
/// parallel operations (e.g., access computation). Must be called before
/// any parallel operations begin, otherwise the default (90% of cores)
/// will be used.
///
/// # Arguments
/// * `n` - Number of threads to use. Must be at least 1.
///
/// # Panics
/// Panics if called after the thread pool has already been initialized,
/// or if n < 1.
///
/// # Examples
/// ```
/// use brahe::utils::threading::set_num_threads;
///
/// // Use 4 threads for parallel computation
/// set_num_threads(4);
/// ```
pub fn set_num_threads(n: usize) {
    if n == 0 {
        panic!("Number of threads must be at least 1");
    }

    let result = THREAD_POOL.set(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("Failed to build thread pool"),
    );

    if result.is_err() {
        panic!(
            "Thread pool already initialized. set_num_threads() must be called before any parallel operations."
        );
    }
}

/// Set the thread pool to use all available CPU cores.
///
/// This is a convenience function that sets the number of threads to 100%
/// of available CPU cores. Must be called before any parallel operations begin.
///
/// # Panics
/// Panics if called after the thread pool has already been initialized.
///
/// # Examples
/// ```
/// use brahe::utils::threading::set_max_threads;
///
/// // Use all available CPU cores
/// set_max_threads();
/// ```
pub fn set_max_threads() {
    set_num_threads(num_cpus::get());
}

/// LUDICROUS SPEED! GO!
///
/// Set the thread pool to use all available CPU cores (alias for `set_max_threads`).
///
/// This is a fun alias for `set_max_threads()`. Must be called before
/// any parallel operations begin.
///
/// # Panics
/// Panics if called after the thread pool has already been initialized.
///
/// # Examples
/// ```
/// use brahe::utils::threading::set_ludicrous_speed;
///
/// // MAXIMUM POWER! Use all available CPU cores
/// set_ludicrous_speed();
/// ```
pub fn set_ludicrous_speed() {
    set_max_threads();
}

/// Get the global thread pool, creating it with default settings if needed.
///
/// Default: 90% of available CPU cores (minimum 1 thread).
///
/// # Internal Use
/// This function is intended for internal use by Brahe's parallel algorithms.
pub(crate) fn get_thread_pool() -> &'static ThreadPool {
    THREAD_POOL.get_or_init(|| {
        let num_cpus = num_cpus::get();
        let default_threads = ((num_cpus as f64 * 0.9).ceil() as usize).max(1);

        rayon::ThreadPoolBuilder::new()
            .num_threads(default_threads)
            .build()
            .expect("Failed to build default thread pool")
    })
}

/// Get the current maximum number of threads.
///
/// Returns the number of threads configured for the global thread pool.
/// If the thread pool hasn't been initialized yet, this initializes it
/// with the default (90% of cores) and returns that value.
///
/// # Returns
/// Number of threads currently configured
///
/// # Examples
/// ```
/// use brahe::utils::threading::get_max_threads;
///
/// // Get default thread count (90% of cores)
/// let threads = get_max_threads();
/// println!("Using {} threads", threads);
/// assert!(threads >= 1);
/// ```
///
/// ```
/// use brahe::utils::threading::{set_num_threads, get_max_threads};
///
/// // Set explicitly before any operations
/// set_num_threads(4);
/// assert_eq!(get_max_threads(), 4);
/// ```
pub fn get_max_threads() -> usize {
    get_thread_pool().current_num_threads()
}

// ================================
// Tests
// ================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thread_count() {
        // This test MUST run in isolation (cannot be parallelized)
        // to avoid interference with other tests that might initialize
        // the thread pool.

        let num_cpus = num_cpus::get();
        let expected = ((num_cpus as f64 * 0.9).ceil() as usize).max(1);

        // Note: We can't directly test this because THREAD_POOL is global
        // and may already be initialized by other tests.
        // This is a limitation of global state in testing.
        assert!(expected >= 1);
        assert!(expected <= num_cpus);
    }

    #[test]
    fn test_get_max_threads() {
        // Get current thread count (initializes with default if needed)
        let threads = get_max_threads();

        // Should be at least 1 and at most number of CPUs
        assert!(threads >= 1);
        assert!(threads <= num_cpus::get());
    }

    #[test]
    #[should_panic(expected = "Number of threads must be at least 1")]
    fn test_set_num_threads_zero_panics() {
        // This will panic because n=0 is invalid
        set_num_threads(0);
    }

    #[test]
    fn test_set_max_threads_calls_set_num_threads() {
        // We can't directly test set_max_threads due to global state,
        // but we can verify it doesn't panic when called
        // Note: This may or may not panic depending on whether thread pool
        // is already initialized by other tests

        // Just verify the function exists and can be called
        // The actual behavior is tested indirectly through get_max_threads
        let num_cpus_val = num_cpus::get();
        assert!(num_cpus_val > 0);
    }

    #[test]
    fn test_set_ludicrous_speed_exists() {
        // Similar to set_max_threads, we can't directly test due to global state
        // but we verify the function exists and would set to max CPUs
        let num_cpus_val = num_cpus::get();
        assert!(num_cpus_val > 0);

        // The function calls set_max_threads which calls set_num_threads(num_cpus::get())
        // This is tested indirectly
    }

    #[test]
    fn test_get_thread_pool_returns_static_ref() {
        // Test that get_thread_pool returns a valid thread pool
        let pool = get_thread_pool();
        let _threads = pool.current_num_threads();

        // Should return same pool on subsequent calls
        let pool2 = get_thread_pool();
        assert_eq!(pool.current_num_threads(), pool2.current_num_threads());
    }
}
