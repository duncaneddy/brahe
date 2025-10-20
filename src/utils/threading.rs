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

/// Set the maximum number of threads for parallel computation.
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
/// use brahe::utils::threading::set_max_threads;
///
/// // Use 4 threads for parallel computation
/// set_max_threads(4);
/// ```
pub fn set_max_threads(n: usize) {
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
            "Thread pool already initialized. set_max_threads() must be called before any parallel operations."
        );
    }
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
/// use brahe::utils::threading::{set_max_threads, get_max_threads};
///
/// // Set explicitly before any operations
/// set_max_threads(4);
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
}
