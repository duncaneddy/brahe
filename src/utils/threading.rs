/*!
 * Thread pool management for parallel computation
 *
 * This module provides global thread pool management using rayon,
 * with a default of 90% of available CPU cores.
 */

use rayon::ThreadPool;
use std::sync::{Arc, LazyLock, Mutex};

// Global thread pool singleton - can be reinitialized
static THREAD_POOL: LazyLock<Mutex<Option<Arc<ThreadPool>>>> = LazyLock::new(|| Mutex::new(None));

/// Set the number of threads for parallel computation.
///
/// This function configures the global thread pool used by Brahe for
/// parallel operations (e.g., access computation). Can be called multiple
/// times to reinitialize the thread pool with a different number of threads.
///
/// # Arguments
/// * `n` - Number of threads to use. Must be at least 1.
///
/// # Panics
/// Panics if n < 1 or if the thread pool fails to build.
///
/// # Examples
/// ```
/// use brahe::utils::threading::set_num_threads;
///
/// // Use 4 threads for parallel computation
/// set_num_threads(4);
///
/// // Can be called again to change thread count
/// set_num_threads(8);
/// ```
pub fn set_num_threads(n: usize) {
    if n == 0 {
        panic!("Number of threads must be at least 1");
    }

    let new_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("Failed to build thread pool"),
    );

    let mut pool = THREAD_POOL.lock().expect("Thread pool mutex poisoned");
    *pool = Some(new_pool);
}

/// Set the thread pool to use all available CPU cores.
///
/// This is a convenience function that sets the number of threads to 100%
/// of available CPU cores. Can be called multiple times to reinitialize.
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
/// This is a fun alias for `set_max_threads()`. Can be called multiple times
/// to reinitialize.
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
///
/// # Note
/// Returns a clone of the thread pool Arc. The pool is initialized with default
/// settings if it hasn't been configured yet.
pub(crate) fn get_thread_pool() -> Arc<ThreadPool> {
    let mut pool_guard = THREAD_POOL.lock().expect("Thread pool mutex poisoned");

    if pool_guard.is_none() {
        let num_cpus = num_cpus::get();
        let default_threads = ((num_cpus as f64 * 0.9).ceil() as usize).max(1);

        *pool_guard = Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(default_threads)
                .build()
                .expect("Failed to build default thread pool"),
        ));
    }

    pool_guard.as_ref().unwrap().clone()
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
    use serial_test::serial;

    #[test]
    fn test_default_thread_count() {
        let num_cpus = num_cpus::get();
        let expected = ((num_cpus as f64 * 0.9).ceil() as usize).max(1);

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
    #[serial]
    #[should_panic(expected = "Number of threads must be at least 1")]
    fn test_set_num_threads_zero_panics() {
        set_num_threads(0);
    }

    #[test]
    fn test_set_num_threads_reinitialize() {
        // Test that we can reinitialize the thread pool without panicking
        set_num_threads(2);
        assert_eq!(get_max_threads(), 2);

        // Reinitialize with different thread count
        set_num_threads(4);
        assert_eq!(get_max_threads(), 4);

        // Reinitialize again
        set_num_threads(1);
        assert_eq!(get_max_threads(), 1);
    }

    #[test]
    #[serial]
    fn test_set_max_threads() {
        let num_cpus_val = num_cpus::get();

        // Set to max threads
        set_max_threads();
        assert_eq!(get_max_threads(), num_cpus_val);

        // Should be able to call again
        set_max_threads();
        assert_eq!(get_max_threads(), num_cpus_val);
    }

    #[test]
    #[serial]
    fn test_set_ludicrous_speed() {
        let num_cpus_val = num_cpus::get();

        // Set to ludicrous speed
        set_ludicrous_speed();
        assert_eq!(get_max_threads(), num_cpus_val);

        // Should be able to call again
        set_ludicrous_speed();
        assert_eq!(get_max_threads(), num_cpus_val);
    }

    #[test]
    fn test_get_thread_pool_returns_valid_pool() {
        // Test that get_thread_pool returns a valid thread pool
        let pool = get_thread_pool();
        let threads = pool.current_num_threads();

        // Should have at least 1 thread
        assert!(threads >= 1);
    }

    #[test]
    #[serial]
    fn test_reinitialize_changes_thread_count() {
        // Initialize with specific thread count
        set_num_threads(3);
        let pool1 = get_thread_pool();
        assert_eq!(pool1.current_num_threads(), 3);

        // Reinitialize with different count
        set_num_threads(5);
        let pool2 = get_thread_pool();
        assert_eq!(pool2.current_num_threads(), 5);

        // Verify get_max_threads reflects the change
        assert_eq!(get_max_threads(), 5);
    }

    #[test]
    #[serial]
    fn test_mixed_function_reinitialization() {
        // Start with specific thread count
        set_num_threads(2);
        assert_eq!(get_max_threads(), 2);

        // Switch to max threads
        set_max_threads();
        let max_threads = get_max_threads();
        assert!(max_threads >= 2);

        // Switch to ludicrous speed (should be same as max)
        set_ludicrous_speed();
        assert_eq!(get_max_threads(), max_threads);

        // Go back to specific count
        set_num_threads(1);
        assert_eq!(get_max_threads(), 1);

        // Back to max again
        set_max_threads();
        assert_eq!(get_max_threads(), max_threads);
    }

    #[test]
    fn test_get_max_threads_reflects_set_num_threads() {
        let test_values = [1, 2, 4, 8];

        for n in test_values {
            set_num_threads(n);
            assert_eq!(
                get_max_threads(),
                n,
                "Expected {} threads, got {}",
                n,
                get_max_threads()
            );
        }
    }

    #[test]
    fn test_thread_pool_arc_cloning() {
        // Set initial pool
        set_num_threads(4);

        // Get multiple references to the pool
        let pool1 = get_thread_pool();
        let pool2 = get_thread_pool();
        let pool3 = get_thread_pool();

        // All should report same thread count
        assert_eq!(pool1.current_num_threads(), 4);
        assert_eq!(pool2.current_num_threads(), 4);
        assert_eq!(pool3.current_num_threads(), 4);

        // After reinitialization, new references should see new count
        set_num_threads(6);
        let pool4 = get_thread_pool();
        assert_eq!(pool4.current_num_threads(), 6);

        // Old references still point to old pool (Arc behavior)
        assert_eq!(pool1.current_num_threads(), 4);
    }
}
