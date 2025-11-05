# Utilities

The `utils` module provides utility functions that support core functionality of the Brahe library. While most users won't need to interact with these functions directly, they offer control over caching, parallel computation, and output formatting.

## Caching

Brahe automatically manages a local cache directory to store downloaded data such as Earth Orientation Parameters (EOP) and TLE data. Caching both minimizes the load and impact that brahe-related requests have on host servers as well as improve performance for users by eliminating unneeded network requests. The caching utilities provide functions to locate and manage these cache directories.

By default, cache data is stored in `~/.cache/brahe` on Unix systems or the equivalent on other platforms. The location can be customized using the `BRAHE_CACHE` environment variable. Cache directories are automatically created on first access and organized into subdirectories for different data types (e.g., `eop/`, `celestrak/`).

See [Caching](caching.md) for complete details.

## Threading

Brahe uses a global thread pool to parallelize computationally intensive operations, such as computing access windows between satellites and ground locations. The threading utilities allow you to configure the number of threads used by the thread pool.

By default, Brahe uses 90% of available CPU cores, but you can manually set the thread count to optimize performance based on your workload and system resources. The thread pool can be reconfigured at any time during program execution.

See [Multithreading](threading.md) for complete details.

## String Formatting

The string formatting utilities provide functions for converting numerical values into human-readable strings. Currently, this includes formatting time durations (in seconds) into strings like "2 hours and 30.5 minutes" or "2h 30m" (short form).

These utilities are useful for displaying results to users in a more intuitive format than raw numerical values.

See [String Formatting](string_formatting.md) for complete details.

---

## See Also

- [Caching](caching.md) - Cache directory management
- [Multithreading](threading.md) - Thread pool configuration
- [String Formatting](string_formatting.md) - Human-readable output formatting
- [Utilities API Reference](../../library_api/utils/index.md) - Complete utilities function documentation