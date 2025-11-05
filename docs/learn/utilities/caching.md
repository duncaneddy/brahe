# Caching

Brahe automatically manages a local cache directory to store downloaded data such as Earth Orientation Parameters (EOP) and TLE data. The caching utilities provide functions to locate and manage these cache directories.

For complete API details, see the [Caching API Reference](../../library_api/utils/caching.md).

## Default Cache Location

By default, Brahe stores cache data in a platform-specific location:

- **Unix/Linux/macOS**: `~/.cache/brahe`
- **Windows**: `%LOCALAPPDATA%\brahe\cache`

All cache directories are automatically created on first access, so you don't need to manually create them.

## Environment Variable Override

You can customize the cache location by setting the `BRAHE_CACHE` environment variable:

```bash
export BRAHE_CACHE=/custom/path/to/cache
```

This is useful for:

- Using a different storage location with more space
- Sharing cache data across multiple users
- Testing with isolated cache directories

## Getting Cache Directories

### Main Cache Directory

The main cache directory is the root location for all Brahe cache data:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/caching.py:15:17"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/caching.rs:12:14"
    ```

### EOP Cache Directory

Earth Orientation Parameters are stored in a dedicated subdirectory:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/caching.py:19:21"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/caching.rs:16:18"
    ```

### CelesTrak Cache Directory

Satellite TLE data downloaded from CelesTrak is stored in its own subdirectory:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/caching.py:23:25"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/caching.rs:20:22"
    ```

### Custom Subdirectories

You can create custom subdirectories within the cache for your own data:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/caching.py:27:29"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/caching.rs:24:26"
    ```

## Complete Example

Here's a complete example demonstrating all cache directory functions:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/caching.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/caching.rs:7"
    ```

## Cache Management

!!! note "Automatic Cleanup"

    Brahe does not automatically clean up old cache files. If you need to free up disk space, you can manually delete files from the cache directory. Brahe will re-download any needed data on the next request.

!!! tip "Sharing Cache Between Users"

    If you're working on server with multiple users using Brahe, you can share the same cache directory by setting the `BRAHE_CACHE` environment variable to a common location. This avoids duplicate downloads of EOP and TLE data.

---

## See Also

- [Utilities Overview](index.md) - Overview of all utilities
- [Caching API Reference](../../library_api/utils/caching.md) - Complete caching function documentation
