# Starlink Public Ephemerides

Starlink publishes the Modified ITC ephemerides it submits to Space-Track at
`https://api.starlink.com/public-files/ephemerides/`, one file per satellite, refreshed about
every eight hours. The directory has no index page; its contents are listed in a plain-text
[`MANIFEST.txt`](https://api.starlink.com/public-files/ephemerides/MANIFEST.txt). Brahe's
`StarlinkClient` retrieves them with the same caching, rate limiting and `BRAHE_NETWORK_MODE`
behavior as the other ephemeris clients, and returns either the parsed `ITC` message or an
`OrbitTrajectory` built from it. See [Modified ITC Ephemeris Format](itc.md) for the file format
itself.

## Getting Started

The following example looks up a satellite by name in the manifest and loads its ephemeris as a
trajectory. It reads a cached manifest and ephemeris file, using a week-long `cache_max_age` so the
cached manifest is not refreshed while it is under a week old.

=== "Python"
    ``` python
    --8<-- "./examples/datasets/starlink_get_ephemeris.py:12"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datasets/starlink_get_ephemeris.rs:7"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/starlink_get_ephemeris.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/starlink_get_ephemeris.rs.txt"
        ```

## The Manifest

`get_manifest` serves the cached `MANIFEST.txt` while it is younger than `cache_max_age` (3600
seconds by default) and asks the server for a newer listing otherwise, keeping the cached copy when
nothing has changed. `refresh_manifest` asks the server regardless of the cache's age, while
`cached_manifest` and `previous_manifest` never touch the network: the former returns whatever is on disk, and the latter returns the listing that
was current before the last change, kept so a caller can see what moved.

Each entry carries the NORAD catalog ID, the object name, the operational or special category, the
ephemeris start epoch at minute resolution decoded from the file name's day-time group, the
ephemeris stop epoch decoded from the file name's metadata field when it holds GPS seconds, and the
exact file name as listed. `find_by_norad_id` and `find_by_object_name` look up a single entry;
`changed_since` compares two manifests and returns the entries whose file name is new or has
changed, which answers "which satellites have new ephemerides since my last refresh". `to_dataframe`
returns the listing as a table with columns `norad_cat_id` (`UInt32`), `object_name` and `category`
(strings), `ephemeris_start` and `ephemeris_stop` (millisecond-resolution naive UTC datetimes, the
stop column null where the file name carries no stop), and `file_name`.

=== "Python"
    ``` python
    --8<-- "./examples/datasets/starlink_manifest.py:14"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datasets/starlink_manifest.rs:9"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/starlink_manifest.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/starlink_manifest.rs.txt"
        ```

## Downloading Ephemerides

`download_ephemeris` fetches a satellite's file only when the manifest names one that is not
already cached, validating the downloaded body as a Modified ITC message before writing it.
`get_ephemeris` downloads (if needed) and returns the parsed `ITC` message. `get_trajectory` does
the same and converts it to an `OrbitTrajectory` in the file's state frame (EME2000), with the RTN
covariance rotated into that frame using the block-diagonal convention; `get_trajectory` accepts an
optional `covariance_variant`, and `get_trajectory_with_covariance_variant` takes it as a required
`variant`, including `OrbitRelativeFrameVariant.ROTATING` for the alternative rotation.

!!! note
    Downloading a satellite's new ephemeris deletes the other cached files for that NORAD ID; saved copies are unaffected.

=== "Python"
    ``` python
    --8<-- "./examples/datasets/starlink_trajectory_access.py:12"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datasets/starlink_trajectory_access.rs:8"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/starlink_trajectory_access.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/starlink_trajectory_access.rs.txt"
        ```

`save_ephemeris` downloads a satellite's file and writes it to a destination outside the cache: an
existing directory, or a path whose last component has no extension, is treated as a directory and
the original file name is kept, while a path with an extension is used as the file name directly,
and a missing directory is created. A destination inside the cache directory is an error, checked
before anything is downloaded. The `keep_cached` argument chooses how the file reaches the
destination: false moves the downloaded file, so the destination becomes the only copy and a later
`download_ephemeris` or `get_ephemeris` call for the same satellite downloads it again; true copies
the file instead, so the cache keeps its copy alongside the one written to the destination.

## Bulk Downloads

`download_all(concurrency)` downloads every file the manifest lists that is not already cached, one
file per NORAD ID, using up to `concurrency` worker threads that share the client's rate limiter. As
of writing the full listing is about 11,000 files and approximately 22 GB, so a first run against the
public mirror is a long transfer; the call stops at the first failure and returns it, and it never
prunes files no longer listed. `save_all` runs `download_all` and writes every downloaded file into
a destination directory, created if missing; a destination inside the cache directory is an error,
checked before anything is downloaded. With `keep_cached` false each file is moved, so the
destination holds every listed ephemeris and the cache holds none of them, and a later
`download_ephemeris` or `download_all` call downloads them again; with `keep_cached` true each file
is copied, so the cache keeps its copies alongside the ones written to the destination. Files for
satellites no longer listed stay until `prune_cache` removes them.
`prune_cache` removes cached ephemeris files the current manifest no longer lists, without touching
`MANIFEST.txt` or `MANIFEST.previous.txt`. `cached_files` lists the ephemeris files currently in the
cache. The cache directory itself is not coordinated across processes or across clients sharing it,
so only one bulk download should run against a given cache directory at a time.

## Rate Limits and Offline Use

`StarlinkClient` rate limits requests to 1000 per minute and 30000 per hour by default, configured
through the same `RateLimitConfig` used by the other clients (see
[Rate Limiting](spacetrack/rate_limiting.md)); `max_retries` controls how many times a transient
failure is retried. `BRAHE_NETWORK_MODE` applies as it does for the other clients: `offline` serves
a cached manifest or file of any age, `offline-strict` serves only a fresh manifest and errors on
one that is stale or missing, and in `online` mode a refresh that fails is an error rather than a
silent fall back to a stale cached copy.

The manifest and ephemeris files are cached under `$BRAHE_CACHE/starlink/` for the default public
endpoint. A client constructed with any other `base_url` caches under
`$BRAHE_CACHE/starlink/mirrors/<host>-<hash>/` instead, so a client pointed at a different mirror
never reads or writes the public endpoint's cached files.

---

## See Also

- [Modified ITC Ephemeris Format](itc.md) -- The file format `StarlinkClient` downloads
- [File Operations](spacetrack/file_operations.md) -- Space-Track's own SP Ephemeris downloads, in the same format
- [Caching](../utilities/caching.md) -- Cache directory layout and offline behavior
- [Starlink API Reference](../../library_api/ephemeris/starlink.md) -- Class and method documentation
