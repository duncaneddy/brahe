# Environment Variables

Brahe reads a small number of environment variables that control where data is stored, whether the library may use the network, and how it authenticates to Space-Track. Each variable is read when it is needed, so a change made from inside a running program takes effect on the next operation that consults it.

| Variable | Default | Controls |
|---|---|---|
| `BRAHE_CACHE` | `~/.cache/brahe` | Root directory for every downloaded artifact: SPICE kernels, gravity models, Celestrak and Space-Track responses, EOP and space weather files, star catalogs, plot textures. |
| `BRAHE_NETWORK_MODE` | `online` | Whether brahe may open network connections, and what happens to cached files that have passed their time-to-live. |
| `SPACETRACK_USER` | unset | Space-Track.org identity used by the `brahe spacetrack` CLI commands. |
| `SPACETRACK_PASS` | unset | Space-Track.org password paired with `SPACETRACK_USER`. |

Variables that only matter when developing brahe itself (`TEST_SPACETRACK_USER`, `TEST_SPACETRACK_PASS`, `TEST_SPACETRACK_BASE_URL`, `BRAHE_FIGURE_OUTPUT_DIR`) are described in the [development guide](../../development_guide.md#developer-environment-variables).

## `BRAHE_CACHE`

Sets the cache root. Subdirectories such as `naif/`, `icgem/`, `celestrak/`, `eop/`, and `space_weather/` are created underneath it on first use. See [Caching](caching.md) for the per-dataset layout and the functions that return these paths.

```bash
export BRAHE_CACHE=/data/brahe-cache
```

## `BRAHE_NETWORK_MODE`

Selects one of three policies. Matching is case-insensitive; an unrecognized value raises an error from the first operation that would need the network or a cache decision, rather than silently behaving as `online`.

| value | requests | cached file within TTL | cached file past TTL | no cached file |
|---|---|---|---|---|
| `online` (default) | allowed | served | refreshed | downloaded |
| `offline` | never | served | served | error |
| `offline-strict` | never | served | error | error |

"Cached file" means any artifact stored under `BRAHE_CACHE`. Artifacts with no time-to-live (SPICE kernels, ICGEM models, Horizons SPKs, plot textures) never count as past TTL, so they are served in every mode once present. Artifacts with a time-to-live are Celestrak responses (2 hours by default), GCAT tables, SBDB lookups, the ICGEM model index (30 days), and the EOP and space weather files managed by the caching providers. The EOP and space weather caching providers seed a missing file from data bundled with the library before applying this policy, so for those two the "no cached file" column never applies; only the time-to-live columns govern their behavior. In the offline modes (`offline` and `offline-strict`), a model that is already downloaded is served by resolving its name from the cached index regardless of the index's own age; a model name not yet downloaded still needs an index within its time-to-live. In `online` mode a stale index is always refreshed first, so a model republished under a new hash is still re-fetched.

`offline` is the mode for machines without network access or for reproducible runs that must not depend on remote services: whatever is on disk is used, and anything missing is reported as an error naming the resource. `offline-strict` adds the requirement that cached data be within its time-to-live, which is appropriate when stale orbital or Earth orientation data would silently degrade a result.

A `cache_max_age` of zero has different effects in the two offline modes: under `offline` every cached file is stale but still served, so calling a force-refresh with a zero TTL is a no-op; under `offline-strict` every call becomes an error, since every cached file is immediately past its limit.

Requests to loopback addresses (`localhost`, `127.0.0.0/8`, `::1`) are never treated as network access and succeed in every mode, so local mock servers keep working offline.

`offline-strict` judges the EOP and space weather files by the age of the file on disk, not the epoch of the data it contains. A file seeded from the data bundled with the library carries the time it was seeded, not the bundled data's own epoch. With `auto_refresh` enabled, the caching providers apply this policy on every accessor call rather than only at construction, so a file that goes stale between calls makes every subsequent query error under `offline-strict`, not just the call that created the provider.

```bash
export BRAHE_NETWORK_MODE=offline
```

The active mode can be read back from code:

=== "Python"

    ```python
    import brahe as bh

    print(bh.network_mode())
    ```

=== "Rust"

    ```rust
    use brahe::utils::network_mode;

    println!("{}", network_mode().unwrap());
    ```

A blocked request raises an error of the form `BRAHE_NETWORK_MODE is offline; <resource> is not cached and cannot be downloaded`, where `<resource>` names the request or file, so the missing artifact can be fetched on a connected machine and copied into `BRAHE_CACHE`. A stale cache rejected under `offline-strict` raises `BRAHE_NETWORK_MODE is offline-strict; <resource> is older than its cache limit and cannot be refreshed`.

The `refresh()` methods of `CachingEOPProvider` and `CachingSpaceWeatherProvider` follow the same table: under `offline` a stale file is kept and `refresh()` returns without downloading; under `offline-strict` a stale file makes `refresh()` return an error instead.

`brahe.network_mode()` raises `RuntimeError` when `BRAHE_NETWORK_MODE` holds an unrecognized value. Library operations blocked or rejected by the mode raise `brahe.BraheError`. Downloads performed by `brahe.plots` (basemap and texture fetches) raise `RuntimeError` instead, since that code runs in Python rather than the Rust core.

## `SPACETRACK_USER` and `SPACETRACK_PASS`

Credentials for [Space-Track.org](https://www.space-track.org). The `brahe spacetrack` CLI commands read them at startup and exit with an error when either is unset; the library's `SpaceTrackClient` takes credentials as constructor arguments and does not read these variables. See [Space-Track](../ephemeris/spacetrack/index.md).

---

## See Also

- [Caching](caching.md) - Cache directory layout and helper functions
- [Utilities API Reference](../../library_api/utils/index.md)
