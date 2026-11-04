# IGS Precise Products

The International GNSS Service (IGS) and its analysis centers publish precise orbit, clock, and Earth-rotation products in public archives, organized into one directory per GPS week. Brahe's `IGSProductsClient` lists the products in those directories, downloads and caches them, and returns orbit files as parsed `SP3` objects that chain into per-satellite trajectories with `SP3Satellite.from_files`. See [SP3 Precise Orbit Files](../datafiles/sp3.md) for the file format and the trajectory types. The client uses the same caching, retry, and `BRAHE_NETWORK_MODE` behavior as the other ephemeris clients.

## Sources

A client reads from one of three archives, chosen with `IGSProductSource` or its name (`"CDDIS"`, `"GARNER"`, or `"BKG"`).

| Source | Default base URL | Holdings |
|---|---|---|
| CDDIS | `https://cddis.nasa.gov/archive/gnss/products/` | The full archive from all analysis centers. Requires a NASA Earthdata login. |
| Garner | `http://garner.ucsd.edu/pub/products/` | The Scripps Institution of Oceanography mirror. The full archive from GPS week 576, from all analysis centers. No login. |
| BKG | `https://igs.bkg.bund.de/root_ftp/IGS/products/` | The Federal Agency for Cartography and Geodesy. GPS week 2238 and later only, and only the IGS combination and CAS products. No login. |

Garner is served over plain HTTP. Its HTTPS certificate chain ends at a root authority (emSign Root TLS CA - G1) that is not in the Mozilla and `certifi` trust bundles, so verification fails, and the client therefore does not use HTTPS for it. Files downloaded from Garner are not protected in transit. BKG and CDDIS use HTTPS. A product that is missing on the chosen source is an error; the client never falls back to another source.

## Default Source and Credentials

`IGSProductsClient()` with no arguments uses CDDIS when the environment variables `CDDIS_EARTHDATA_USERNAME` and `CDDIS_EARTHDATA_PASSWORD` are both non-empty, and Garner otherwise. The Garner fallback prints a warning once per process:

```
Warning: CDDIS_EARTHDATA_USERNAME/CDDIS_EARTHDATA_PASSWORD not set; using the Garner (SIO) mirror over unencrypted HTTP. Set the variables or choose a source explicitly to silence this warning.
```

Choosing a source explicitly never warns. In Python, `IGSProductsClient(source="CDDIS", cddis_username=..., cddis_password=...)` passes credentials directly instead of reading the environment; in Rust, `IGSProductsClient::with_cddis_credentials` does the same. A `base_url` override requires an explicit source and keeps the credentials from the environment.

CDDIS redirects each request to the Earthdata login service. The client follows that redirect itself and sends the credentials, as HTTP Basic authentication, only to `urs.earthdata.nasa.gov`; the file host never receives them. The session cookie that CDDIS sets after login is kept for the redirects that follow.

## Listing Products

`list_products(epoch, ...)` lists the GPS week directory that contains `epoch`, and `list_products_between(start, end, ...)` lists every product whose coverage overlaps `[start, end]`. Each result is an `IGSProduct` holding the parsed `IGSProductFileName`, the published name, the URL, the size in bytes when the listing provides it (the Apache listings of Garner and BKG do not), and the GPS week of the directory it was listed in. `list_files(epoch)` returns the raw names of the directory, including files that are not products, and `list_weeks()` returns the week directories of the archive.

Both typed listings accept the filters `center`, `solution`, `content`, and `format`. The center is a code such as `COD` or `IGS`, compared without regard to case, and the other three take an `IGSSolutionType`, `IGSContentType`, or `IGSFileFormat`, or in Python their string codes (`"FIN"`, `"ORB"`, `"SP3"`). Names that match neither file name convention are left out of typed listings.

Two conventions name the files. Long names, used from GPS week 2238, look like `IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz` and encode the center, version, campaign, solution, start epoch, span, sampling, content, format, and compression. Short names, used before week 2238, look like `igs21000.sp3.Z` and encode the center, GPS week, day of week, and an optional ultra-rapid hour, with the content and format taken from the extension. The center and solution of a short name come from a three-letter code: `igr` is the IGS rapid product, `igu` the IGS ultra-rapid product, and unlisted codes are final products. `IGSProductFileName.parse` decodes either form, and `list_analysis_centers` returns the table of centers.

An ultra-rapid product covers 48 hours: 24 hours fitted to observations and 24 hours predicted. A short-name ultra-rapid file is filed under the week of its named time, but its coverage begins 24 hours before that time. For this reason `list_products_between` lists, besides the weeks containing `start` through `end`, the weeks that can hold products filed up to one day after `end` or beginning up to three days before `start`. Such an extra week that does not exist on the source is skipped.

Listings are cached under `$BRAHE_CACHE/igs/listings/` for one hour by default, so a product published during the day appears on the next listing. The `cache_max_age` argument sets that age in seconds.

=== "Python"
    ``` python
    --8<-- "./examples/igs/igs_list_products.py:12"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/igs/igs_list_products.rs:8"
    ```

## Downloading Products

`download(product)` fetches a product into the cache and returns the cached path of the file as published. Published products do not change, so a cached file is returned without a request. The cache path is `$BRAHE_CACHE/igs/<source>/<week>/<published name>`, and `get_igs_cache_dir()` returns the `igs` directory. `download_between(start, end, ...)` downloads every product that `list_products_between` returns for the same arguments and returns the cached paths in time order.

`save(product, destination, keep_cached=False)` downloads a product and writes it outside the cache. An existing directory, or a path whose last component has no extension, is treated as a directory and the published name is kept, while a path with an extension is the file written; a missing directory is created. A gzip product is written decompressed, and in a directory it loses the `.gz` suffix. Unix-compressed `.Z` files and uncompressed files are written unchanged. A destination inside the cache is an error, checked before anything is downloaded. With `keep_cached` false the cached copy is removed, so a later `download` fetches the file again, and with `keep_cached` true the cache keeps it. `save_between(start, end, directory, ...)` saves every product that `download_between` would return into `directory` and returns the written paths in time order.

`download_sp3(product)` downloads an SP3 product and returns the parsed `SP3`. It raises an error for a product that is not an SP3 file.

## Downloading a Range of Orbit Files

`download_sp3_range(start, end, center, solution, ...)` returns the daily orbit files of one center and solution that cover `[start, end]` as a list of `SP3` objects in time order, ready for `SP3Satellite.from_files`. With `pad=True`, the default, it adds the day before `start` and the day after `end`, so that velocities derived for the samples at the ends of the range come from centered fits. A padding file that is missing from the listing, that is a `.Z` file, or whose download answers 404 is skipped, while a missing file inside the range is an error that names the expected file and the source. Any other download or parse failure, for a padding file or a file inside the range, is an error. For long names the client selects the daily `01D` orbit file; when a center publishes several samplings, for example COD at 5 and 15 minutes, it chooses the coarsest unless `sampling` gives an interval in seconds. In either convention a gzip or uncompressed file is chosen over a `.Z` file for the same day, whatever their samplings. For short names it selects `cccwwwwd.sp3`, and `sampling` does not apply because those names do not encode one. A range that crosses GPS week 2238 returns short-name and long-name files together, and chaining them works.

=== "Python"
    ``` python
    --8<-- "./examples/igs/igs_download_range.py:12"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/igs/igs_download_range.rs:8"
    ```

### Why Ultra-Rapid Ranges Are Rejected

IGS issues an ultra-rapid orbit file every six hours, and each covers 48 hours. Consecutive issues therefore describe the same times with different values: the 00:00 issue predicts noon, and the 06:00 issue fits noon from observations. Chaining them would overlap, and choosing which issue supplies each time is a policy decision that depends on the application. `download_sp3_range` raises an error for the ultra-rapid solution before it makes any request. To use ultra-rapid orbits, list the products, choose the issues, and download them with `download_sp3`.

### Unix-Compressed Files

Files with a `.Z` suffix were compressed with the Unix `compress` utility, which Brahe does not decompress. `download` and `save` handle them as opaque files, so a raw `.Z` file can be cached or saved and decompressed externally with `gzip -d`. `download_sp3` raises an error for a `.Z` product before any request is made, and `download_sp3_range` raises it when a file inside the range is only published as `.Z`. Gzip files are decompressed by Brahe, so the error appears for short-name weeks whose orbit files an archive publishes only as `.Z`.

## Offline and Network Mode

`BRAHE_NETWORK_MODE=offline` serves cached listings of any age and cached files, and an uncached listing or file is an error that names the mode. `offline-strict` serves a cached listing only while it is younger than `cache_max_age`. Offline, the weeks that `list_products_between` adds around a range are taken from the cache when present and skipped when not, so an offline listing can omit products filed in an uncached neighboring week; the weeks that contain part of the range must be cached. Cached product files are served in every mode because they do not expire. A failed request is retried up to `max_retries` times for transient errors, and in `online` mode a failure is an error rather than a fall back to an older cached listing. See [Environment Variables](../utilities/environment_variables.md).

---

## See Also

- [SP3 Precise Orbit Files](../datafiles/sp3.md) -- The format of the orbit products and the satellite trajectory types
- [Ephemeris Data Sources](index.md) -- Overview of the other ephemeris clients
- [Starlink Public Ephemerides](starlink.md) -- A client with the same cache and save semantics
- [Caching](../utilities/caching.md) -- Cache directory layout and offline behavior
- [IGS Products API Reference](../../library_api/ephemeris/igs.md) -- Class and function documentation
