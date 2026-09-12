// Python bindings for the Starlink client module.

/// One line of the Starlink manifest with the epochs decoded from the name.
///
/// ``ephemeris_start`` has minute resolution (the file name carries only
/// ``DDDHHMM``); its year comes from ``ephemeris_stop`` when Starlink's
/// metadata field decodes as GPS seconds, otherwise from the manifest's
/// reference epoch.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     manifest = bh.StarlinkManifest.parse(
///         "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\n",
///         bh.Epoch(2026, 9, 11, 6, 0, 0.0, 0.0, time_system=bh.UTC),
///     )
///     entry = manifest.find_by_norad_id(100001)
///     assert entry.object_name == "STARLINK-38128"
///     assert entry.ephemeris_stop is not None
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "StarlinkManifestEntry")]
#[derive(Clone)]
pub struct PyStarlinkManifestEntry {
    pub(crate) inner: starlink::StarlinkManifestEntry,
}

#[pymethods]
impl PyStarlinkManifestEntry {
    /// Parsed view of the listing's file name.
    ///
    /// Rendering it back to a string normalises the catalog number and the
    /// category spelling, so requests, cache files and prune keys use
    /// ``file_name_string()``, which returns the listing's exact text.
    ///
    /// Returns:
    ///     EphemerisFileName: The manifest line's file name, field by field.
    #[getter]
    fn file_name(&self) -> PyEphemerisFileName {
        PyEphemerisFileName {
            inner: self.inner.file_name.clone(),
        }
    }

    /// NORAD catalog number.
    ///
    /// Returns:
    ///     int: Catalog number.
    #[getter]
    fn norad_cat_id(&self) -> u32 {
        self.inner.norad_cat_id
    }

    /// Common name, for example ``STARLINK-38128``.
    ///
    /// Returns:
    ///     str: Object name.
    #[getter]
    fn object_name(&self) -> String {
        self.inner.object_name.clone()
    }

    /// Operational or Special.
    ///
    /// Returns:
    ///     EphemerisFileCategory: The category.
    #[getter]
    fn category(&self) -> PyEphemerisFileCategory {
        PyEphemerisFileCategory {
            inner: self.inner.category,
        }
    }

    /// Ephemeris start, UTC, minute resolution.
    ///
    /// Returns:
    ///     Epoch: The start epoch.
    #[getter]
    fn ephemeris_start(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.ephemeris_start,
        }
    }

    /// Ephemeris stop decoded from the metadata field, when present.
    ///
    /// Returns:
    ///     Epoch | None: The stop epoch, or None when the metadata field does not decode as GPS seconds.
    #[getter]
    fn ephemeris_stop(&self) -> Option<PyEpoch> {
        self.inner.ephemeris_stop.map(|obj| PyEpoch { obj })
    }

    /// The file name exactly as listed in the manifest, which is what the
    /// mirror serves and what the cache stores.
    ///
    /// Returns:
    ///     str: The listing's file name with extension.
    fn file_name_string(&self) -> String {
        self.inner.file_name_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "StarlinkManifestEntry(norad_cat_id={}, object_name='{}', file_name='{}')",
            self.inner.norad_cat_id,
            self.inner.object_name,
            self.inner.file_name_string()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Starlink's manifest as a table of entries.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.StarlinkClient()
///     manifest = client.get_manifest()
///     print(len(manifest))
///     for entry in manifest:
///         print(entry.norad_cat_id)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "StarlinkManifest")]
#[derive(Clone)]
pub struct PyStarlinkManifest {
    pub(crate) inner: starlink::StarlinkManifest,
}

#[pymethods]
impl PyStarlinkManifest {
    /// Parses manifest text; blank lines are skipped and any other line must
    /// be a Space-Track file name.
    ///
    /// Args:
    ///     text (str): Manifest contents.
    ///     reference (Epoch): Epoch used to place start dates in a year and to validate GPS-second metadata.
    ///     last_modified (Epoch, optional): Server ``Last-Modified`` for the listing, if known.
    ///
    /// Returns:
    ///     StarlinkManifest: The parsed table.
    ///
    /// Raises:
    ///     BraheError: If any non-blank line is not a compliant file name.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     reference = bh.Epoch(2026, 9, 11, 6, 0, 0.0, 0.0, time_system=bh.UTC)
    ///     manifest = bh.StarlinkManifest.parse(
    ///         "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\n",
    ///         reference,
    ///     )
    ///     assert manifest.entries()[0].norad_cat_id == 100001
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (text, reference, last_modified=None))]
    fn parse(
        text: &str,
        reference: &PyEpoch,
        last_modified: Option<&PyEpoch>,
    ) -> PyResult<Self> {
        starlink::StarlinkManifest::parse(text, reference.obj, last_modified.map(|e| e.obj))
            .map(|inner| Self { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// ``Last-Modified`` reported by the server for this listing, when known.
    ///
    /// Returns:
    ///     Epoch | None: The listing's last-modified time.
    #[getter]
    fn last_modified(&self) -> Option<PyEpoch> {
        self.inner.last_modified.map(|obj| PyEpoch { obj })
    }

    /// When this copy of the listing was retrieved or cached.
    ///
    /// Returns:
    ///     Epoch: The retrieval time.
    #[getter]
    fn retrieved(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.retrieved,
        }
    }

    /// All entries in manifest order.
    ///
    /// Returns:
    ///     list[StarlinkManifestEntry]: The entries.
    fn entries(&self) -> Vec<PyStarlinkManifestEntry> {
        self.inner
            .entries()
            .iter()
            .map(|e| PyStarlinkManifestEntry { inner: e.clone() })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __iter__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let entries = self.entries();
        let list = PyList::new(py, entries)?;
        Ok(list.try_iter()?.into())
    }

    fn __getitem__(&self, index: isize) -> PyResult<PyStarlinkManifestEntry> {
        let actual = normalize_index(index, self.inner.len(), "StarlinkManifest")?;
        Ok(PyStarlinkManifestEntry {
            inner: self.inner.entries()[actual].clone(),
        })
    }

    /// Whether the manifest has no entries.
    ///
    /// Returns:
    ///     bool: True when empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Entry for a NORAD catalog number.
    ///
    /// Args:
    ///     norad_cat_id (int): Catalog number.
    ///
    /// Returns:
    ///     StarlinkManifestEntry | None: The entry, or None if absent.
    fn find_by_norad_id(&self, norad_cat_id: u32) -> Option<PyStarlinkManifestEntry> {
        self.inner
            .find_by_norad_id(norad_cat_id)
            .map(|e| PyStarlinkManifestEntry { inner: e.clone() })
    }

    /// Entry whose object name matches exactly.
    ///
    /// Args:
    ///     object_name (str): Common name, case-sensitive.
    ///
    /// Returns:
    ///     StarlinkManifestEntry | None: The entry, or None if absent.
    fn find_by_object_name(&self, object_name: &str) -> Option<PyStarlinkManifestEntry> {
        self.inner
            .find_by_object_name(object_name)
            .map(|e| PyStarlinkManifestEntry { inner: e.clone() })
    }

    /// Entries whose file name differs from, or is absent in, ``other``.
    ///
    /// Args:
    ///     other (StarlinkManifest): An earlier manifest to compare against.
    ///
    /// Returns:
    ///     list[StarlinkManifestEntry]: New or updated entries, in manifest order.
    fn changed_since(&self, other: &PyStarlinkManifest) -> Vec<PyStarlinkManifestEntry> {
        self.inner
            .changed_since(&other.inner)
            .into_iter()
            .map(|e| PyStarlinkManifestEntry { inner: e.clone() })
            .collect()
    }

    /// The manifest as a polars ``DataFrame``.
    ///
    /// Columns: ``norad_cat_id`` (UInt32), ``object_name`` (String),
    /// ``category`` (String), ``ephemeris_start`` (Datetime, milliseconds),
    /// ``ephemeris_stop`` (Datetime, milliseconds, null when the name carries
    /// no stop), ``file_name`` (String).
    ///
    /// Returns:
    ///     polars.DataFrame: One row per entry.
    ///
    /// Raises:
    ///     BraheError: If polars is not installed or the DataFrame cannot be built.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.StarlinkClient()
    ///     df = client.get_manifest().to_dataframe()
    ///     print(df.head())
    ///     ```
    fn to_dataframe(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import("polars")?;
        let entries = self.inner.entries();

        let millis = |epoch: &time::Epoch| -> i64 { (epoch.unix_timestamp() * 1000.0).round() as i64 };

        let norad_list = PyList::empty(py);
        let name_list = PyList::empty(py);
        let category_list = PyList::empty(py);
        let start_list = PyList::empty(py);
        let stop_list = PyList::empty(py);
        let file_list = PyList::empty(py);

        for e in entries {
            norad_list.append(e.norad_cat_id)?;
            name_list.append(&e.object_name)?;
            category_list.append(e.category.to_string())?;
            start_list.append(millis(&e.ephemeris_start))?;
            stop_list.append(e.ephemeris_stop.as_ref().map(millis))?;
            file_list.append(e.file_name_string())?;
        }

        let data = PyDict::new(py);
        data.set_item("norad_cat_id", norad_list)?;
        data.set_item("object_name", name_list)?;
        data.set_item("category", category_list)?;
        data.set_item("ephemeris_start", start_list)?;
        data.set_item("ephemeris_stop", stop_list)?;
        data.set_item("file_name", file_list)?;

        let datetime_ms = polars.getattr("Datetime")?.call1(("ms",))?;
        let schema = PyDict::new(py);
        schema.set_item("norad_cat_id", polars.getattr("UInt32")?)?;
        schema.set_item("object_name", polars.getattr("String")?)?;
        schema.set_item("category", polars.getattr("String")?)?;
        schema.set_item("ephemeris_start", &datetime_ms)?;
        schema.set_item("ephemeris_stop", &datetime_ms)?;
        schema.set_item("file_name", polars.getattr("String")?)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("schema", schema)?;
        let df = polars.call_method("DataFrame", (data,), Some(&kwargs))?;
        Ok(df.unbind())
    }

    fn __repr__(&self) -> String {
        format!("StarlinkManifest(entries={})", self.inner.len())
    }
}

/// Client for Starlink's public Modified ITC ephemerides.
///
/// Requests are blocking, rate limited (1000 per minute and 30000 per hour by
/// default), retried on transient failures, and honour ``BRAHE_NETWORK_MODE``.
/// The manifest is cached under ``$BRAHE_CACHE/starlink/MANIFEST.txt`` for
/// the default base URL, or under ``$BRAHE_CACHE/starlink/mirrors/<label>``
/// for any other base URL so clients pointed at different mirrors never
/// share a cache, and re-fetched with a conditional GET once it is older
/// than ``cache_max_age``; when the listing changes, the prior copy is kept
/// as ``MANIFEST.previous.txt`` so the caller can ask which satellites
/// moved. The cache directory is not coordinated across processes or across
/// clients sharing it; run one bulk download at a time.
///
/// Args:
///     base_url (str, optional): Custom base URL for testing or a mirror.
///     cache_max_age (float, optional): Seconds a cached manifest is served without a refresh. Default: 3600.0.
///     max_retries (int, optional): Extra attempts after the first on transient failures. Default: 3.
///     rate_limit (RateLimitConfig, optional): Custom per-minute and per-hour request caps.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.StarlinkClient()
///     manifest = client.get_manifest()
///     traj = client.get_trajectory(manifest[0].norad_cat_id)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "StarlinkClient")]
pub struct PyStarlinkClient {
    inner: starlink::StarlinkClient,
}

#[pymethods]
impl PyStarlinkClient {
    #[new]
    #[pyo3(signature = (base_url=None, cache_max_age=None, max_retries=None, rate_limit=None))]
    fn new(
        base_url: Option<&str>,
        cache_max_age: Option<f64>,
        max_retries: Option<u32>,
        rate_limit: Option<PyRateLimitConfig>,
    ) -> Self {
        let mut client = match (base_url, cache_max_age) {
            (Some(url), Some(age)) => {
                starlink::StarlinkClient::with_base_url_and_cache_age(url, age)
            }
            (Some(url), None) => starlink::StarlinkClient::with_base_url(url),
            (None, Some(age)) => starlink::StarlinkClient::with_cache_age(age),
            (None, None) => starlink::StarlinkClient::new(),
        };
        if let Some(n) = max_retries {
            client = client.max_retries(n);
        }
        if let Some(rl) = rate_limit {
            client = client.rate_limit(rl.inner);
        }
        PyStarlinkClient { inner: client }
    }

    /// Base URL without a trailing slash.
    ///
    /// Returns:
    ///     str: The directory URL.
    #[getter]
    fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    /// Manifest cache age in seconds.
    ///
    /// Returns:
    ///     float: Seconds.
    #[getter]
    fn cache_max_age(&self) -> f64 {
        self.inner.cache_max_age()
    }

    /// Directory holding the cached manifest and ephemeris files.
    ///
    /// For the default base URL this is ``$BRAHE_CACHE/starlink``. For any
    /// other base URL it is ``$BRAHE_CACHE/starlink/mirrors/<label>``, where
    /// ``<label>`` is derived from the base URL's host so that clients
    /// pointed at different mirrors do not read or write each other's
    /// cached manifest and ephemeris files.
    ///
    /// Returns:
    ///     str: The cache directory, created if missing.
    ///
    /// Raises:
    ///     BraheError: If the cache directory cannot be created.
    fn cache_dir(&self) -> PyResult<String> {
        self.inner
            .cache_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Returns the manifest, serving the cached copy while it is younger than
    /// ``cache_max_age`` and refreshing it with a conditional GET otherwise.
    ///
    /// ``BRAHE_NETWORK_MODE`` applies as for the other clients: ``offline``
    /// serves a cached manifest of any age, ``offline-strict`` serves only a
    /// fresh one and rejects a stale or missing one, and a refresh that fails
    /// is an error rather than a silent fall back to the stale copy.
    ///
    /// Returns:
    ///     StarlinkManifest: The current listing.
    ///
    /// Raises:
    ///     BraheError: If the cached copy cannot be served under the current mode and no refresh succeeds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     manifest = bh.StarlinkClient().get_manifest()
    ///     assert len(manifest) > 0
    ///     ```
    fn get_manifest(&self, py: Python<'_>) -> PyResult<PyStarlinkManifest> {
        py.detach(|| self.inner.get_manifest())
            .map(|inner| PyStarlinkManifest { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Fetches the manifest from the server regardless of cache age.
    ///
    /// Sends the cached ``ETag`` and ``Last-Modified`` as conditional
    /// headers; a 304 answer touches the cached file so it is fresh again. A
    /// changed listing moves the old copy to ``MANIFEST.previous.txt`` before
    /// the new one is written.
    ///
    /// Returns:
    ///     StarlinkManifest: The listing now on disk.
    ///
    /// Raises:
    ///     BraheError: On network failure, offline mode, or a malformed listing.
    fn refresh_manifest(&self, py: Python<'_>) -> PyResult<PyStarlinkManifest> {
        py.detach(|| self.inner.refresh_manifest())
            .map(|inner| PyStarlinkManifest { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// The cached manifest, whatever its age, without touching the network.
    ///
    /// Returns:
    ///     StarlinkManifest | None: The cached listing, or None if nothing is cached.
    ///
    /// Raises:
    ///     BraheError: If the cached file cannot be read or parsed.
    fn cached_manifest(&self, py: Python<'_>) -> PyResult<Option<PyStarlinkManifest>> {
        py.detach(|| self.inner.cached_manifest())
            .map(|opt| opt.map(|inner| PyStarlinkManifest { inner }))
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// The listing that was current before the last change, for ``changed_since``.
    ///
    /// Returns:
    ///     StarlinkManifest | None: The previous listing, or None if the manifest has never changed on this machine.
    ///
    /// Raises:
    ///     BraheError: If the file cannot be read or parsed.
    fn previous_manifest(&self, py: Python<'_>) -> PyResult<Option<PyStarlinkManifest>> {
        py.detach(|| self.inner.previous_manifest())
            .map(|opt| opt.map(|inner| PyStarlinkManifest { inner }))
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Downloads a satellite's current ephemeris file into the cache.
    ///
    /// The manifest names one file per satellite; if that file is already
    /// cached no request is made. After a download every other cached file
    /// for the same NORAD ID is deleted.
    ///
    /// Args:
    ///     norad_cat_id (int): NORAD catalog number.
    ///
    /// Returns:
    ///     str: Path of the cached file.
    ///
    /// Raises:
    ///     BraheError: If the ID is not listed, the download fails, or the body is not a valid ITC file.
    fn download_ephemeris(&self, py: Python<'_>, norad_cat_id: u32) -> PyResult<String> {
        py.detach(|| self.inner.download_ephemeris(norad_cat_id))
            .map(|p| p.to_string_lossy().into_owned())
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Downloads (if needed) and parses a satellite's current ephemeris.
    ///
    /// Args:
    ///     norad_cat_id (int): NORAD catalog number.
    ///
    /// Returns:
    ///     ITC: The parsed message in SI units.
    ///
    /// Raises:
    ///     BraheError: See ``download_ephemeris``.
    fn get_ephemeris(&self, py: Python<'_>, norad_cat_id: u32) -> PyResult<PyITC> {
        py.detach(|| self.inner.get_ephemeris(norad_cat_id))
            .map(|inner| PyITC { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Downloads (if needed) a satellite's ephemeris as an ``OrbitTrajectory``
    /// in the file's state frame with covariance rotated from RTN using the
    /// block-diagonal (inertial) convention by default.
    ///
    /// Args:
    ///     norad_cat_id (int): NORAD catalog number.
    ///     covariance_variant (OrbitRelativeFrameVariant, optional): ``INERTIAL`` (default, block-diagonal) or ``ROTATING`` RTN rotation.
    ///
    /// Returns:
    ///     OrbitTrajectory: Six-dimensional Cartesian trajectory.
    ///
    /// Raises:
    ///     BraheError: See ``download_ephemeris``.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.StarlinkClient().get_trajectory(100001)
    ///     ```
    #[pyo3(signature = (norad_cat_id, covariance_variant=None))]
    fn get_trajectory(
        &self,
        py: Python<'_>,
        norad_cat_id: u32,
        covariance_variant: Option<PyOrbitRelativeFrameVariant>,
    ) -> PyResult<PyOrbitalTrajectory> {
        let variant = covariance_variant
            .map(|v| v.variant)
            .unwrap_or(frames::OrbitRelativeFrameVariant::Inertial);
        py.detach(|| {
            self.inner
                .get_trajectory_with_covariance_variant(norad_cat_id, variant)
        })
        .map(|trajectory| PyOrbitalTrajectory { trajectory })
        .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Like ``get_trajectory`` with an explicit RTN covariance convention.
    ///
    /// Args:
    ///     norad_cat_id (int): NORAD catalog number.
    ///     variant (OrbitRelativeFrameVariant): ``INERTIAL`` (block-diagonal) or ``ROTATING`` RTN rotation.
    ///
    /// Returns:
    ///     OrbitTrajectory: Six-dimensional Cartesian trajectory.
    ///
    /// Raises:
    ///     BraheError: See ``download_ephemeris``.
    fn get_trajectory_with_covariance_variant(
        &self,
        py: Python<'_>,
        norad_cat_id: u32,
        variant: PyOrbitRelativeFrameVariant,
    ) -> PyResult<PyOrbitalTrajectory> {
        py.detach(|| {
            self.inner
                .get_trajectory_with_covariance_variant(norad_cat_id, variant.variant)
        })
        .map(|trajectory| PyOrbitalTrajectory { trajectory })
        .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Downloads (if needed) a satellite's ephemeris and copies it to ``destination``.
    ///
    /// An existing directory, or a path whose last component has no
    /// extension, is treated as a directory (created if missing) and the
    /// original file name is kept; a path with an extension is the file name.
    /// Saved copies are outside the cache and never evicted.
    ///
    /// Args:
    ///     norad_cat_id (int): NORAD catalog number.
    ///     destination (str): Directory or file path.
    ///
    /// Returns:
    ///     str: Path written.
    ///
    /// Raises:
    ///     BraheError: On download or filesystem failure.
    fn save_ephemeris(
        &self,
        py: Python<'_>,
        norad_cat_id: u32,
        destination: &str,
    ) -> PyResult<String> {
        py.detach(|| self.inner.save_ephemeris(norad_cat_id, destination))
            .map(|p| p.to_string_lossy().into_owned())
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Downloads every file the manifest lists that is not already cached,
    /// using up to ``concurrency`` threads that share the rate limiter.
    ///
    /// The full manifest is about 11,100 files and about 22 GB, so a first
    /// run against the public mirror is a long transfer. Stops at the first
    /// failure and returns it. Files already cached are not re-fetched. No
    /// files are pruned; superseded files for a satellite whose file is
    /// downloaded are still evicted. Call ``prune_cache`` to remove files the
    /// manifest no longer lists.
    ///
    /// Args:
    ///     concurrency (int, optional): Worker threads, at least 1. Default: 8.
    ///
    /// Returns:
    ///     list[str]: Cache path of one file per listed NORAD ID, in manifest order.
    ///
    /// Raises:
    ///     BraheError: The first failure, or ``concurrency == 0``.
    #[pyo3(signature = (concurrency=8))]
    fn download_all(&self, py: Python<'_>, concurrency: usize) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.download_all(concurrency))
            .map(|paths| {
                paths
                    .into_iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect()
            })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Runs ``download_all`` and copies every file into ``destination``.
    ///
    /// Args:
    ///     destination (str): Directory, created if missing.
    ///     concurrency (int, optional): Worker threads, at least 1. Default: 8.
    ///
    /// Returns:
    ///     list[str]: Paths written, in manifest order.
    ///
    /// Raises:
    ///     BraheError: On download failure or if ``destination`` is an existing file.
    #[pyo3(signature = (destination, concurrency=8))]
    fn save_all(
        &self,
        py: Python<'_>,
        destination: &str,
        concurrency: usize,
    ) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.save_all(destination, concurrency))
            .map(|paths| {
                paths
                    .into_iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect()
            })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Deletes cached ephemeris files the current manifest no longer lists.
    ///
    /// Returns:
    ///     int: Number of files deleted.
    ///
    /// Raises:
    ///     BraheError: If the manifest cannot be obtained or a file cannot be removed.
    fn prune_cache(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.prune_cache())
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Cached ephemeris files, sorted by name.
    ///
    /// Returns:
    ///     list[str]: Files whose names are Space-Track ephemeris names.
    ///
    /// Raises:
    ///     BraheError: If the cache directory cannot be read.
    fn cached_files(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.cached_files())
            .map(|paths| {
                paths
                    .into_iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect()
            })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }
}
