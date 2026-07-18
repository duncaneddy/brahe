// Python bindings for the star catalog datasets module.

use brahe::datasets::star_catalog;
use brahe::datasets::star_catalog::StarRecord;

// ─── FK5 ────────────────────────────────────────────────────────────

/// A single record from the FK5 star catalog.
///
/// Positions and proper motions are J2000.0 (the FK5 system's native
/// equinox/equator); the legacy B1950 columns and formal-error columns of
/// the source file are not represented.
///
/// Attributes:
///     fk5_id (int): FK5 catalog running number
///     ra (float): Right ascension, J2000.0. Units: *deg*
///     dec (float): Declination, J2000.0. Units: *deg*
///     pm_ra (float): Proper motion in right ascension (mu_alpha* = mu_alpha cos(dec)), J2000.0. Units: *mas/yr*
///     pm_dec (float): Proper motion in declination, J2000.0. Units: *mas/yr*
///     epoch_ra_1900 (float | None): Mean epoch of right ascension observations, minus 1900. Units: *yr*
///     epoch_dec_1900 (float | None): Mean epoch of declination observations, minus 1900. Units: *yr*
///     vmag (float | None): Visual magnitude. Units: *mag*
///     vmag_flag (str | None): Visual magnitude quality/note flag
///     spectral_type (str | None): Spectral type
///     parallax (float | None): Trigonometric parallax. Units: *mas*
///     radial_velocity (float | None): Radial velocity. Units: *km/s*
///     hd_id (str | None): Henry Draper (HD) catalog identifier
///     dm_id (str | None): Durchmusterung (DM) catalog identifier
///     gc_id (str | None): Groombridge Catalogue (GC) identifier
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     fk5 = datasets.star_catalog.get_fk5()
///     record = fk5.get_by_id(699)
///     if record:
///         print(f"RA: {record.ra} deg, Dec: {record.dec} deg")
///     ```
#[pyclass(name = "FK5Record", module = "brahe._brahe", from_py_object)]
#[derive(Clone)]
struct PyFK5Record {
    inner: star_catalog::FK5Record,
}

#[pymethods]
impl PyFK5Record {
    #[getter]
    fn fk5_id(&self) -> u32 {
        self.inner.fk5_id
    }
    #[getter]
    fn ra(&self) -> f64 {
        self.inner.ra
    }
    #[getter]
    fn dec(&self) -> f64 {
        self.inner.dec
    }
    #[getter]
    fn pm_ra(&self) -> f64 {
        self.inner.pm_ra
    }
    #[getter]
    fn pm_dec(&self) -> f64 {
        self.inner.pm_dec
    }
    #[getter]
    fn epoch_ra_1900(&self) -> Option<f64> {
        self.inner.epoch_ra_1900
    }
    #[getter]
    fn epoch_dec_1900(&self) -> Option<f64> {
        self.inner.epoch_dec_1900
    }
    #[getter]
    fn vmag(&self) -> Option<f64> {
        self.inner.vmag
    }
    #[getter]
    fn vmag_flag(&self) -> Option<&str> {
        self.inner.vmag_flag.as_deref()
    }
    #[getter]
    fn spectral_type(&self) -> Option<&str> {
        self.inner.spectral_type.as_deref()
    }
    #[getter]
    fn parallax(&self) -> Option<f64> {
        self.inner.parallax
    }
    #[getter]
    fn radial_velocity(&self) -> Option<f64> {
        self.inner.radial_velocity
    }
    #[getter]
    fn hd_id(&self) -> Option<&str> {
        self.inner.hd_id.as_deref()
    }
    #[getter]
    fn dm_id(&self) -> Option<&str> {
        self.inner.dm_id.as_deref()
    }
    #[getter]
    fn gc_id(&self) -> Option<&str> {
        self.inner.gc_id.as_deref()
    }

    /// Catalog identifier string.
    ///
    /// Returns:
    ///     str: Catalog identifier, e.g. ``"FK5 1"``
    fn id(&self) -> String {
        self.inner.id()
    }

    /// Common or cross-catalog name, if available.
    ///
    /// Returns:
    ///     str | None: Cross-catalog name, e.g. ``"HD 358"``
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    /// Unit vector toward the star, evaluated at the record's reference epoch.
    ///
    /// Computed from `ra`/`dec` with a unit range, so no proper-motion
    /// propagation is applied.
    ///
    /// Returns:
    ///     numpy.ndarray: Cartesian unit vector `[x, y, z]` toward the star. Units: dimensionless
    fn unit_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let u = self.inner.unit_vector();
        vector_to_numpy!(py, u, 3, f64)
    }

    /// Right ascension and declination propagated to a target epoch using proper motion.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch to propagate the position to
    ///     angle_format (AngleFormat): Desired angle format (`RADIANS` or `DEGREES`) for the returned `(ra, dec)`
    ///
    /// Returns:
    ///     tuple[float, float]: Right ascension and declination at `epoch`. Units: (*angle*, *angle*)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2030, 1, 1, 0, 0, 0.0, 0.0, "UTC")
    ///     ra, dec = record.radec_at_epoch(epc, bh.AngleFormat.DEGREES)
    ///     ```
    fn radec_at_epoch(&self, epoch: &PyEpoch, angle_format: &PyAngleFormat) -> (f64, f64) {
        self.inner.radec_at_epoch(epoch.obj, angle_format.value)
    }

    fn __repr__(&self) -> String {
        format!(
            "FK5Record(fk5_id={}, ra={}, dec={})",
            self.inner.fk5_id, self.inner.ra, self.inner.dec
        )
    }
}

/// Container for FK5 star catalog records with lookup and filter methods.
///
/// Provides lookup by FK5 identifier, magnitude filtering, and cone-search
/// filtering. Filter methods return a new ``FK5Catalog`` instance.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     fk5 = datasets.star_catalog.get_fk5()
///     print(f"Loaded {len(fk5)} records")
///
///     bright = fk5.filter_by_magnitude(3.0)
///     print(f"Bright stars: {len(bright)}")
///
///     df = fk5.to_dataframe()
///     print(df.head())
///     ```
#[pyclass(name = "FK5Catalog", module = "brahe._brahe")]
struct PyFK5Catalog {
    inner: star_catalog::FK5Catalog,
}

#[pymethods]
impl PyFK5Catalog {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("FK5Catalog({} records)", self.inner.len())
    }

    /// Look up a record by FK5 catalog identifier.
    ///
    /// Args:
    ///     fk5_id (int): FK5 catalog running number
    ///
    /// Returns:
    ///     FK5Record | None: The matching record, or None if not found
    fn get_by_id(&self, fk5_id: u32) -> Option<PyFK5Record> {
        self.inner.get_by_id(fk5_id).map(|r| PyFK5Record {
            inner: r.clone(),
        })
    }

    /// Filter records to those with visual magnitude at or brighter than `max_mag`.
    ///
    /// Records with unknown magnitude are excluded. Recall that smaller
    /// (more negative) magnitudes are brighter, so this keeps `vmag <= max_mag`.
    ///
    /// Args:
    ///     max_mag (float): Faintest visual magnitude to include. Units: *mag*
    ///
    /// Returns:
    ///     FK5Catalog: New catalog containing matching records
    fn filter_by_magnitude(&self, max_mag: f64) -> PyFK5Catalog {
        PyFK5Catalog {
            inner: self.inner.filter_by_magnitude(max_mag),
        }
    }

    /// Filter records to those within an angular radius of a cone center.
    ///
    /// Args:
    ///     ra (float): Cone center right ascension. Units: *angle*
    ///     dec (float): Cone center declination. Units: *angle*
    ///     radius (float): Cone half-angle. Units: *angle*
    ///     angle_format (AngleFormat): Format for `ra`, `dec`, and `radius` (`RADIANS` or `DEGREES`)
    ///
    /// Returns:
    ///     FK5Catalog: New catalog containing matching records
    fn filter_by_cone(
        &self,
        ra: f64,
        dec: f64,
        radius: f64,
        angle_format: &PyAngleFormat,
    ) -> PyFK5Catalog {
        PyFK5Catalog {
            inner: self.inner.filter_by_cone(ra, dec, radius, angle_format.value),
        }
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// One row per record, one column per field. Missing optional values
    /// become nulls.
    ///
    /// Returns:
    ///     polars.DataFrame: DataFrame with all catalog fields as columns
    ///
    /// Example:
    ///     ```python
    ///     df = fk5.to_dataframe()
    ///     print(df.head())
    ///     ```
    fn to_dataframe(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import("polars")?;
        let records = self.inner.records();

        macro_rules! str_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field.as_deref())?;
                }
                list
            }};
        }

        macro_rules! f64_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field)?;
                }
                list
            }};
        }

        let fk5_id_list = PyList::empty(py);
        for r in records {
            fk5_id_list.append(r.fk5_id)?;
        }

        let data_dict = PyDict::new(py);
        data_dict.set_item("fk5_id", fk5_id_list)?;
        data_dict.set_item("ra", f64_list!(ra))?;
        data_dict.set_item("dec", f64_list!(dec))?;
        data_dict.set_item("pm_ra", f64_list!(pm_ra))?;
        data_dict.set_item("pm_dec", f64_list!(pm_dec))?;
        data_dict.set_item("epoch_ra_1900", f64_list!(epoch_ra_1900))?;
        data_dict.set_item("epoch_dec_1900", f64_list!(epoch_dec_1900))?;
        data_dict.set_item("vmag", f64_list!(vmag))?;
        data_dict.set_item("vmag_flag", str_list!(vmag_flag))?;
        data_dict.set_item("spectral_type", str_list!(spectral_type))?;
        data_dict.set_item("parallax", f64_list!(parallax))?;
        data_dict.set_item("radial_velocity", f64_list!(radial_velocity))?;
        data_dict.set_item("hd_id", str_list!(hd_id))?;
        data_dict.set_item("dm_id", str_list!(dm_id))?;
        data_dict.set_item("gc_id", str_list!(gc_id))?;

        let df = polars.call_method1("DataFrame", (data_dict,))?;
        Ok(df.unbind())
    }

    /// Get all records as a list.
    ///
    /// Returns:
    ///     list[FK5Record]: All records in the catalog
    fn records(&self) -> Vec<PyFK5Record> {
        self.inner
            .records()
            .iter()
            .map(|r| PyFK5Record {
                inner: r.clone(),
            })
            .collect()
    }
}

// ─── Hipparcos ──────────────────────────────────────────────────────

/// A single record from the Hipparcos star catalog.
///
/// Positions and proper motions are ICRS at epoch J1991.25. The Hipparcos
/// main catalog does not carry a radial velocity column, so no radial
/// velocity value is available for Hipparcos records.
///
/// Attributes:
///     hip_id (int): Hipparcos catalog identifier
///     vmag (float | None): Visual magnitude. Units: *mag*
///     var_flag (str | None): Magnitude uncertainty/variability flag
///     ra (float): Right ascension, ICRS, epoch J1991.25. Units: *deg*
///     dec (float): Declination, ICRS, epoch J1991.25. Units: *deg*
///     parallax (float | None): Trigonometric parallax. Units: *mas*
///     pm_ra (float | None): Proper motion in right ascension (mu_alpha* = mu_alpha cos(dec)), ICRS. Units: *mas/yr*
///     pm_dec (float | None): Proper motion in declination, ICRS. Units: *mas/yr*
///     e_ra (float | None): Standard error in right ascension. Units: *mas*
///     e_dec (float | None): Standard error in declination. Units: *mas*
///     e_parallax (float | None): Standard error in parallax. Units: *mas*
///     e_pm_ra (float | None): Standard error in right ascension proper motion. Units: *mas/yr*
///     e_pm_dec (float | None): Standard error in declination proper motion. Units: *mas/yr*
///     bt_mag (float | None): Mean Tycho BT magnitude. Units: *mag*
///     vt_mag (float | None): Mean Tycho VT magnitude. Units: *mag*
///     b_v (float | None): Johnson B-V colour. Units: *mag*
///     hp_mag (float | None): Hipparcos-system magnitude. Units: *mag*
///     hvar_type (str | None): Variability type flag
///     mult_flag (str | None): Double/multiple system flag
///     hd_id (int | None): Henry Draper (HD) catalog identifier
///     bd_id (str | None): Raw Bonner Durchmusterung (BD) identifier (see `name()` for the expanded form)
///     cod_id (str | None): Raw Cordoba Durchmusterung (CoD) identifier (see `name()` for the expanded form)
///     cpd_id (str | None): Raw Cape Photographic Durchmusterung (CPD) identifier (see `name()` for the expanded form)
///     spectral_type (str | None): Spectral type
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     hip = datasets.star_catalog.get_hipparcos()
///     sirius = hip.get_by_id(32349)
///     if sirius:
///         print(f"Vmag: {sirius.vmag}")
///     ```
#[pyclass(name = "HipparcosRecord", module = "brahe._brahe", from_py_object)]
#[derive(Clone)]
struct PyHipparcosRecord {
    inner: star_catalog::HipparcosRecord,
}

#[pymethods]
impl PyHipparcosRecord {
    #[getter]
    fn hip_id(&self) -> u32 {
        self.inner.hip_id
    }
    #[getter]
    fn vmag(&self) -> Option<f64> {
        self.inner.vmag
    }
    #[getter]
    fn var_flag(&self) -> Option<&str> {
        self.inner.var_flag.as_deref()
    }
    #[getter]
    fn ra(&self) -> f64 {
        self.inner.ra
    }
    #[getter]
    fn dec(&self) -> f64 {
        self.inner.dec
    }
    #[getter]
    fn parallax(&self) -> Option<f64> {
        self.inner.parallax
    }
    #[getter]
    fn pm_ra(&self) -> Option<f64> {
        self.inner.pm_ra
    }
    #[getter]
    fn pm_dec(&self) -> Option<f64> {
        self.inner.pm_dec
    }
    #[getter]
    fn e_ra(&self) -> Option<f64> {
        self.inner.e_ra
    }
    #[getter]
    fn e_dec(&self) -> Option<f64> {
        self.inner.e_dec
    }
    #[getter]
    fn e_parallax(&self) -> Option<f64> {
        self.inner.e_parallax
    }
    #[getter]
    fn e_pm_ra(&self) -> Option<f64> {
        self.inner.e_pm_ra
    }
    #[getter]
    fn e_pm_dec(&self) -> Option<f64> {
        self.inner.e_pm_dec
    }
    #[getter]
    fn bt_mag(&self) -> Option<f64> {
        self.inner.bt_mag
    }
    #[getter]
    fn vt_mag(&self) -> Option<f64> {
        self.inner.vt_mag
    }
    #[getter]
    fn b_v(&self) -> Option<f64> {
        self.inner.b_v
    }
    #[getter]
    fn hp_mag(&self) -> Option<f64> {
        self.inner.hp_mag
    }
    #[getter]
    fn hvar_type(&self) -> Option<&str> {
        self.inner.hvar_type.as_deref()
    }
    #[getter]
    fn mult_flag(&self) -> Option<&str> {
        self.inner.mult_flag.as_deref()
    }
    #[getter]
    fn hd_id(&self) -> Option<u32> {
        self.inner.hd_id
    }
    #[getter]
    fn bd_id(&self) -> Option<&str> {
        self.inner.bd_id.as_deref()
    }
    #[getter]
    fn cod_id(&self) -> Option<&str> {
        self.inner.cod_id.as_deref()
    }
    #[getter]
    fn cpd_id(&self) -> Option<&str> {
        self.inner.cpd_id.as_deref()
    }
    #[getter]
    fn spectral_type(&self) -> Option<&str> {
        self.inner.spectral_type.as_deref()
    }

    /// Catalog identifier string.
    ///
    /// Returns:
    ///     str: Catalog identifier, e.g. ``"HIP 1"``
    fn id(&self) -> String {
        self.inner.id()
    }

    /// Common or cross-catalog name, if available.
    ///
    /// Returns:
    ///     str | None: Cross-catalog name, e.g. ``"HD 224700"``
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    /// Unit vector toward the star, evaluated at the record's reference epoch.
    ///
    /// Computed from `ra`/`dec` with a unit range, so no proper-motion
    /// propagation is applied.
    ///
    /// Returns:
    ///     numpy.ndarray: Cartesian unit vector `[x, y, z]` toward the star. Units: dimensionless
    fn unit_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let u = self.inner.unit_vector();
        vector_to_numpy!(py, u, 3, f64)
    }

    /// Right ascension and declination propagated to a target epoch using proper motion.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch to propagate the position to
    ///     angle_format (AngleFormat): Desired angle format (`RADIANS` or `DEGREES`) for the returned `(ra, dec)`
    ///
    /// Returns:
    ///     tuple[float, float]: Right ascension and declination at `epoch`. Units: (*angle*, *angle*)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2030, 1, 1, 0, 0, 0.0, 0.0, "UTC")
    ///     ra, dec = record.radec_at_epoch(epc, bh.AngleFormat.DEGREES)
    ///     ```
    fn radec_at_epoch(&self, epoch: &PyEpoch, angle_format: &PyAngleFormat) -> (f64, f64) {
        self.inner.radec_at_epoch(epoch.obj, angle_format.value)
    }

    fn __repr__(&self) -> String {
        format!(
            "HipparcosRecord(hip_id={}, ra={}, dec={})",
            self.inner.hip_id, self.inner.ra, self.inner.dec
        )
    }
}

/// Container for Hipparcos star catalog records with lookup and filter methods.
///
/// Provides lookup by Hipparcos identifier, magnitude filtering, and
/// cone-search filtering. Filter methods return a new ``HipparcosCatalog``
/// instance.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     hip = datasets.star_catalog.get_hipparcos()
///     print(f"Loaded {len(hip)} records")
///
///     bright = hip.filter_by_magnitude(5.0)
///     print(f"Bright stars: {len(bright)}")
///
///     df = hip.to_dataframe()
///     print(df.head())
///     ```
#[pyclass(name = "HipparcosCatalog", module = "brahe._brahe")]
struct PyHipparcosCatalog {
    inner: star_catalog::HipparcosCatalog,
}

#[pymethods]
impl PyHipparcosCatalog {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("HipparcosCatalog({} records)", self.inner.len())
    }

    /// Look up a record by Hipparcos catalog identifier.
    ///
    /// Args:
    ///     hip_id (int): Hipparcos catalog identifier
    ///
    /// Returns:
    ///     HipparcosRecord | None: The matching record, or None if not found
    fn get_by_id(&self, hip_id: u32) -> Option<PyHipparcosRecord> {
        self.inner.get_by_id(hip_id).map(|r| PyHipparcosRecord {
            inner: r.clone(),
        })
    }

    /// Filter records to those with visual magnitude at or brighter than `max_mag`.
    ///
    /// Records with unknown magnitude are excluded. Recall that smaller
    /// (more negative) magnitudes are brighter, so this keeps `vmag <= max_mag`.
    ///
    /// Args:
    ///     max_mag (float): Faintest visual magnitude to include. Units: *mag*
    ///
    /// Returns:
    ///     HipparcosCatalog: New catalog containing matching records
    fn filter_by_magnitude(&self, max_mag: f64) -> PyHipparcosCatalog {
        PyHipparcosCatalog {
            inner: self.inner.filter_by_magnitude(max_mag),
        }
    }

    /// Filter records to those within an angular radius of a cone center.
    ///
    /// Args:
    ///     ra (float): Cone center right ascension. Units: *angle*
    ///     dec (float): Cone center declination. Units: *angle*
    ///     radius (float): Cone half-angle. Units: *angle*
    ///     angle_format (AngleFormat): Format for `ra`, `dec`, and `radius` (`RADIANS` or `DEGREES`)
    ///
    /// Returns:
    ///     HipparcosCatalog: New catalog containing matching records
    fn filter_by_cone(
        &self,
        ra: f64,
        dec: f64,
        radius: f64,
        angle_format: &PyAngleFormat,
    ) -> PyHipparcosCatalog {
        PyHipparcosCatalog {
            inner: self.inner.filter_by_cone(ra, dec, radius, angle_format.value),
        }
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// One row per record, one column per field. Missing optional values
    /// become nulls.
    ///
    /// Returns:
    ///     polars.DataFrame: DataFrame with all catalog fields as columns
    ///
    /// Example:
    ///     ```python
    ///     df = hip.to_dataframe()
    ///     print(df.head())
    ///     ```
    fn to_dataframe(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import("polars")?;
        let records = self.inner.records();

        macro_rules! str_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field.as_deref())?;
                }
                list
            }};
        }

        macro_rules! f64_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field)?;
                }
                list
            }};
        }

        // Same shape as f64_list!, named separately for Option<u32> columns
        // so the macro name doesn't misdescribe the column's polars dtype.
        macro_rules! opt_u32_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field)?;
                }
                list
            }};
        }

        let hip_id_list = PyList::empty(py);
        for r in records {
            hip_id_list.append(r.hip_id)?;
        }

        let data_dict = PyDict::new(py);
        data_dict.set_item("hip_id", hip_id_list)?;
        data_dict.set_item("vmag", f64_list!(vmag))?;
        data_dict.set_item("var_flag", str_list!(var_flag))?;
        data_dict.set_item("ra", f64_list!(ra))?;
        data_dict.set_item("dec", f64_list!(dec))?;
        data_dict.set_item("parallax", f64_list!(parallax))?;
        data_dict.set_item("pm_ra", f64_list!(pm_ra))?;
        data_dict.set_item("pm_dec", f64_list!(pm_dec))?;
        data_dict.set_item("e_ra", f64_list!(e_ra))?;
        data_dict.set_item("e_dec", f64_list!(e_dec))?;
        data_dict.set_item("e_parallax", f64_list!(e_parallax))?;
        data_dict.set_item("e_pm_ra", f64_list!(e_pm_ra))?;
        data_dict.set_item("e_pm_dec", f64_list!(e_pm_dec))?;
        data_dict.set_item("bt_mag", f64_list!(bt_mag))?;
        data_dict.set_item("vt_mag", f64_list!(vt_mag))?;
        data_dict.set_item("b_v", f64_list!(b_v))?;
        data_dict.set_item("hp_mag", f64_list!(hp_mag))?;
        data_dict.set_item("hvar_type", str_list!(hvar_type))?;
        data_dict.set_item("mult_flag", str_list!(mult_flag))?;
        data_dict.set_item("hd_id", opt_u32_list!(hd_id))?;
        data_dict.set_item("bd_id", str_list!(bd_id))?;
        data_dict.set_item("cod_id", str_list!(cod_id))?;
        data_dict.set_item("cpd_id", str_list!(cpd_id))?;
        data_dict.set_item("spectral_type", str_list!(spectral_type))?;

        let df = polars.call_method1("DataFrame", (data_dict,))?;
        Ok(df.unbind())
    }

    /// Get all records as a list.
    ///
    /// Returns:
    ///     list[HipparcosRecord]: All records in the catalog
    fn records(&self) -> Vec<PyHipparcosRecord> {
        self.inner
            .records()
            .iter()
            .map(|r| PyHipparcosRecord {
                inner: r.clone(),
            })
            .collect()
    }
}

// ─── Tycho-2 ────────────────────────────────────────────────────────

/// A single record from the Tycho-2 star catalog.
///
/// A small fraction of entries (`pflag == "X"`) have no mean astrometric
/// solution: their `ra`/`dec`/`pm_ra`/`pm_dec`/`epoch_ra`/`epoch_dec`
/// fields are all ``None``. `id()`/`name()`/`unit_vector()`/
/// `radec_at_epoch()` fall back to the always-present observed position
/// (`ra_observed`/`dec_observed`, epoch ~1991.5) for such records.
///
/// Tycho-2 does not carry a parallax or radial velocity column.
///
/// Attributes:
///     tyc1 (int): Tycho-2 identifier, first component (GSC region number)
///     tyc2 (int): Tycho-2 identifier, second component (running number within region)
///     tyc3 (int): Tycho-2 identifier, third component (component number, for double/multiple entries)
///     pflag (str | None): Mean position flag: None/blank for a normal entry, "P" if the mean position was obtained from the photocenter, "X" if no mean position could be computed
///     ra (float | None): Mean right ascension, ICRS. None when `pflag == "X"`. Units: *deg*
///     dec (float | None): Mean declination, ICRS. None when `pflag == "X"`. Units: *deg*
///     pm_ra (float | None): Proper motion in right ascension (mu_alpha* = mu_alpha cos(dec)). Units: *mas/yr*
///     pm_dec (float | None): Proper motion in declination. Units: *mas/yr*
///     epoch_ra (float | None): Mean epoch of the right ascension. Units: *yr*
///     epoch_dec (float | None): Mean epoch of the declination. Units: *yr*
///     bt_mag (float | None): Tycho-2 BT (blue) magnitude. Units: *mag*
///     vt_mag (float | None): Tycho-2 VT (visual) magnitude. Units: *mag*
///     vmag (float | None): Johnson V-band approximation. Units: *mag*
///     tycho1_flag (str | None): Set ("T") if this entry also has a Tycho-1 record
///     hip_id (int | None): Hipparcos catalog identifier, if this star is also in Hipparcos
///     ra_observed (float): Observed right ascension, epoch ~1991.5. Always present. Units: *deg*
///     dec_observed (float): Observed declination, epoch ~1991.5. Always present. Units: *deg*
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     tyc = datasets.star_catalog.get_tycho2()
///     record = tyc.get_by_id(1, 8, 1)
///     ```
#[pyclass(name = "Tycho2Record", module = "brahe._brahe", from_py_object)]
#[derive(Clone)]
struct PyTycho2Record {
    inner: star_catalog::Tycho2Record,
}

#[pymethods]
impl PyTycho2Record {
    #[getter]
    fn tyc1(&self) -> u16 {
        self.inner.tyc1
    }
    #[getter]
    fn tyc2(&self) -> u16 {
        self.inner.tyc2
    }
    #[getter]
    fn tyc3(&self) -> u8 {
        self.inner.tyc3
    }
    #[getter]
    fn pflag(&self) -> Option<&str> {
        self.inner.pflag.as_deref()
    }
    #[getter]
    fn ra(&self) -> Option<f64> {
        self.inner.ra
    }
    #[getter]
    fn dec(&self) -> Option<f64> {
        self.inner.dec
    }
    #[getter]
    fn pm_ra(&self) -> Option<f64> {
        self.inner.pm_ra
    }
    #[getter]
    fn pm_dec(&self) -> Option<f64> {
        self.inner.pm_dec
    }
    #[getter]
    fn epoch_ra(&self) -> Option<f64> {
        self.inner.epoch_ra
    }
    #[getter]
    fn epoch_dec(&self) -> Option<f64> {
        self.inner.epoch_dec
    }
    #[getter]
    fn bt_mag(&self) -> Option<f64> {
        self.inner.bt_mag
    }
    #[getter]
    fn vt_mag(&self) -> Option<f64> {
        self.inner.vt_mag
    }
    #[getter]
    fn vmag(&self) -> Option<f64> {
        self.inner.vmag
    }
    #[getter]
    fn tycho1_flag(&self) -> Option<&str> {
        self.inner.tycho1_flag.as_deref()
    }
    #[getter]
    fn hip_id(&self) -> Option<u32> {
        self.inner.hip_id
    }
    #[getter]
    fn ra_observed(&self) -> f64 {
        self.inner.ra_observed
    }
    #[getter]
    fn dec_observed(&self) -> f64 {
        self.inner.dec_observed
    }

    /// Catalog identifier string.
    ///
    /// Returns:
    ///     str: Catalog identifier, e.g. ``"TYC 1-8-1"``
    fn id(&self) -> String {
        self.inner.id()
    }

    /// Common or cross-catalog name, if available.
    ///
    /// Returns:
    ///     str | None: Cross-catalog name, e.g. ``"HIP 416"``
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    /// Unit vector toward the star, evaluated at the record's reference epoch.
    ///
    /// Computed from the effective `ra`/`dec` (falling back to the observed
    /// position for `pflag == "X"` records) with a unit range, so no
    /// proper-motion propagation is applied.
    ///
    /// Returns:
    ///     numpy.ndarray: Cartesian unit vector `[x, y, z]` toward the star. Units: dimensionless
    fn unit_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let u = self.inner.unit_vector();
        vector_to_numpy!(py, u, 3, f64)
    }

    /// Right ascension and declination propagated to a target epoch using proper motion.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch to propagate the position to
    ///     angle_format (AngleFormat): Desired angle format (`RADIANS` or `DEGREES`) for the returned `(ra, dec)`
    ///
    /// Returns:
    ///     tuple[float, float]: Right ascension and declination at `epoch`. Units: (*angle*, *angle*)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2030, 1, 1, 0, 0, 0.0, 0.0, "UTC")
    ///     ra, dec = record.radec_at_epoch(epc, bh.AngleFormat.DEGREES)
    ///     ```
    fn radec_at_epoch(&self, epoch: &PyEpoch, angle_format: &PyAngleFormat) -> (f64, f64) {
        self.inner.radec_at_epoch(epoch.obj, angle_format.value)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tycho2Record(tyc1={}, tyc2={}, tyc3={})",
            self.inner.tyc1, self.inner.tyc2, self.inner.tyc3
        )
    }
}

/// Container for Tycho-2 star catalog records with lookup and filter methods.
///
/// Provides lookup by TYC identifier triple, magnitude filtering, and
/// cone-search filtering. Filter methods return a new ``Tycho2Catalog``
/// instance.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     tyc = datasets.star_catalog.get_tycho2()
///     print(f"Loaded {len(tyc)} records")
///
///     bright = tyc.filter_by_magnitude(8.0)
///     print(f"Bright stars: {len(bright)}")
///     ```
#[pyclass(name = "Tycho2Catalog", module = "brahe._brahe")]
struct PyTycho2Catalog {
    inner: star_catalog::Tycho2Catalog,
}

#[pymethods]
impl PyTycho2Catalog {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Tycho2Catalog({} records)", self.inner.len())
    }

    /// Look up a record by TYC identifier triple.
    ///
    /// Args:
    ///     tyc1 (int): GSC region number
    ///     tyc2 (int): Running number within region
    ///     tyc3 (int): Component number (for double/multiple entries)
    ///
    /// Returns:
    ///     Tycho2Record | None: The matching record, or None if not found
    fn get_by_id(&self, tyc1: u16, tyc2: u16, tyc3: u8) -> Option<PyTycho2Record> {
        self.inner
            .get_by_id(tyc1, tyc2, tyc3)
            .map(|r| PyTycho2Record {
                inner: r.clone(),
            })
    }

    /// Filter records to those with visual magnitude at or brighter than `max_mag`.
    ///
    /// Records with unknown magnitude are excluded. Recall that smaller
    /// (more negative) magnitudes are brighter, so this keeps `vmag <= max_mag`.
    ///
    /// Args:
    ///     max_mag (float): Faintest visual magnitude to include. Units: *mag*
    ///
    /// Returns:
    ///     Tycho2Catalog: New catalog containing matching records
    fn filter_by_magnitude(&self, max_mag: f64) -> PyTycho2Catalog {
        PyTycho2Catalog {
            inner: self.inner.filter_by_magnitude(max_mag),
        }
    }

    /// Filter records to those within an angular radius of a cone center.
    ///
    /// Args:
    ///     ra (float): Cone center right ascension. Units: *angle*
    ///     dec (float): Cone center declination. Units: *angle*
    ///     radius (float): Cone half-angle. Units: *angle*
    ///     angle_format (AngleFormat): Format for `ra`, `dec`, and `radius` (`RADIANS` or `DEGREES`)
    ///
    /// Returns:
    ///     Tycho2Catalog: New catalog containing matching records
    fn filter_by_cone(
        &self,
        ra: f64,
        dec: f64,
        radius: f64,
        angle_format: &PyAngleFormat,
    ) -> PyTycho2Catalog {
        PyTycho2Catalog {
            inner: self.inner.filter_by_cone(ra, dec, radius, angle_format.value),
        }
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// One row per record, one column per field. Missing optional values
    /// become nulls.
    ///
    /// Returns:
    ///     polars.DataFrame: DataFrame with all catalog fields as columns
    ///
    /// Example:
    ///     ```python
    ///     df = tyc.to_dataframe()
    ///     print(df.head())
    ///     ```
    fn to_dataframe(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import("polars")?;
        let records = self.inner.records();

        macro_rules! str_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field.as_deref())?;
                }
                list
            }};
        }

        macro_rules! f64_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field)?;
                }
                list
            }};
        }

        // Same shape as f64_list!, named separately for Option<u32> columns
        // so the macro name doesn't misdescribe the column's polars dtype.
        macro_rules! opt_u32_list {
            ($field:ident) => {{
                let list = PyList::empty(py);
                for r in records {
                    list.append(r.$field)?;
                }
                list
            }};
        }

        let tyc1_list = PyList::empty(py);
        let tyc2_list = PyList::empty(py);
        let tyc3_list = PyList::empty(py);
        for r in records {
            tyc1_list.append(r.tyc1)?;
            tyc2_list.append(r.tyc2)?;
            tyc3_list.append(r.tyc3)?;
        }

        let data_dict = PyDict::new(py);
        data_dict.set_item("tyc1", tyc1_list)?;
        data_dict.set_item("tyc2", tyc2_list)?;
        data_dict.set_item("tyc3", tyc3_list)?;
        data_dict.set_item("pflag", str_list!(pflag))?;
        data_dict.set_item("ra", f64_list!(ra))?;
        data_dict.set_item("dec", f64_list!(dec))?;
        data_dict.set_item("pm_ra", f64_list!(pm_ra))?;
        data_dict.set_item("pm_dec", f64_list!(pm_dec))?;
        data_dict.set_item("epoch_ra", f64_list!(epoch_ra))?;
        data_dict.set_item("epoch_dec", f64_list!(epoch_dec))?;
        data_dict.set_item("bt_mag", f64_list!(bt_mag))?;
        data_dict.set_item("vt_mag", f64_list!(vt_mag))?;
        data_dict.set_item("vmag", f64_list!(vmag))?;
        data_dict.set_item("tycho1_flag", str_list!(tycho1_flag))?;
        data_dict.set_item("hip_id", opt_u32_list!(hip_id))?;
        data_dict.set_item("ra_observed", f64_list!(ra_observed))?;
        data_dict.set_item("dec_observed", f64_list!(dec_observed))?;

        let df = polars.call_method1("DataFrame", (data_dict,))?;
        Ok(df.unbind())
    }

    /// Get all records as a list.
    ///
    /// Returns:
    ///     list[Tycho2Record]: All records in the catalog
    fn records(&self) -> Vec<PyTycho2Record> {
        self.inner
            .records()
            .iter()
            .map(|r| PyTycho2Record {
                inner: r.clone(),
            })
            .collect()
    }
}

// ─── Standalone functions ──────────────────────────────────────────

/// Download and parse the FK5 star catalog.
///
/// Fetches the fixed-width FK5 catalog text file with file-based caching.
/// FK5 is a fixed, published catalog, so by default the cached copy never
/// goes stale.
///
/// Args:
///     cache_max_age (float, optional): Maximum cache age in seconds.
///         Defaults to None, meaning the cached copy never goes stale.
///         Pass 0 to force a fresh download.
///
/// Returns:
///     FK5Catalog: Parsed FK5 catalog container
///
/// Raises:
///     BraheError: If download or parsing fails.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     fk5 = datasets.star_catalog.get_fk5()
///     print(f"Loaded {len(fk5)} records")
///     ```
#[pyfunction]
#[pyo3(name = "star_catalog_get_fk5", signature = (cache_max_age=None))]
fn py_star_catalog_get_fk5(py: Python<'_>, cache_max_age: Option<f64>) -> PyResult<PyFK5Catalog> {
    // Release the GIL during the (potentially slow) download/parse so other
    // Python threads can keep running.
    let catalog = py.detach(|| star_catalog::get_fk5_catalog(cache_max_age))?;
    Ok(PyFK5Catalog { inner: catalog })
}

/// Download and parse the Hipparcos star catalog.
///
/// Fetches the pipe-delimited Hipparcos catalog text file with file-based
/// caching. Hipparcos is a fixed, published catalog, so by default the
/// cached copy never goes stale.
///
/// Args:
///     cache_max_age (float, optional): Maximum cache age in seconds.
///         Defaults to None, meaning the cached copy never goes stale.
///         Pass 0 to force a fresh download.
///
/// Returns:
///     HipparcosCatalog: Parsed Hipparcos catalog container
///
/// Raises:
///     BraheError: If download or parsing fails.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     hip = datasets.star_catalog.get_hipparcos()
///     print(f"Loaded {len(hip)} records")
///
///     sirius = hip.get_by_id(32349)
///     if sirius:
///         print(f"Sirius Vmag: {sirius.vmag}")
///     ```
#[pyfunction]
#[pyo3(name = "star_catalog_get_hipparcos", signature = (cache_max_age=None))]
fn py_star_catalog_get_hipparcos(
    py: Python<'_>,
    cache_max_age: Option<f64>,
) -> PyResult<PyHipparcosCatalog> {
    // Release the GIL during the (potentially slow) download/parse so other
    // Python threads can keep running.
    let catalog = py.detach(|| star_catalog::get_hipparcos_catalog(cache_max_age))?;
    Ok(PyHipparcosCatalog { inner: catalog })
}

/// Download and parse the Tycho-2 star catalog.
///
/// Fetches the pipe-delimited Tycho-2 catalog text file with file-based
/// caching. Tycho-2 is a fixed, published catalog, so by default the cached
/// copy never goes stale. The source file is large (~526 MB, ~2.54 million
/// records), so the first call may take some time.
///
/// Args:
///     cache_max_age (float, optional): Maximum cache age in seconds.
///         Defaults to None, meaning the cached copy never goes stale.
///         Pass 0 to force a fresh download.
///
/// Returns:
///     Tycho2Catalog: Parsed Tycho-2 catalog container
///
/// Raises:
///     BraheError: If download or parsing fails.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     tyc = datasets.star_catalog.get_tycho2()
///     print(f"Loaded {len(tyc)} records")
///     ```
#[pyfunction]
#[pyo3(name = "star_catalog_get_tycho2", signature = (cache_max_age=None))]
fn py_star_catalog_get_tycho2(
    py: Python<'_>,
    cache_max_age: Option<f64>,
) -> PyResult<PyTycho2Catalog> {
    // Release the GIL during the (potentially slow) download/parse so other
    // Python threads can keep running.
    let catalog = py.detach(|| star_catalog::get_tycho2_catalog(cache_max_age))?;
    Ok(PyTycho2Catalog { inner: catalog })
}

// Functions and classes are registered in mod.rs via add_function()/add_class() calls
