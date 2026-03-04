// Python bindings for the GCAT datasets module.

use crate::datasets::gcat;

// ─── Record wrappers ───────────────────────────────────────────────

/// A single record from the GCAT SATCAT catalog.
///
/// Contains physical, orbital, and administrative metadata for an artificial
/// space object. All 41+ columns from the ``satcat.tsv`` file are represented.
///
/// Attributes:
///     jcat (str): JCAT catalog identifier (primary key)
///     satcat (str | None): NORAD SATCAT number
///     launch_tag (str | None): Launch tag identifier
///     piece (str | None): Piece designation
///     object_type (str | None): Object type code
///     name (str | None): Current object name
///     pl_name (str | None): Payload name
///     ldate (str | None): Launch date
///     parent (str | None): Parent object identifier
///     sdate (str | None): Separation/deployment date
///     primary (str | None): Primary body orbited
///     ddate (str | None): Decay/deorbit date
///     status (str | None): Current status code
///     dest (str | None): Destination/orbit description
///     owner (str | None): Owner/operator organization
///     state (str | None): Responsible state/country
///     manufacturer (str | None): Manufacturer organization
///     bus (str | None): Spacecraft bus type
///     motor (str | None): Motor/propulsion type
///     mass (float | None): Launch mass in kg
///     mass_flag (str | None): Mass quality flag
///     dry_mass (float | None): Dry mass in kg
///     dry_flag (str | None): Dry mass quality flag
///     tot_mass (float | None): Total mass in kg
///     tot_flag (str | None): Total mass quality flag
///     length (float | None): Length in meters
///     length_flag (str | None): Length quality flag
///     diameter (float | None): Diameter in meters
///     diameter_flag (str | None): Diameter quality flag
///     span (float | None): Span in meters
///     span_flag (str | None): Span quality flag
///     shape (str | None): Shape description
///     odate (str | None): Operational orbit epoch date
///     perigee (float | None): Perigee altitude in km
///     perigee_flag (str | None): Perigee quality flag
///     apogee (float | None): Apogee altitude in km
///     apogee_flag (str | None): Apogee quality flag
///     inc (float | None): Orbital inclination in degrees
///     inc_flag (str | None): Inclination quality flag
///     op_orbit (str | None): Operational orbit class
///     oqual (str | None): Orbit quality code
///     alt_names (str | None): Alternative names
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     satcat = datasets.gcat.get_satcat()
///     record = satcat.get_by_satcat("25544")
///     if record:
///         print(f"Name: {record.name}")
///         print(f"Perigee: {record.perigee} km")
///     ```
#[pyclass(name = "GCATSatcatRecord", from_py_object)]
#[derive(Clone)]
struct PyGCATSatcatRecord {
    inner: gcat::GCATSatcatRecord,
}

#[pymethods]
impl PyGCATSatcatRecord {
    #[getter]
    fn jcat(&self) -> &str {
        &self.inner.jcat
    }
    #[getter]
    fn satcat(&self) -> Option<&str> {
        self.inner.satcat.as_deref()
    }
    #[getter]
    fn launch_tag(&self) -> Option<&str> {
        self.inner.launch_tag.as_deref()
    }
    #[getter]
    fn piece(&self) -> Option<&str> {
        self.inner.piece.as_deref()
    }
    #[getter]
    fn object_type(&self) -> Option<&str> {
        self.inner.object_type.as_deref()
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }
    #[getter]
    fn pl_name(&self) -> Option<&str> {
        self.inner.pl_name.as_deref()
    }
    #[getter]
    fn ldate(&self) -> Option<&str> {
        self.inner.ldate.as_deref()
    }
    #[getter]
    fn parent(&self) -> Option<&str> {
        self.inner.parent.as_deref()
    }
    #[getter]
    fn sdate(&self) -> Option<&str> {
        self.inner.sdate.as_deref()
    }
    #[getter]
    fn primary(&self) -> Option<&str> {
        self.inner.primary.as_deref()
    }
    #[getter]
    fn ddate(&self) -> Option<&str> {
        self.inner.ddate.as_deref()
    }
    #[getter]
    fn status(&self) -> Option<&str> {
        self.inner.status.as_deref()
    }
    #[getter]
    fn dest(&self) -> Option<&str> {
        self.inner.dest.as_deref()
    }
    #[getter]
    fn owner(&self) -> Option<&str> {
        self.inner.owner.as_deref()
    }
    #[getter]
    fn state(&self) -> Option<&str> {
        self.inner.state.as_deref()
    }
    #[getter]
    fn manufacturer(&self) -> Option<&str> {
        self.inner.manufacturer.as_deref()
    }
    #[getter]
    fn bus(&self) -> Option<&str> {
        self.inner.bus.as_deref()
    }
    #[getter]
    fn motor(&self) -> Option<&str> {
        self.inner.motor.as_deref()
    }
    #[getter]
    fn mass(&self) -> Option<f64> {
        self.inner.mass
    }
    #[getter]
    fn mass_flag(&self) -> Option<&str> {
        self.inner.mass_flag.as_deref()
    }
    #[getter]
    fn dry_mass(&self) -> Option<f64> {
        self.inner.dry_mass
    }
    #[getter]
    fn dry_flag(&self) -> Option<&str> {
        self.inner.dry_flag.as_deref()
    }
    #[getter]
    fn tot_mass(&self) -> Option<f64> {
        self.inner.tot_mass
    }
    #[getter]
    fn tot_flag(&self) -> Option<&str> {
        self.inner.tot_flag.as_deref()
    }
    #[getter]
    fn length(&self) -> Option<f64> {
        self.inner.length
    }
    #[getter]
    fn length_flag(&self) -> Option<&str> {
        self.inner.length_flag.as_deref()
    }
    #[getter]
    fn diameter(&self) -> Option<f64> {
        self.inner.diameter
    }
    #[getter]
    fn diameter_flag(&self) -> Option<&str> {
        self.inner.diameter_flag.as_deref()
    }
    #[getter]
    fn span(&self) -> Option<f64> {
        self.inner.span
    }
    #[getter]
    fn span_flag(&self) -> Option<&str> {
        self.inner.span_flag.as_deref()
    }
    #[getter]
    fn shape(&self) -> Option<&str> {
        self.inner.shape.as_deref()
    }
    #[getter]
    fn odate(&self) -> Option<&str> {
        self.inner.odate.as_deref()
    }
    #[getter]
    fn perigee(&self) -> Option<f64> {
        self.inner.perigee
    }
    #[getter]
    fn perigee_flag(&self) -> Option<&str> {
        self.inner.perigee_flag.as_deref()
    }
    #[getter]
    fn apogee(&self) -> Option<f64> {
        self.inner.apogee
    }
    #[getter]
    fn apogee_flag(&self) -> Option<&str> {
        self.inner.apogee_flag.as_deref()
    }
    #[getter]
    fn inc(&self) -> Option<f64> {
        self.inner.inc
    }
    #[getter]
    fn inc_flag(&self) -> Option<&str> {
        self.inner.inc_flag.as_deref()
    }
    #[getter]
    fn op_orbit(&self) -> Option<&str> {
        self.inner.op_orbit.as_deref()
    }
    #[getter]
    fn oqual(&self) -> Option<&str> {
        self.inner.oqual.as_deref()
    }
    #[getter]
    fn alt_names(&self) -> Option<&str> {
        self.inner.alt_names.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "GCATSatcatRecord(jcat='{}', satcat={}, name={})",
            self.inner.jcat,
            self.inner
                .satcat
                .as_ref()
                .map(|s| format!("'{}'", s))
                .unwrap_or_else(|| "None".to_string()),
            self.inner
                .name
                .as_ref()
                .map(|s| format!("'{}'", s))
                .unwrap_or_else(|| "None".to_string()),
        )
    }
}

/// A single record from the GCAT PSATCAT catalog.
///
/// Contains payload-specific metadata including mission details, UN registry
/// information, and disposal orbit parameters.
///
/// Attributes:
///     jcat (str): JCAT catalog identifier (primary key)
///     piece (str | None): Piece designation
///     name (str | None): Payload name
///     ldate (str | None): Launch date
///     tlast (str | None): Last contact date
///     top (str | None): Operational start date
///     tdate (str | None): End of operations date
///     tf (str | None): Date quality flag
///     program (str | None): Program/constellation name
///     plane (str | None): Orbital plane identifier
///     att (str | None): Attitude control type
///     mvr (str | None): Maneuver capability
///     class_ (str | None): Mission class
///     category (str | None): Mission category
///     result (str | None): Mission result/outcome
///     control (str | None): Control authority
///     discipline (str | None): Mission discipline
///     un_state (str | None): UN registry state
///     un_reg (str | None): UN registry number
///     un_period (float | None): UN registered orbital period in minutes
///     un_perigee (float | None): UN registered perigee in km
///     un_apogee (float | None): UN registered apogee in km
///     un_inc (float | None): UN registered inclination in degrees
///     disp_epoch (str | None): Disposal orbit epoch
///     disp_peri (float | None): Disposal perigee in km
///     disp_apo (float | None): Disposal apogee in km
///     disp_inc (float | None): Disposal inclination in degrees
///     comment (str | None): Comments
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     psatcat = datasets.gcat.get_psatcat()
///     record = psatcat.get_by_jcat("S049652")
///     if record:
///         print(f"Program: {record.program}")
///         print(f"Category: {record.category}")
///     ```
#[pyclass(name = "GCATPsatcatRecord", from_py_object)]
#[derive(Clone)]
struct PyGCATPsatcatRecord {
    inner: gcat::GCATPsatcatRecord,
}

#[pymethods]
impl PyGCATPsatcatRecord {
    #[getter]
    fn jcat(&self) -> &str {
        &self.inner.jcat
    }
    #[getter]
    fn piece(&self) -> Option<&str> {
        self.inner.piece.as_deref()
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }
    #[getter]
    fn ldate(&self) -> Option<&str> {
        self.inner.ldate.as_deref()
    }
    #[getter]
    fn tlast(&self) -> Option<&str> {
        self.inner.tlast.as_deref()
    }
    #[getter]
    fn top(&self) -> Option<&str> {
        self.inner.top.as_deref()
    }
    #[getter]
    fn tdate(&self) -> Option<&str> {
        self.inner.tdate.as_deref()
    }
    #[getter]
    fn tf(&self) -> Option<&str> {
        self.inner.tf.as_deref()
    }
    #[getter]
    fn program(&self) -> Option<&str> {
        self.inner.program.as_deref()
    }
    #[getter]
    fn plane(&self) -> Option<&str> {
        self.inner.plane.as_deref()
    }
    #[getter]
    fn att(&self) -> Option<&str> {
        self.inner.att.as_deref()
    }
    #[getter]
    fn mvr(&self) -> Option<&str> {
        self.inner.mvr.as_deref()
    }
    // "class" is a Python keyword, so use "class_" as the getter name
    #[getter(class_)]
    fn class(&self) -> Option<&str> {
        self.inner.class.as_deref()
    }
    #[getter]
    fn category(&self) -> Option<&str> {
        self.inner.category.as_deref()
    }
    #[getter]
    fn result(&self) -> Option<&str> {
        self.inner.result.as_deref()
    }
    #[getter]
    fn control(&self) -> Option<&str> {
        self.inner.control.as_deref()
    }
    #[getter]
    fn discipline(&self) -> Option<&str> {
        self.inner.discipline.as_deref()
    }
    #[getter]
    fn un_state(&self) -> Option<&str> {
        self.inner.un_state.as_deref()
    }
    #[getter]
    fn un_reg(&self) -> Option<&str> {
        self.inner.un_reg.as_deref()
    }
    #[getter]
    fn un_period(&self) -> Option<f64> {
        self.inner.un_period
    }
    #[getter]
    fn un_perigee(&self) -> Option<f64> {
        self.inner.un_perigee
    }
    #[getter]
    fn un_apogee(&self) -> Option<f64> {
        self.inner.un_apogee
    }
    #[getter]
    fn un_inc(&self) -> Option<f64> {
        self.inner.un_inc
    }
    #[getter]
    fn disp_epoch(&self) -> Option<&str> {
        self.inner.disp_epoch.as_deref()
    }
    #[getter]
    fn disp_peri(&self) -> Option<f64> {
        self.inner.disp_peri
    }
    #[getter]
    fn disp_apo(&self) -> Option<f64> {
        self.inner.disp_apo
    }
    #[getter]
    fn disp_inc(&self) -> Option<f64> {
        self.inner.disp_inc
    }
    #[getter]
    fn comment(&self) -> Option<&str> {
        self.inner.comment.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "GCATPsatcatRecord(jcat='{}', name={})",
            self.inner.jcat,
            self.inner
                .name
                .as_ref()
                .map(|s| format!("'{}'", s))
                .unwrap_or_else(|| "None".to_string()),
        )
    }
}

// ─── Catalog container wrappers ────────────────────────────────────

/// Container for GCAT SATCAT records with search and filter methods.
///
/// Provides lookup by JCAT/SATCAT number, name search, and various field-based
/// filters. All filter methods return a new ``GCATSatcat`` instance.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     satcat = datasets.gcat.get_satcat()
///     print(f"Loaded {len(satcat)} records")
///
///     # Search by name
///     iss_results = satcat.search_by_name("ISS")
///     print(f"Found {len(iss_results)} ISS-related objects")
///
///     # Filter by type and status
///     active_payloads = satcat.filter_by_type("P").filter_by_status("O")
///
///     # Convert to DataFrame
///     df = satcat.to_dataframe()
///     print(df.head())
///     ```
#[pyclass(name = "GCATSatcat")]
struct PyGCATSatcat {
    inner: gcat::GCATSatcat,
}

#[pymethods]
impl PyGCATSatcat {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("GCATSatcat({} records)", self.inner.len())
    }

    /// Look up a record by JCAT identifier.
    ///
    /// Args:
    ///     jcat (str): JCAT catalog identifier
    ///
    /// Returns:
    ///     GCATSatcatRecord | None: The matching record, or None if not found
    ///
    /// Example:
    ///     ```python
    ///     record = satcat.get_by_jcat("S049652")
    ///     ```
    fn get_by_jcat(&self, jcat: &str) -> Option<PyGCATSatcatRecord> {
        self.inner.get_by_jcat(jcat).map(|r| PyGCATSatcatRecord {
            inner: r.clone(),
        })
    }

    /// Look up a record by NORAD SATCAT number.
    ///
    /// Args:
    ///     satcat_num (str): NORAD SATCAT number (e.g. "25544" for ISS)
    ///
    /// Returns:
    ///     GCATSatcatRecord | None: The matching record, or None if not found
    ///
    /// Example:
    ///     ```python
    ///     iss = satcat.get_by_satcat("25544")
    ///     ```
    fn get_by_satcat(&self, satcat_num: &str) -> Option<PyGCATSatcatRecord> {
        self.inner
            .get_by_satcat(satcat_num)
            .map(|r| PyGCATSatcatRecord {
                inner: r.clone(),
            })
    }

    /// Search records by name (case-insensitive substring match).
    ///
    /// Args:
    ///     pattern (str): Search pattern (case-insensitive)
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     starlinks = satcat.search_by_name("starlink")
    ///     ```
    fn search_by_name(&self, pattern: &str) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.search_by_name(pattern),
        }
    }

    /// Filter records by object type (exact match).
    ///
    /// Args:
    ///     object_type (str): Object type code (e.g. "P" for payload, "R" for rocket body)
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     payloads = satcat.filter_by_type("P")
    ///     ```
    fn filter_by_type(&self, object_type: &str) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_type(object_type),
        }
    }

    /// Filter records by owner (exact match).
    ///
    /// Args:
    ///     owner (str): Owner/operator organization name
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     nasa_objects = satcat.filter_by_owner("NASA")
    ///     ```
    fn filter_by_owner(&self, owner: &str) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_owner(owner),
        }
    }

    /// Filter records by responsible state (exact match).
    ///
    /// Args:
    ///     state (str): State/country code (e.g. "US", "RU")
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     us_objects = satcat.filter_by_state("US")
    ///     ```
    fn filter_by_state(&self, state: &str) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_state(state),
        }
    }

    /// Filter records by status code (exact match).
    ///
    /// Args:
    ///     status (str): Status code (e.g. "O" for operational, "D" for decayed)
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     operational = satcat.filter_by_status("O")
    ///     ```
    fn filter_by_status(&self, status: &str) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_status(status),
        }
    }

    /// Filter records by perigee altitude range in km.
    ///
    /// Args:
    ///     min_km (float): Minimum perigee altitude in km
    ///     max_km (float): Maximum perigee altitude in km
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     leo = satcat.filter_by_perigee_range(200.0, 2000.0)
    ///     ```
    fn filter_by_perigee_range(&self, min_km: f64, max_km: f64) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_perigee_range(min_km, max_km),
        }
    }

    /// Filter records by apogee altitude range in km.
    ///
    /// Args:
    ///     min_km (float): Minimum apogee altitude in km
    ///     max_km (float): Maximum apogee altitude in km
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     leo = satcat.filter_by_apogee_range(200.0, 2000.0)
    ///     ```
    fn filter_by_apogee_range(&self, min_km: f64, max_km: f64) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_apogee_range(min_km, max_km),
        }
    }

    /// Filter records by inclination range in degrees.
    ///
    /// Args:
    ///     min_deg (float): Minimum inclination in degrees
    ///     max_deg (float): Maximum inclination in degrees
    ///
    /// Returns:
    ///     GCATSatcat: New catalog containing matching records
    ///
    /// Example:
    ///     ```python
    ///     polar = satcat.filter_by_inc_range(85.0, 100.0)
    ///     ```
    fn filter_by_inc_range(&self, min_deg: f64, max_deg: f64) -> PyGCATSatcat {
        PyGCATSatcat {
            inner: self.inner.filter_by_inc_range(min_deg, max_deg),
        }
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// All record fields become columns. String fields are ``Utf8`` type,
    /// numeric fields are ``Float64`` type. Missing values are ``None``.
    ///
    /// Returns:
    ///     polars.DataFrame: DataFrame with all catalog fields as columns
    ///
    /// Example:
    ///     ```python
    ///     df = satcat.to_dataframe()
    ///     print(df.head())
    ///     print(df.filter(pl.col("status") == "O").shape)
    ///     ```
    fn to_dataframe(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import("polars")?;

        // Build column data as Python lists
        let records = self.inner.records();

        // Helper macros to collect columns into Python lists
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

        // Build jcat column (non-optional)
        let jcat_list = PyList::empty(py);
        for r in records {
            jcat_list.append(&r.jcat)?;
        }

        // Build dict of column name -> list
        let data_dict = PyDict::new(py);
        data_dict.set_item("jcat", jcat_list)?;
        data_dict.set_item("satcat", str_list!(satcat))?;
        data_dict.set_item("launch_tag", str_list!(launch_tag))?;
        data_dict.set_item("piece", str_list!(piece))?;
        data_dict.set_item("object_type", str_list!(object_type))?;
        data_dict.set_item("name", str_list!(name))?;
        data_dict.set_item("pl_name", str_list!(pl_name))?;
        data_dict.set_item("ldate", str_list!(ldate))?;
        data_dict.set_item("parent", str_list!(parent))?;
        data_dict.set_item("sdate", str_list!(sdate))?;
        data_dict.set_item("primary", str_list!(primary))?;
        data_dict.set_item("ddate", str_list!(ddate))?;
        data_dict.set_item("status", str_list!(status))?;
        data_dict.set_item("dest", str_list!(dest))?;
        data_dict.set_item("owner", str_list!(owner))?;
        data_dict.set_item("state", str_list!(state))?;
        data_dict.set_item("manufacturer", str_list!(manufacturer))?;
        data_dict.set_item("bus", str_list!(bus))?;
        data_dict.set_item("motor", str_list!(motor))?;
        data_dict.set_item("mass", f64_list!(mass))?;
        data_dict.set_item("mass_flag", str_list!(mass_flag))?;
        data_dict.set_item("dry_mass", f64_list!(dry_mass))?;
        data_dict.set_item("dry_flag", str_list!(dry_flag))?;
        data_dict.set_item("tot_mass", f64_list!(tot_mass))?;
        data_dict.set_item("tot_flag", str_list!(tot_flag))?;
        data_dict.set_item("length", f64_list!(length))?;
        data_dict.set_item("length_flag", str_list!(length_flag))?;
        data_dict.set_item("diameter", f64_list!(diameter))?;
        data_dict.set_item("diameter_flag", str_list!(diameter_flag))?;
        data_dict.set_item("span", f64_list!(span))?;
        data_dict.set_item("span_flag", str_list!(span_flag))?;
        data_dict.set_item("shape", str_list!(shape))?;
        data_dict.set_item("odate", str_list!(odate))?;
        data_dict.set_item("perigee", f64_list!(perigee))?;
        data_dict.set_item("perigee_flag", str_list!(perigee_flag))?;
        data_dict.set_item("apogee", f64_list!(apogee))?;
        data_dict.set_item("apogee_flag", str_list!(apogee_flag))?;
        data_dict.set_item("inc", f64_list!(inc))?;
        data_dict.set_item("inc_flag", str_list!(inc_flag))?;
        data_dict.set_item("op_orbit", str_list!(op_orbit))?;
        data_dict.set_item("oqual", str_list!(oqual))?;
        data_dict.set_item("alt_names", str_list!(alt_names))?;

        let df = polars.call_method1("DataFrame", (data_dict,))?;
        Ok(df.unbind())
    }

    /// Get all records as a list.
    ///
    /// Returns:
    ///     list[GCATSatcatRecord]: All records in the catalog
    fn records(&self) -> Vec<PyGCATSatcatRecord> {
        self.inner
            .records()
            .iter()
            .map(|r| PyGCATSatcatRecord {
                inner: r.clone(),
            })
            .collect()
    }
}

/// Container for GCAT PSATCAT records with search and filter methods.
///
/// Provides lookup by JCAT, name search, and various field-based filters.
/// All filter methods return a new ``GCATPsatcat`` instance.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     psatcat = datasets.gcat.get_psatcat()
///     print(f"Loaded {len(psatcat)} records")
///
///     # Filter for active payloads
///     active = psatcat.filter_active()
///     print(f"Active payloads: {len(active)}")
///
///     # Convert to DataFrame
///     df = psatcat.to_dataframe()
///     ```
#[pyclass(name = "GCATPsatcat")]
struct PyGCATPsatcat {
    inner: gcat::GCATPsatcat,
}

#[pymethods]
impl PyGCATPsatcat {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("GCATPsatcat({} records)", self.inner.len())
    }

    /// Look up a record by JCAT identifier.
    ///
    /// Args:
    ///     jcat (str): JCAT catalog identifier
    ///
    /// Returns:
    ///     GCATPsatcatRecord | None: The matching record, or None if not found
    fn get_by_jcat(&self, jcat: &str) -> Option<PyGCATPsatcatRecord> {
        self.inner
            .get_by_jcat(jcat)
            .map(|r| PyGCATPsatcatRecord {
                inner: r.clone(),
            })
    }

    /// Search records by name (case-insensitive substring match).
    ///
    /// Args:
    ///     pattern (str): Search pattern (case-insensitive)
    ///
    /// Returns:
    ///     GCATPsatcat: New catalog containing matching records
    fn search_by_name(&self, pattern: &str) -> PyGCATPsatcat {
        PyGCATPsatcat {
            inner: self.inner.search_by_name(pattern),
        }
    }

    /// Filter records by mission category (exact match).
    ///
    /// Args:
    ///     category (str): Category code (e.g. "COM", "IMG", "TECH", "SCI", "NAV")
    ///
    /// Returns:
    ///     GCATPsatcat: New catalog containing matching records
    fn filter_by_category(&self, category: &str) -> PyGCATPsatcat {
        PyGCATPsatcat {
            inner: self.inner.filter_by_category(category),
        }
    }

    /// Filter records by mission class (exact match).
    ///
    /// Args:
    ///     class_ (str): Mission class code (e.g. "A", "B", "C", "D")
    ///
    /// Returns:
    ///     GCATPsatcat: New catalog containing matching records
    #[pyo3(name = "filter_by_class")]
    fn filter_by_class(&self, class_: &str) -> PyGCATPsatcat {
        PyGCATPsatcat {
            inner: self.inner.filter_by_class(class_),
        }
    }

    /// Filter records by mission result (exact match).
    ///
    /// Args:
    ///     result (str): Result code (e.g. "S" for success, "F" for failure)
    ///
    /// Returns:
    ///     GCATPsatcat: New catalog containing matching records
    fn filter_by_result(&self, result: &str) -> PyGCATPsatcat {
        PyGCATPsatcat {
            inner: self.inner.filter_by_result(result),
        }
    }

    /// Filter for active payloads (result is "S" and no end date or ``tdate="*"``).
    ///
    /// Returns:
    ///     GCATPsatcat: New catalog containing active records
    fn filter_active(&self) -> PyGCATPsatcat {
        PyGCATPsatcat {
            inner: self.inner.filter_active(),
        }
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// Returns:
    ///     polars.DataFrame: DataFrame with all catalog fields as columns
    ///
    /// Example:
    ///     ```python
    ///     df = psatcat.to_dataframe()
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

        let jcat_list = PyList::empty(py);
        for r in records {
            jcat_list.append(&r.jcat)?;
        }

        let data_dict = PyDict::new(py);
        data_dict.set_item("jcat", jcat_list)?;
        data_dict.set_item("piece", str_list!(piece))?;
        data_dict.set_item("name", str_list!(name))?;
        data_dict.set_item("ldate", str_list!(ldate))?;
        data_dict.set_item("tlast", str_list!(tlast))?;
        data_dict.set_item("top", str_list!(top))?;
        data_dict.set_item("tdate", str_list!(tdate))?;
        data_dict.set_item("tf", str_list!(tf))?;
        data_dict.set_item("program", str_list!(program))?;
        data_dict.set_item("plane", str_list!(plane))?;
        data_dict.set_item("att", str_list!(att))?;
        data_dict.set_item("mvr", str_list!(mvr))?;
        data_dict.set_item("class", str_list!(class))?;
        data_dict.set_item("category", str_list!(category))?;
        data_dict.set_item("result", str_list!(result))?;
        data_dict.set_item("control", str_list!(control))?;
        data_dict.set_item("discipline", str_list!(discipline))?;
        data_dict.set_item("un_state", str_list!(un_state))?;
        data_dict.set_item("un_reg", str_list!(un_reg))?;
        data_dict.set_item("un_period", f64_list!(un_period))?;
        data_dict.set_item("un_perigee", f64_list!(un_perigee))?;
        data_dict.set_item("un_apogee", f64_list!(un_apogee))?;
        data_dict.set_item("un_inc", f64_list!(un_inc))?;
        data_dict.set_item("disp_epoch", str_list!(disp_epoch))?;
        data_dict.set_item("disp_peri", f64_list!(disp_peri))?;
        data_dict.set_item("disp_apo", f64_list!(disp_apo))?;
        data_dict.set_item("disp_inc", f64_list!(disp_inc))?;
        data_dict.set_item("comment", str_list!(comment))?;

        let df = polars.call_method1("DataFrame", (data_dict,))?;
        Ok(df.unbind())
    }

    /// Get all records as a list.
    ///
    /// Returns:
    ///     list[GCATPsatcatRecord]: All records in the catalog
    fn records(&self) -> Vec<PyGCATPsatcatRecord> {
        self.inner
            .records()
            .iter()
            .map(|r| PyGCATPsatcatRecord {
                inner: r.clone(),
            })
            .collect()
    }
}

// ─── Standalone functions ──────────────────────────────────────────

/// Download and parse the GCAT SATCAT catalog.
///
/// Fetches the SATCAT TSV file from GCAT with file-based caching (default 24h).
/// Returns a ``GCATSatcat`` container with search and filter methods.
///
/// Args:
///     cache_max_age (float, optional): Maximum cache age in seconds.
///         Defaults to 86400 (24 hours). Pass 0 to force a fresh download.
///
/// Returns:
///     GCATSatcat: Parsed SATCAT catalog container
///
/// Raises:
///     BraheError: If download or parsing fails.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     # Default 24h cache
///     satcat = datasets.gcat.get_satcat()
///     print(f"Loaded {len(satcat)} records")
///
///     # Custom cache age (1 hour)
///     satcat = datasets.gcat.get_satcat(cache_max_age=3600)
///
///     # Look up ISS
///     iss = satcat.get_by_satcat("25544")
///     if iss:
///         print(f"ISS: {iss.name}")
///     ```
#[pyfunction]
#[pyo3(name = "gcat_get_satcat", signature = (cache_max_age=None))]
fn py_gcat_get_satcat(cache_max_age: Option<f64>) -> PyResult<PyGCATSatcat> {
    let catalog = gcat::get_satcat(cache_max_age)?;
    Ok(PyGCATSatcat { inner: catalog })
}

/// Download and parse the GCAT PSATCAT catalog.
///
/// Fetches the PSATCAT TSV file from GCAT with file-based caching (default 24h).
/// Returns a ``GCATPsatcat`` container with search and filter methods.
///
/// Args:
///     cache_max_age (float, optional): Maximum cache age in seconds.
///         Defaults to 86400 (24 hours). Pass 0 to force a fresh download.
///
/// Returns:
///     GCATPsatcat: Parsed PSATCAT catalog container
///
/// Raises:
///     BraheError: If download or parsing fails.
///
/// Example:
///     ```python
///     import brahe.datasets as datasets
///
///     psatcat = datasets.gcat.get_psatcat()
///     print(f"Loaded {len(psatcat)} records")
///
///     active = psatcat.filter_active()
///     print(f"Active payloads: {len(active)}")
///     ```
#[pyfunction]
#[pyo3(name = "gcat_get_psatcat", signature = (cache_max_age=None))]
fn py_gcat_get_psatcat(cache_max_age: Option<f64>) -> PyResult<PyGCATPsatcat> {
    let catalog = gcat::get_psatcat(cache_max_age)?;
    Ok(PyGCATPsatcat { inner: catalog })
}

// Functions and classes are registered in mod.rs via add_function()/add_class() calls
