// CCSDS Python bindings for OEM, OMM, and OPM message types.
//
// Design: scalar metadata exposed as #[getter] properties, collections via
// wrapper types with __len__, __getitem__ (negative indexing), and __iter__.

use crate::ccsds::common::CCSDSFormat;
use crate::ccsds::oem::{OEM as RustOEM, OEMSegment, OEMStateVector};
use crate::ccsds::omm::OMM as RustOMM;
use crate::ccsds::opm::{OPM as RustOPM, OPMManeuver};

// ─────────────────────────────────────────────
// OEM wrapper types
// ─────────────────────────────────────────────

/// Collection wrapper for OEM segments, supporting len/indexing/iteration.
///
/// Example:
///     ```python
///     segments = oem.segments
///     print(len(segments))        # number of segments
///     seg = segments[0]           # first segment
///     for seg in segments:        # iterate
///         print(seg.object_name)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMSegments")]
pub struct PyOEMSegments {
    inner: Vec<OEMSegment>,
}

#[pymethods]
impl PyOEMSegments {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<PyOEMSegment> {
        let len = self.inner.len() as isize;
        let actual = if index < 0 { len + index } else { index };
        if actual < 0 || actual >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "segment index {} out of range (have {})",
                index,
                self.inner.len()
            )));
        }
        Ok(PyOEMSegment {
            inner: self.inner[actual as usize].clone(),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOEMSegmentIterator>> {
        let py = slf.py();
        Py::new(
            py,
            PyOEMSegmentIterator {
                segments: slf.into(),
                index: 0,
            },
        )
    }

    fn __repr__(&self) -> String {
        format!("OEMSegments(len={})", self.inner.len())
    }
}

/// Iterator over OEM segments.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMSegmentIterator")]
pub struct PyOEMSegmentIterator {
    segments: Py<PyOEMSegments>,
    index: usize,
}

#[pymethods]
impl PyOEMSegmentIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<PyOEMSegment> {
        let segments = self.segments.borrow(py);
        if self.index >= segments.inner.len() {
            return None;
        }
        let seg = PyOEMSegment {
            inner: segments.inner[self.index].clone(),
        };
        self.index += 1;
        Some(seg)
    }
}

/// Wrapper for a single OEM segment with property access to metadata and states.
///
/// Example:
///     ```python
///     seg = oem.segments[0]
///     print(seg.object_name)      # str
///     print(seg.ref_frame)        # str
///     print(len(seg.states))      # number of state vectors
///     for sv in seg.states:
///         print(sv["position"])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMSegment")]
pub struct PyOEMSegment {
    inner: OEMSegment,
}

#[pymethods]
impl PyOEMSegment {
    /// Spacecraft name.
    ///
    /// Returns:
    ///     str: Object name
    #[getter]
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// International designator (e.g., "1996-062A").
    ///
    /// Returns:
    ///     str: Object ID
    #[getter]
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Central body name (e.g., "EARTH", "MARS BARYCENTER").
    ///
    /// Returns:
    ///     str: Center name
    #[getter]
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Reference frame name (e.g., "J2000", "GCRF").
    ///
    /// Returns:
    ///     str: Reference frame
    #[getter]
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Time system name (e.g., "UTC", "TDB").
    ///
    /// Returns:
    ///     str: Time system
    #[getter]
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    /// Start time of the ephemeris data as a CCSDS datetime string.
    ///
    /// Returns:
    ///     str: Start time
    #[getter]
    fn start_time(&self) -> String {
        crate::ccsds::common::format_ccsds_datetime(&self.inner.metadata.start_time)
    }

    /// Stop time of the ephemeris data as a CCSDS datetime string.
    ///
    /// Returns:
    ///     str: Stop time
    #[getter]
    fn stop_time(&self) -> String {
        crate::ccsds::common::format_ccsds_datetime(&self.inner.metadata.stop_time)
    }

    /// Interpolation method (e.g., "HERMITE", "LAGRANGE"), or None.
    ///
    /// Returns:
    ///     str: Interpolation method, or None
    #[getter]
    fn interpolation(&self) -> Option<String> {
        self.inner.metadata.interpolation.clone()
    }

    /// Interpolation degree, or None.
    ///
    /// Returns:
    ///     int: Interpolation degree, or None
    #[getter]
    fn interpolation_degree(&self) -> Option<u32> {
        self.inner.metadata.interpolation_degree
    }

    /// Comments from the metadata block.
    ///
    /// Returns:
    ///     list[str]: Comments
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Number of state vectors in this segment.
    ///
    /// Returns:
    ///     int: Number of states
    #[getter]
    fn num_states(&self) -> usize {
        self.inner.states.len()
    }

    /// Number of covariance matrices in this segment.
    ///
    /// Returns:
    ///     int: Number of covariances
    #[getter]
    fn num_covariances(&self) -> usize {
        self.inner.covariances.len()
    }

    /// Collection of state vectors, supporting len/indexing/iteration.
    ///
    /// Returns:
    ///     OEMStates: State vector collection
    #[getter]
    fn states(&self) -> PyOEMStates {
        PyOEMStates {
            inner: self.inner.states.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OEMSegment(object='{}', states={}, frame='{}')",
            self.inner.metadata.object_name,
            self.inner.states.len(),
            self.inner.metadata.ref_frame
        )
    }
}

/// Collection wrapper for OEM state vectors, supporting len/indexing/iteration.
///
/// Each state vector is returned as a dict with keys:
/// 'epoch', 'position', 'velocity', 'acceleration'.
///
/// Example:
///     ```python
///     states = seg.states
///     print(len(states))
///     sv = states[0]
///     print(sv["position"])
///     for sv in states:
///         print(sv["epoch"])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMStates")]
pub struct PyOEMStates {
    inner: Vec<OEMStateVector>,
}

/// Helper to convert an OEMStateVector to a Python dict.
fn state_vector_to_dict<'py>(
    py: Python<'py>,
    sv: &OEMStateVector,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("epoch", crate::ccsds::common::format_ccsds_datetime(&sv.epoch))?;
    dict.set_item("position", sv.position.to_vec())?;
    dict.set_item("velocity", sv.velocity.to_vec())?;
    dict.set_item("acceleration", sv.acceleration.map(|a| a.to_vec()))?;
    Ok(dict)
}

#[pymethods]
impl PyOEMStates {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let len = self.inner.len() as isize;
        let actual = if index < 0 { len + index } else { index };
        if actual < 0 || actual >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "state index {} out of range (have {})",
                index,
                self.inner.len()
            )));
        }
        state_vector_to_dict(py, &self.inner[actual as usize])
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOEMStateIterator>> {
        let py = slf.py();
        Py::new(
            py,
            PyOEMStateIterator {
                states: slf.into(),
                index: 0,
            },
        )
    }

    fn __repr__(&self) -> String {
        format!("OEMStates(len={})", self.inner.len())
    }
}

/// Iterator over OEM state vectors.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMStateIterator")]
pub struct PyOEMStateIterator {
    states: Py<PyOEMStates>,
    index: usize,
}

#[pymethods]
impl PyOEMStateIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let states = self.states.borrow(py);
        if self.index >= states.inner.len() {
            return Ok(None);
        }
        let dict = state_vector_to_dict(py, &states.inner[self.index])?;
        self.index += 1;
        Ok(Some(dict))
    }
}

// ─────────────────────────────────────────────
// OPM wrapper types
// ─────────────────────────────────────────────

/// Collection wrapper for OPM maneuvers, supporting len/indexing/iteration.
///
/// Each maneuver is returned as a dict with keys:
/// 'epoch_ignition', 'duration', 'delta_mass', 'ref_frame', 'dv'.
///
/// Example:
///     ```python
///     maneuvers = opm.maneuvers
///     print(len(maneuvers))
///     m = maneuvers[0]
///     print(m["ref_frame"])
///     for m in maneuvers:
///         print(m["dv"])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPMManeuvers")]
pub struct PyOPMManeuvers {
    inner: Vec<OPMManeuver>,
}

/// Helper to convert an OPMManeuver to a Python dict.
fn maneuver_to_dict<'py>(
    py: Python<'py>,
    m: &OPMManeuver,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "epoch_ignition",
        crate::ccsds::common::format_ccsds_datetime(&m.epoch_ignition),
    )?;
    dict.set_item("duration", m.duration)?;
    dict.set_item("delta_mass", m.delta_mass)?;
    dict.set_item("ref_frame", format!("{}", m.ref_frame))?;
    dict.set_item("dv", m.dv.to_vec())?;
    Ok(dict)
}

#[pymethods]
impl PyOPMManeuvers {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let len = self.inner.len() as isize;
        let actual = if index < 0 { len + index } else { index };
        if actual < 0 || actual >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "maneuver index {} out of range (have {})",
                index,
                self.inner.len()
            )));
        }
        maneuver_to_dict(py, &self.inner[actual as usize])
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOPMManeuverIterator>> {
        let py = slf.py();
        Py::new(
            py,
            PyOPMManeuverIterator {
                maneuvers: slf.into(),
                index: 0,
            },
        )
    }

    fn __repr__(&self) -> String {
        format!("OPMManeuvers(len={})", self.inner.len())
    }
}

/// Iterator over OPM maneuvers.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPMManeuverIterator")]
pub struct PyOPMManeuverIterator {
    maneuvers: Py<PyOPMManeuvers>,
    index: usize,
}

#[pymethods]
impl PyOPMManeuverIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let maneuvers = self.maneuvers.borrow(py);
        if self.index >= maneuvers.inner.len() {
            return Ok(None);
        }
        let dict = maneuver_to_dict(py, &maneuvers.inner[self.index])?;
        self.index += 1;
        Ok(Some(dict))
    }
}

// ─────────────────────────────────────────────
// PyOEM — Orbit Ephemeris Message
// ─────────────────────────────────────────────

/// Python wrapper for CCSDS Orbit Ephemeris Message (OEM).
///
/// OEM messages contain time-ordered sequences of state vectors (position
/// and velocity), optionally with accelerations and covariance matrices.
///
/// Header-level fields are properties (no parentheses). Segment data is
/// accessed via the ``segments`` property which supports indexing and iteration.
///
/// Example:
///     ```python
///     from brahe.ccsds import OEM
///
///     oem = OEM.from_file("ephemeris.oem")
///     print(oem.originator)           # property
///     print(len(oem.segments))        # number of segments
///     seg = oem.segments[0]
///     print(seg.object_name)          # segment property
///     for sv in seg.states:
///         print(sv["position"])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEM")]
pub struct PyOEM {
    inner: RustOEM,
}

#[pymethods]
impl PyOEM {
    // --- constructors ---

    /// Parse an OEM from a string, auto-detecting the format (KVN, XML, or JSON).
    ///
    /// Args:
    ///     content (str): String content of the OEM message
    ///
    /// Returns:
    ///     OEM: Parsed OEM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustOEM::from_str(content)?;
        Ok(PyOEM { inner })
    }

    /// Parse an OEM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the OEM file
    ///
    /// Returns:
    ///     OEM: Parsed OEM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustOEM::from_file(path)?;
        Ok(PyOEM { inner })
    }

    // --- serialization methods ---

    /// Write the OEM to a string in the specified format.
    ///
    /// Args:
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    ///
    /// Returns:
    ///     str: Serialized OEM string
    fn to_string(&self, format: &str) -> PyResult<String> {
        let fmt = parse_format(format)?;
        let result = self.inner.to_string(fmt)?;
        Ok(result)
    }

    /// Write the OEM to a file in the specified format.
    ///
    /// Args:
    ///     path (str): Output file path
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    fn to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.to_file(path, fmt)?;
        Ok(())
    }

    /// Convert the OEM to a Python dictionary.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OEM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item(
            "classification",
            self.inner.header.classification.as_deref(),
        )?;
        header.set_item(
            "creation_date",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date),
        )?;
        header.set_item("originator", &self.inner.header.originator)?;
        header.set_item("message_id", self.inner.header.message_id.as_deref())?;
        header.set_item("comments", &self.inner.header.comments)?;
        dict.set_item("header", header)?;

        // Segments
        let segments = pyo3::types::PyList::empty(py);
        for seg in &self.inner.segments {
            let seg_dict = pyo3::types::PyDict::new(py);

            // Metadata
            let meta = pyo3::types::PyDict::new(py);
            meta.set_item("object_name", &seg.metadata.object_name)?;
            meta.set_item("object_id", &seg.metadata.object_id)?;
            meta.set_item("center_name", &seg.metadata.center_name)?;
            meta.set_item("ref_frame", format!("{}", seg.metadata.ref_frame))?;
            meta.set_item("time_system", format!("{}", seg.metadata.time_system))?;
            meta.set_item(
                "start_time",
                crate::ccsds::common::format_ccsds_datetime(&seg.metadata.start_time),
            )?;
            meta.set_item(
                "stop_time",
                crate::ccsds::common::format_ccsds_datetime(&seg.metadata.stop_time),
            )?;
            meta.set_item("interpolation", seg.metadata.interpolation.as_deref())?;
            meta.set_item("interpolation_degree", seg.metadata.interpolation_degree)?;
            meta.set_item("comments", &seg.metadata.comments)?;
            seg_dict.set_item("metadata", meta)?;

            // States
            let states = pyo3::types::PyList::empty(py);
            for sv in &seg.states {
                states.append(state_vector_to_dict(py, sv)?)?;
            }
            seg_dict.set_item("states", states)?;
            seg_dict.set_item("comments", &seg.comments)?;
            seg_dict.set_item("num_covariances", seg.covariances.len())?;

            segments.append(seg_dict)?;
        }
        dict.set_item("segments", segments)?;

        Ok(dict)
    }

    // --- shortcut method ---

    /// Get a state vector from a segment by index (shortcut).
    ///
    /// This is a convenience method equivalent to ``oem.segments[segment_idx].states[state_idx]``.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///     state_idx (int): State vector index (0-based)
    ///
    /// Returns:
    ///     dict: State vector with 'epoch', 'position', 'velocity', 'acceleration'
    fn state<'py>(
        &self,
        py: Python<'py>,
        segment_idx: usize,
        state_idx: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let seg = self.inner.segments.get(segment_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "segment index {} out of range (have {})",
                segment_idx,
                self.inner.segments.len()
            ))
        })?;
        let sv = seg.states.get(state_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "state index {} out of range (have {})",
                state_idx,
                seg.states.len()
            ))
        })?;
        state_vector_to_dict(py, sv)
    }

    // --- header-level properties ---

    /// CCSDS format version.
    ///
    /// Returns:
    ///     float: Format version
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    /// Originator of the message.
    ///
    /// Returns:
    ///     str: Originator string
    #[getter]
    fn originator(&self) -> String {
        self.inner.header.originator.clone()
    }

    /// Classification of the message, or None.
    ///
    /// Returns:
    ///     str: Classification string, or None
    #[getter]
    fn classification(&self) -> Option<String> {
        self.inner.header.classification.clone()
    }

    /// Segment collection, supporting len/indexing/iteration.
    ///
    /// Returns:
    ///     OEMSegments: Collection of OEM segments
    #[getter]
    fn segments(&self) -> PyOEMSegments {
        PyOEMSegments {
            inner: self.inner.segments.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OEM(segments={}, originator='{}')",
            self.inner.segments.len(),
            self.inner.header.originator
        )
    }
}

// ─────────────────────────────────────────────
// PyOMM — Orbit Mean-elements Message
// ─────────────────────────────────────────────

/// Python wrapper for CCSDS Orbit Mean-elements Message (OMM).
///
/// OMM is flat (no segments), so all fields are exposed as properties.
///
/// Example:
///     ```python
///     from brahe.ccsds import OMM
///
///     omm = OMM.from_file("gp_data.omm")
///     print(omm.object_name)        # property
///     print(omm.eccentricity)       # property
///     d = omm.to_dict()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OMM")]
pub struct PyOMM {
    inner: RustOMM,
}

#[pymethods]
impl PyOMM {
    // --- constructors ---

    /// Parse an OMM from a string, auto-detecting the format.
    ///
    /// Args:
    ///     content (str): String content of the OMM message
    ///
    /// Returns:
    ///     OMM: Parsed OMM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustOMM::from_str(content)?;
        Ok(PyOMM { inner })
    }

    /// Parse an OMM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the OMM file
    ///
    /// Returns:
    ///     OMM: Parsed OMM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustOMM::from_file(path)?;
        Ok(PyOMM { inner })
    }

    // --- serialization ---

    /// Convert the OMM to a Python dictionary.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OMM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item(
            "creation_date",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date),
        )?;
        header.set_item("originator", &self.inner.header.originator)?;
        dict.set_item("header", header)?;

        // Metadata
        let meta = pyo3::types::PyDict::new(py);
        meta.set_item("object_name", &self.inner.metadata.object_name)?;
        meta.set_item("object_id", &self.inner.metadata.object_id)?;
        meta.set_item("center_name", &self.inner.metadata.center_name)?;
        meta.set_item("ref_frame", format!("{}", self.inner.metadata.ref_frame))?;
        meta.set_item(
            "time_system",
            format!("{}", self.inner.metadata.time_system),
        )?;
        meta.set_item(
            "mean_element_theory",
            &self.inner.metadata.mean_element_theory,
        )?;
        dict.set_item("metadata", meta)?;

        // Mean elements
        let me = pyo3::types::PyDict::new(py);
        me.set_item(
            "epoch",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.mean_elements.epoch),
        )?;
        me.set_item("mean_motion", self.inner.mean_elements.mean_motion)?;
        me.set_item(
            "semi_major_axis",
            self.inner.mean_elements.semi_major_axis,
        )?;
        me.set_item("eccentricity", self.inner.mean_elements.eccentricity)?;
        me.set_item("inclination", self.inner.mean_elements.inclination)?;
        me.set_item("ra_of_asc_node", self.inner.mean_elements.ra_of_asc_node)?;
        me.set_item(
            "arg_of_pericenter",
            self.inner.mean_elements.arg_of_pericenter,
        )?;
        me.set_item("mean_anomaly", self.inner.mean_elements.mean_anomaly)?;
        me.set_item("gm", self.inner.mean_elements.gm)?;
        dict.set_item("mean_elements", me)?;

        // TLE parameters
        if let Some(ref tle) = self.inner.tle_parameters {
            let tle_dict = pyo3::types::PyDict::new(py);
            tle_dict.set_item("ephemeris_type", tle.ephemeris_type)?;
            tle_dict.set_item(
                "classification_type",
                tle.classification_type.map(|c| c.to_string()),
            )?;
            tle_dict.set_item("norad_cat_id", tle.norad_cat_id)?;
            tle_dict.set_item("element_set_no", tle.element_set_no)?;
            tle_dict.set_item("rev_at_epoch", tle.rev_at_epoch)?;
            tle_dict.set_item("bstar", tle.bstar)?;
            tle_dict.set_item("bterm", tle.bterm)?;
            tle_dict.set_item("mean_motion_dot", tle.mean_motion_dot)?;
            tle_dict.set_item("mean_motion_ddot", tle.mean_motion_ddot)?;
            tle_dict.set_item("agom", tle.agom)?;
            dict.set_item("tle_parameters", tle_dict)?;
        }

        // Spacecraft parameters
        if let Some(ref sc) = self.inner.spacecraft_parameters {
            let sc_dict = pyo3::types::PyDict::new(py);
            sc_dict.set_item("mass", sc.mass)?;
            sc_dict.set_item("solar_rad_area", sc.solar_rad_area)?;
            sc_dict.set_item("solar_rad_coeff", sc.solar_rad_coeff)?;
            sc_dict.set_item("drag_area", sc.drag_area)?;
            sc_dict.set_item("drag_coeff", sc.drag_coeff)?;
            dict.set_item("spacecraft_parameters", sc_dict)?;
        }

        // User-defined
        if let Some(ref ud) = self.inner.user_defined {
            let ud_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &ud.parameters {
                ud_dict.set_item(k, v)?;
            }
            dict.set_item("user_defined", ud_dict)?;
        }

        Ok(dict)
    }

    // --- header properties ---

    /// CCSDS format version.
    ///
    /// Returns:
    ///     float: Format version
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    // --- metadata properties ---

    /// Object name.
    ///
    /// Returns:
    ///     str: Object name
    #[getter]
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// Object ID (international designator).
    ///
    /// Returns:
    ///     str: Object ID
    #[getter]
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Center body name.
    ///
    /// Returns:
    ///     str: Center name
    #[getter]
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Reference frame name.
    ///
    /// Returns:
    ///     str: Reference frame
    #[getter]
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Time system name.
    ///
    /// Returns:
    ///     str: Time system
    #[getter]
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    /// Mean element theory (e.g., "SGP/SGP4").
    ///
    /// Returns:
    ///     str: Mean element theory
    #[getter]
    fn mean_element_theory(&self) -> String {
        self.inner.metadata.mean_element_theory.clone()
    }

    // --- mean element properties ---

    /// Mean motion in rev/day, or None if not set.
    ///
    /// Returns:
    ///     float: Mean motion, or None
    #[getter]
    fn mean_motion(&self) -> Option<f64> {
        self.inner.mean_elements.mean_motion
    }

    /// Eccentricity.
    ///
    /// Returns:
    ///     float: Eccentricity
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.mean_elements.eccentricity
    }

    /// Inclination in degrees.
    ///
    /// Returns:
    ///     float: Inclination (degrees)
    #[getter]
    fn inclination(&self) -> f64 {
        self.inner.mean_elements.inclination
    }

    /// Right ascension of ascending node in degrees.
    ///
    /// Returns:
    ///     float: RAAN (degrees)
    #[getter]
    fn ra_of_asc_node(&self) -> f64 {
        self.inner.mean_elements.ra_of_asc_node
    }

    /// Argument of pericenter in degrees.
    ///
    /// Returns:
    ///     float: Argument of pericenter (degrees)
    #[getter]
    fn arg_of_pericenter(&self) -> f64 {
        self.inner.mean_elements.arg_of_pericenter
    }

    /// Mean anomaly in degrees.
    ///
    /// Returns:
    ///     float: Mean anomaly (degrees)
    #[getter]
    fn mean_anomaly(&self) -> f64 {
        self.inner.mean_elements.mean_anomaly
    }

    /// Gravitational parameter in m^3/s^2, or None.
    ///
    /// Returns:
    ///     float: GM, or None
    #[getter]
    fn gm(&self) -> Option<f64> {
        self.inner.mean_elements.gm
    }

    // --- TLE parameter properties ---

    /// NORAD catalog ID, or None.
    ///
    /// Returns:
    ///     int: NORAD catalog ID, or None
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.norad_cat_id)
    }

    /// BSTAR drag term, or None.
    ///
    /// Returns:
    ///     float: BSTAR, or None
    #[getter]
    fn bstar(&self) -> Option<f64> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.bstar)
    }

    /// First derivative of mean motion (rev/day^2), or None.
    ///
    /// Returns:
    ///     float: Mean motion dot, or None
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.mean_motion_dot)
    }

    /// Second derivative of mean motion (rev/day^3), or None.
    ///
    /// Returns:
    ///     float: Mean motion double-dot, or None
    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.mean_motion_ddot)
    }

    /// Element set number, or None.
    ///
    /// Returns:
    ///     int: Element set number, or None
    #[getter]
    fn element_set_no(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.element_set_no)
    }

    /// Revolution number at epoch, or None.
    ///
    /// Returns:
    ///     int: Rev at epoch, or None
    #[getter]
    fn rev_at_epoch(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.rev_at_epoch)
    }

    /// Classification type character, or None.
    ///
    /// Returns:
    ///     str: Classification type, or None
    #[getter]
    fn classification_type(&self) -> Option<String> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.classification_type.map(|c| c.to_string()))
    }

    /// Ephemeris type, or None.
    ///
    /// Returns:
    ///     int: Ephemeris type, or None
    #[getter]
    fn ephemeris_type(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.ephemeris_type)
    }

    fn __repr__(&self) -> String {
        format!(
            "OMM(object='{}', id='{}')",
            self.inner.metadata.object_name, self.inner.metadata.object_id
        )
    }
}

// ─────────────────────────────────────────────
// PyOPM — Orbit Parameter Message
// ─────────────────────────────────────────────

/// Python wrapper for CCSDS Orbit Parameter Message (OPM).
///
/// Scalar metadata is exposed as properties. Maneuvers are accessed via the
/// ``maneuvers`` property which supports indexing and iteration.
///
/// Example:
///     ```python
///     from brahe.ccsds import OPM
///
///     opm = OPM.from_file("state.opm")
///     print(opm.position)             # property
///     print(len(opm.maneuvers))       # number of maneuvers
///     for m in opm.maneuvers:
///         print(m["dv"])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPM")]
pub struct PyOPM {
    inner: RustOPM,
}

#[pymethods]
impl PyOPM {
    // --- constructors ---

    /// Parse an OPM from a string, auto-detecting the format.
    ///
    /// Args:
    ///     content (str): String content of the OPM message
    ///
    /// Returns:
    ///     OPM: Parsed OPM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustOPM::from_str(content)?;
        Ok(PyOPM { inner })
    }

    /// Parse an OPM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the OPM file
    ///
    /// Returns:
    ///     OPM: Parsed OPM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustOPM::from_file(path)?;
        Ok(PyOPM { inner })
    }

    // --- serialization ---

    /// Convert the OPM to a Python dictionary.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OPM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item(
            "creation_date",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date),
        )?;
        header.set_item("originator", &self.inner.header.originator)?;
        header.set_item("comments", &self.inner.header.comments)?;
        dict.set_item("header", header)?;

        // Metadata
        let meta = pyo3::types::PyDict::new(py);
        meta.set_item("object_name", &self.inner.metadata.object_name)?;
        meta.set_item("object_id", &self.inner.metadata.object_id)?;
        meta.set_item("center_name", &self.inner.metadata.center_name)?;
        meta.set_item("ref_frame", format!("{}", self.inner.metadata.ref_frame))?;
        meta.set_item(
            "time_system",
            format!("{}", self.inner.metadata.time_system),
        )?;
        dict.set_item("metadata", meta)?;

        // State vector
        let sv = pyo3::types::PyDict::new(py);
        sv.set_item(
            "epoch",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.state_vector.epoch),
        )?;
        sv.set_item("position", self.inner.state_vector.position.to_vec())?;
        sv.set_item("velocity", self.inner.state_vector.velocity.to_vec())?;
        dict.set_item("state_vector", sv)?;

        // Keplerian elements
        if let Some(ref kep) = self.inner.keplerian_elements {
            let kep_dict = pyo3::types::PyDict::new(py);
            kep_dict.set_item("semi_major_axis", kep.semi_major_axis)?;
            kep_dict.set_item("eccentricity", kep.eccentricity)?;
            kep_dict.set_item("inclination", kep.inclination)?;
            kep_dict.set_item("ra_of_asc_node", kep.ra_of_asc_node)?;
            kep_dict.set_item("arg_of_pericenter", kep.arg_of_pericenter)?;
            kep_dict.set_item("true_anomaly", kep.true_anomaly)?;
            kep_dict.set_item("mean_anomaly", kep.mean_anomaly)?;
            kep_dict.set_item("gm", kep.gm)?;
            dict.set_item("keplerian_elements", kep_dict)?;
        }

        // Spacecraft parameters
        if let Some(ref sc) = self.inner.spacecraft_parameters {
            let sc_dict = pyo3::types::PyDict::new(py);
            sc_dict.set_item("mass", sc.mass)?;
            sc_dict.set_item("solar_rad_area", sc.solar_rad_area)?;
            sc_dict.set_item("solar_rad_coeff", sc.solar_rad_coeff)?;
            sc_dict.set_item("drag_area", sc.drag_area)?;
            sc_dict.set_item("drag_coeff", sc.drag_coeff)?;
            dict.set_item("spacecraft_parameters", sc_dict)?;
        }

        // Maneuvers
        if !self.inner.maneuvers.is_empty() {
            let mans = pyo3::types::PyList::empty(py);
            for m in &self.inner.maneuvers {
                mans.append(maneuver_to_dict(py, m)?)?;
            }
            dict.set_item("maneuvers", mans)?;
        }

        // User-defined
        if let Some(ref ud) = self.inner.user_defined {
            let ud_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &ud.parameters {
                ud_dict.set_item(k, v)?;
            }
            dict.set_item("user_defined", ud_dict)?;
        }

        Ok(dict)
    }

    // --- header properties ---

    /// CCSDS format version.
    ///
    /// Returns:
    ///     float: Format version
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    // --- metadata properties ---

    /// Object name.
    ///
    /// Returns:
    ///     str: Object name
    #[getter]
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// Object ID (international designator).
    ///
    /// Returns:
    ///     str: Object ID
    #[getter]
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Center body name.
    ///
    /// Returns:
    ///     str: Center name
    #[getter]
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Reference frame name.
    ///
    /// Returns:
    ///     str: Reference frame
    #[getter]
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Time system name.
    ///
    /// Returns:
    ///     str: Time system
    #[getter]
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    // --- state vector properties ---

    /// Position vector [x, y, z] in meters.
    ///
    /// Returns:
    ///     list[float]: Position [x, y, z] in meters
    #[getter]
    fn position(&self) -> Vec<f64> {
        self.inner.state_vector.position.to_vec()
    }

    /// Velocity vector [vx, vy, vz] in m/s.
    ///
    /// Returns:
    ///     list[float]: Velocity [vx, vy, vz] in m/s
    #[getter]
    fn velocity(&self) -> Vec<f64> {
        self.inner.state_vector.velocity.to_vec()
    }

    // --- keplerian properties ---

    /// Whether Keplerian elements are present.
    ///
    /// Returns:
    ///     bool: True if Keplerian elements exist
    #[getter]
    fn has_keplerian_elements(&self) -> bool {
        self.inner.keplerian_elements.is_some()
    }

    /// Semi-major axis in meters, or None.
    ///
    /// Returns:
    ///     float: Semi-major axis (meters), or None
    #[getter]
    fn semi_major_axis(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .map(|k| k.semi_major_axis)
    }

    // --- spacecraft properties ---

    /// Spacecraft mass in kg, or None.
    ///
    /// Returns:
    ///     float: Mass (kg), or None
    #[getter]
    fn mass(&self) -> Option<f64> {
        self.inner
            .spacecraft_parameters
            .as_ref()
            .and_then(|s| s.mass)
    }

    // --- maneuver access ---

    /// Maneuver collection, supporting len/indexing/iteration.
    ///
    /// Returns:
    ///     OPMManeuvers: Collection of maneuvers
    #[getter]
    fn maneuvers(&self) -> PyOPMManeuvers {
        PyOPMManeuvers {
            inner: self.inner.maneuvers.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OPM(object='{}', frame='{}')",
            self.inner.metadata.object_name, self.inner.metadata.ref_frame
        )
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Parse a format string into CCSDSFormat.
fn parse_format(format: &str) -> PyResult<CCSDSFormat> {
    match format.to_uppercase().as_str() {
        "KVN" => Ok(CCSDSFormat::KVN),
        "XML" => Ok(CCSDSFormat::XML),
        "JSON" => Ok(CCSDSFormat::JSON),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown format '{}'. Expected 'KVN', 'XML', or 'JSON'",
            format
        ))),
    }
}
