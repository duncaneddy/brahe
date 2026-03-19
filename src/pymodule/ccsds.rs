// CCSDS Python bindings for OEM, OMM, and OPM message types.

use crate::ccsds::common::CCSDSFormat;
use crate::ccsds::oem::OEM as RustOEM;
use crate::ccsds::omm::OMM as RustOMM;
use crate::ccsds::opm::OPM as RustOPM;

/// Python wrapper for CCSDS Orbit Ephemeris Message (OEM).
///
/// OEM messages contain time-ordered sequences of state vectors (position
/// and velocity), optionally with accelerations and covariance matrices.
///
/// Example:
///     ```python
///     from brahe.ccsds import OEM
///
///     oem = OEM.from_file("ephemeris.oem")
///     print(f"Segments: {oem.num_segments()}")
///     print(f"Object: {oem.object_name(0)}")
///     d = oem.to_dict()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEM")]
pub struct PyOEM {
    inner: RustOEM,
}

#[pymethods]
impl PyOEM {
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
        header.set_item("classification", self.inner.header.classification.as_deref())?;
        header.set_item("creation_date", crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date))?;
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
            meta.set_item("start_time", crate::ccsds::common::format_ccsds_datetime(&seg.metadata.start_time))?;
            meta.set_item("stop_time", crate::ccsds::common::format_ccsds_datetime(&seg.metadata.stop_time))?;
            meta.set_item("interpolation", seg.metadata.interpolation.as_deref())?;
            meta.set_item("interpolation_degree", seg.metadata.interpolation_degree)?;
            meta.set_item("comments", &seg.metadata.comments)?;
            seg_dict.set_item("metadata", meta)?;

            // States
            let states = pyo3::types::PyList::empty(py);
            for sv in &seg.states {
                let sv_dict = pyo3::types::PyDict::new(py);
                sv_dict.set_item("epoch", crate::ccsds::common::format_ccsds_datetime(&sv.epoch))?;
                sv_dict.set_item("position", sv.position.to_vec())?;
                sv_dict.set_item("velocity", sv.velocity.to_vec())?;
                sv_dict.set_item("acceleration", sv.acceleration.map(|a| a.to_vec()))?;
                states.append(sv_dict)?;
            }
            seg_dict.set_item("states", states)?;
            seg_dict.set_item("comments", &seg.comments)?;
            seg_dict.set_item("num_covariances", seg.covariances.len())?;

            segments.append(seg_dict)?;
        }
        dict.set_item("segments", segments)?;

        Ok(dict)
    }

    /// Get the number of segments.
    ///
    /// Returns:
    ///     int: Number of segments
    fn num_segments(&self) -> usize {
        self.inner.segments.len()
    }

    /// Get the object name for a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///
    /// Returns:
    ///     str: Object name
    fn object_name(&self, segment_idx: usize) -> PyResult<String> {
        let seg = get_segment(&self.inner, segment_idx)?;
        Ok(seg.metadata.object_name.clone())
    }

    /// Get the object ID for a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///
    /// Returns:
    ///     str: Object ID (international designator)
    fn object_id(&self, segment_idx: usize) -> PyResult<String> {
        let seg = get_segment(&self.inner, segment_idx)?;
        Ok(seg.metadata.object_id.clone())
    }

    /// Get the center name for a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///
    /// Returns:
    ///     str: Center body name
    fn center_name(&self, segment_idx: usize) -> PyResult<String> {
        let seg = get_segment(&self.inner, segment_idx)?;
        Ok(seg.metadata.center_name.clone())
    }

    /// Get the reference frame name for a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///
    /// Returns:
    ///     str: Reference frame name
    fn ref_frame(&self, segment_idx: usize) -> PyResult<String> {
        let seg = get_segment(&self.inner, segment_idx)?;
        Ok(format!("{}", seg.metadata.ref_frame))
    }

    /// Get the number of state vectors in a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///
    /// Returns:
    ///     int: Number of state vectors
    fn num_states(&self, segment_idx: usize) -> PyResult<usize> {
        let seg = get_segment(&self.inner, segment_idx)?;
        Ok(seg.states.len())
    }

    /// Get a state vector from a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///     state_idx (int): State vector index (0-based)
    ///
    /// Returns:
    ///     dict: State vector with 'epoch', 'position', 'velocity', 'acceleration'
    fn state<'py>(&self, py: Python<'py>, segment_idx: usize, state_idx: usize) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let seg = get_segment(&self.inner, segment_idx)?;
        let sv = seg.states.get(state_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("state index {} out of range", state_idx))
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("epoch", crate::ccsds::common::format_ccsds_datetime(&sv.epoch))?;
        dict.set_item("position", sv.position.to_vec())?;
        dict.set_item("velocity", sv.velocity.to_vec())?;
        dict.set_item("acceleration", sv.acceleration.map(|a| a.to_vec()))?;
        Ok(dict)
    }

    /// Get the number of covariance matrices in a segment.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///
    /// Returns:
    ///     int: Number of covariance matrices
    fn num_covariances(&self, segment_idx: usize) -> PyResult<usize> {
        let seg = get_segment(&self.inner, segment_idx)?;
        Ok(seg.covariances.len())
    }

    /// Get the format version.
    ///
    /// Returns:
    ///     float: CCSDS format version
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    /// Get the originator.
    ///
    /// Returns:
    ///     str: Originator string
    fn originator(&self) -> String {
        self.inner.header.originator.clone()
    }

    /// Get the classification.
    ///
    /// Returns:
    ///     str: Classification string, or None
    fn classification(&self) -> Option<String> {
        self.inner.header.classification.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OEM(segments={}, originator='{}')",
            self.inner.segments.len(),
            self.inner.header.originator
        )
    }
}

/// Helper to get a segment by index with proper error.
fn get_segment(oem: &RustOEM, idx: usize) -> PyResult<&crate::ccsds::oem::OEMSegment> {
    oem.segments.get(idx).ok_or_else(|| {
        pyo3::exceptions::PyIndexError::new_err(format!(
            "segment index {} out of range (have {})", idx, oem.segments.len()
        ))
    })
}

/// Python wrapper for CCSDS Orbit Mean-elements Message (OMM).
///
/// Example:
///     ```python
///     from brahe.ccsds import OMM
///
///     omm = OMM.from_file("gp_data.omm")
///     print(f"Object: {omm.object_name()}")
///     d = omm.to_dict()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OMM")]
pub struct PyOMM {
    inner: RustOMM,
}

#[pymethods]
impl PyOMM {
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

    /// Convert the OMM to a Python dictionary.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OMM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item("creation_date", crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date))?;
        header.set_item("originator", &self.inner.header.originator)?;
        dict.set_item("header", header)?;

        // Metadata
        let meta = pyo3::types::PyDict::new(py);
        meta.set_item("object_name", &self.inner.metadata.object_name)?;
        meta.set_item("object_id", &self.inner.metadata.object_id)?;
        meta.set_item("center_name", &self.inner.metadata.center_name)?;
        meta.set_item("ref_frame", format!("{}", self.inner.metadata.ref_frame))?;
        meta.set_item("time_system", format!("{}", self.inner.metadata.time_system))?;
        meta.set_item("mean_element_theory", &self.inner.metadata.mean_element_theory)?;
        dict.set_item("metadata", meta)?;

        // Mean elements
        let me = pyo3::types::PyDict::new(py);
        me.set_item("epoch", crate::ccsds::common::format_ccsds_datetime(&self.inner.mean_elements.epoch))?;
        me.set_item("mean_motion", self.inner.mean_elements.mean_motion)?;
        me.set_item("semi_major_axis", self.inner.mean_elements.semi_major_axis)?;
        me.set_item("eccentricity", self.inner.mean_elements.eccentricity)?;
        me.set_item("inclination", self.inner.mean_elements.inclination)?;
        me.set_item("ra_of_asc_node", self.inner.mean_elements.ra_of_asc_node)?;
        me.set_item("arg_of_pericenter", self.inner.mean_elements.arg_of_pericenter)?;
        me.set_item("mean_anomaly", self.inner.mean_elements.mean_anomaly)?;
        me.set_item("gm", self.inner.mean_elements.gm)?;
        dict.set_item("mean_elements", me)?;

        // TLE parameters
        if let Some(ref tle) = self.inner.tle_parameters {
            let tle_dict = pyo3::types::PyDict::new(py);
            tle_dict.set_item("ephemeris_type", tle.ephemeris_type)?;
            tle_dict.set_item("classification_type", tle.classification_type.map(|c| c.to_string()))?;
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

    /// Get the object name.
    ///
    /// Returns:
    ///     str: Object name
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// Get the object ID.
    ///
    /// Returns:
    ///     str: Object ID
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Get the center name.
    ///
    /// Returns:
    ///     str: Center body name
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Get the reference frame.
    ///
    /// Returns:
    ///     str: Reference frame name
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Get the time system.
    ///
    /// Returns:
    ///     str: Time system name
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    /// Get the mean element theory.
    ///
    /// Returns:
    ///     str: Mean element theory (e.g., "SGP/SGP4")
    fn mean_element_theory(&self) -> String {
        self.inner.metadata.mean_element_theory.clone()
    }

    /// Get the mean motion in rev/day.
    ///
    /// Returns:
    ///     float: Mean motion, or None if not set
    fn mean_motion(&self) -> Option<f64> {
        self.inner.mean_elements.mean_motion
    }

    /// Get the eccentricity.
    ///
    /// Returns:
    ///     float: Eccentricity
    fn eccentricity(&self) -> f64 {
        self.inner.mean_elements.eccentricity
    }

    /// Get the inclination in degrees.
    ///
    /// Returns:
    ///     float: Inclination (degrees)
    fn inclination(&self) -> f64 {
        self.inner.mean_elements.inclination
    }

    /// Get the right ascension of ascending node in degrees.
    ///
    /// Returns:
    ///     float: RAAN (degrees)
    fn ra_of_asc_node(&self) -> f64 {
        self.inner.mean_elements.ra_of_asc_node
    }

    /// Get the argument of pericenter in degrees.
    ///
    /// Returns:
    ///     float: Argument of pericenter (degrees)
    fn arg_of_pericenter(&self) -> f64 {
        self.inner.mean_elements.arg_of_pericenter
    }

    /// Get the mean anomaly in degrees.
    ///
    /// Returns:
    ///     float: Mean anomaly (degrees)
    fn mean_anomaly(&self) -> f64 {
        self.inner.mean_elements.mean_anomaly
    }

    /// Get the GM in m^3/s^2.
    ///
    /// Returns:
    ///     float: GM, or None if not set
    fn gm(&self) -> Option<f64> {
        self.inner.mean_elements.gm
    }

    /// Get the NORAD catalog ID.
    ///
    /// Returns:
    ///     int: NORAD catalog ID, or None
    fn norad_cat_id(&self) -> Option<u32> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.norad_cat_id)
    }

    /// Get the BSTAR drag term.
    ///
    /// Returns:
    ///     float: BSTAR, or None
    fn bstar(&self) -> Option<f64> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.bstar)
    }

    /// Get the first derivative of mean motion.
    ///
    /// Returns:
    ///     float: Mean motion dot (rev/day^2), or None
    fn mean_motion_dot(&self) -> Option<f64> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.mean_motion_dot)
    }

    /// Get the second derivative of mean motion.
    ///
    /// Returns:
    ///     float: Mean motion double-dot (rev/day^3), or None
    fn mean_motion_ddot(&self) -> Option<f64> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.mean_motion_ddot)
    }

    /// Get the element set number.
    ///
    /// Returns:
    ///     int: Element set number, or None
    fn element_set_no(&self) -> Option<u32> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.element_set_no)
    }

    /// Get the revolution number at epoch.
    ///
    /// Returns:
    ///     int: Rev at epoch, or None
    fn rev_at_epoch(&self) -> Option<u32> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.rev_at_epoch)
    }

    /// Get the classification type.
    ///
    /// Returns:
    ///     str: Classification type character, or None
    fn classification_type(&self) -> Option<String> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.classification_type.map(|c| c.to_string()))
    }

    /// Get the ephemeris type.
    ///
    /// Returns:
    ///     int: Ephemeris type, or None
    fn ephemeris_type(&self) -> Option<u32> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.ephemeris_type)
    }

    /// Get the format version.
    ///
    /// Returns:
    ///     float: CCSDS format version
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    fn __repr__(&self) -> String {
        format!(
            "OMM(object='{}', id='{}')",
            self.inner.metadata.object_name,
            self.inner.metadata.object_id
        )
    }
}

/// Python wrapper for CCSDS Orbit Parameter Message (OPM).
///
/// Example:
///     ```python
///     from brahe.ccsds import OPM
///
///     opm = OPM.from_file("state.opm")
///     pos = opm.position()
///     d = opm.to_dict()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPM")]
pub struct PyOPM {
    inner: RustOPM,
}

#[pymethods]
impl PyOPM {
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

    /// Convert the OPM to a Python dictionary.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OPM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item("creation_date", crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date))?;
        header.set_item("originator", &self.inner.header.originator)?;
        header.set_item("comments", &self.inner.header.comments)?;
        dict.set_item("header", header)?;

        // Metadata
        let meta = pyo3::types::PyDict::new(py);
        meta.set_item("object_name", &self.inner.metadata.object_name)?;
        meta.set_item("object_id", &self.inner.metadata.object_id)?;
        meta.set_item("center_name", &self.inner.metadata.center_name)?;
        meta.set_item("ref_frame", format!("{}", self.inner.metadata.ref_frame))?;
        meta.set_item("time_system", format!("{}", self.inner.metadata.time_system))?;
        dict.set_item("metadata", meta)?;

        // State vector
        let sv = pyo3::types::PyDict::new(py);
        sv.set_item("epoch", crate::ccsds::common::format_ccsds_datetime(&self.inner.state_vector.epoch))?;
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
                let m_dict = pyo3::types::PyDict::new(py);
                m_dict.set_item("epoch_ignition", crate::ccsds::common::format_ccsds_datetime(&m.epoch_ignition))?;
                m_dict.set_item("duration", m.duration)?;
                m_dict.set_item("delta_mass", m.delta_mass)?;
                m_dict.set_item("ref_frame", format!("{}", m.ref_frame))?;
                m_dict.set_item("dv", m.dv.to_vec())?;
                mans.append(m_dict)?;
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

    /// Get the object name.
    ///
    /// Returns:
    ///     str: Object name
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// Get the object ID.
    ///
    /// Returns:
    ///     str: Object ID
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Get the center name.
    ///
    /// Returns:
    ///     str: Center body name
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Get the position vector [x, y, z] in meters.
    ///
    /// Returns:
    ///     list[float]: Position [x, y, z] in meters
    fn position(&self) -> Vec<f64> {
        self.inner.state_vector.position.to_vec()
    }

    /// Get the velocity vector [vx, vy, vz] in m/s.
    ///
    /// Returns:
    ///     list[float]: Velocity [vx, vy, vz] in m/s
    fn velocity(&self) -> Vec<f64> {
        self.inner.state_vector.velocity.to_vec()
    }

    /// Get the reference frame name.
    ///
    /// Returns:
    ///     str: Reference frame
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Get the time system.
    ///
    /// Returns:
    ///     str: Time system
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    /// Get the number of maneuvers.
    ///
    /// Returns:
    ///     int: Number of maneuvers
    fn num_maneuvers(&self) -> usize {
        self.inner.maneuvers.len()
    }

    /// Get maneuver details by index.
    ///
    /// Args:
    ///     idx (int): Maneuver index (0-based)
    ///
    /// Returns:
    ///     dict: Maneuver details
    fn maneuver<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let m = self.inner.maneuvers.get(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("maneuver index {} out of range", idx))
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("epoch_ignition", crate::ccsds::common::format_ccsds_datetime(&m.epoch_ignition))?;
        dict.set_item("duration", m.duration)?;
        dict.set_item("delta_mass", m.delta_mass)?;
        dict.set_item("ref_frame", format!("{}", m.ref_frame))?;
        dict.set_item("dv", m.dv.to_vec())?;
        Ok(dict)
    }

    /// Check if Keplerian elements are present.
    ///
    /// Returns:
    ///     bool: True if Keplerian elements exist
    fn has_keplerian_elements(&self) -> bool {
        self.inner.keplerian_elements.is_some()
    }

    /// Get the semi-major axis in meters.
    ///
    /// Returns:
    ///     float: Semi-major axis (meters), or None
    fn semi_major_axis(&self) -> Option<f64> {
        self.inner.keplerian_elements.as_ref().map(|k| k.semi_major_axis)
    }

    /// Get the spacecraft mass in kg.
    ///
    /// Returns:
    ///     float: Mass (kg), or None
    fn mass(&self) -> Option<f64> {
        self.inner.spacecraft_parameters.as_ref().and_then(|s| s.mass)
    }

    /// Get the format version.
    ///
    /// Returns:
    ///     float: CCSDS format version
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    fn __repr__(&self) -> String {
        format!(
            "OPM(object='{}', frame='{}')",
            self.inner.metadata.object_name,
            self.inner.metadata.ref_frame
        )
    }
}

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
