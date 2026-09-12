// Python bindings for the Modified ITC ephemeris format.

/// Frame in which a Modified ITC file expresses its covariance.
///
/// ``UVW`` is Space-Track's name for the radial, in-track, cross-track
/// frame and is what ``RTN`` writes.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.ITCCovarianceFrame.parse("UVW") == bh.ITCCovarianceFrame.RTN
///     assert str(bh.ITCCovarianceFrame.RTN) == "UVW"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ITCCovarianceFrame")]
#[derive(Clone, PartialEq)]
pub struct PyITCCovarianceFrame {
    pub(crate) inner: itc::ITCCovarianceFrame,
}

#[pymethods]
impl PyITCCovarianceFrame {
    /// Radial, in-track, cross-track (tokens UVW, RTN, RSW, RIC).
    #[classattr]
    #[allow(non_snake_case)]
    fn RTN() -> Self {
        Self { inner: itc::ITCCovarianceFrame::RTN }
    }

    /// Mean equator and equinox of J2000.0 (tokens EME2000, J2000).
    #[classattr]
    #[allow(non_snake_case)]
    fn EME2000() -> Self {
        Self { inner: itc::ITCCovarianceFrame::EME2000 }
    }

    /// International Terrestrial Reference Frame (token ITRF).
    #[classattr]
    #[allow(non_snake_case)]
    fn ITRF() -> Self {
        Self { inner: itc::ITCCovarianceFrame::ITRF }
    }

    /// Parse the covariance-frame header token.
    ///
    /// Args:
    ///     token (str): Fourth header line of a Modified ITC file, case-insensitive.
    ///
    /// Returns:
    ///     ITCCovarianceFrame: The frame.
    ///
    /// Raises:
    ///     BraheError: If the token is not one the handbook lists.
    #[staticmethod]
    fn parse(token: &str) -> PyResult<Self> {
        itc::ITCCovarianceFrame::parse(token)
            .map(|inner| Self { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// The token written on the fourth header line.
    ///
    /// Returns:
    ///     str: ``UVW``, ``EME2000`` or ``ITRF``.
    fn token(&self) -> String {
        self.inner.token().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        let name = match self.inner {
            itc::ITCCovarianceFrame::RTN => "RTN",
            itc::ITCCovarianceFrame::EME2000 => "EME2000",
            itc::ITCCovarianceFrame::ITRF => "ITRF",
        };
        format!("ITCCovarianceFrame.{}", name)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        match self.inner {
            itc::ITCCovarianceFrame::RTN => 0,
            itc::ITCCovarianceFrame::EME2000 => 1,
            itc::ITCCovarianceFrame::ITRF => 2,
        }
    }
}

/// Header of a Modified ITC file.
///
/// The typed fields mirror Starlink's descriptive header lines. The state
/// frame is not written in the file body; ``ITC.from_file`` infers it from
/// the file name's DataType and it defaults to ``CelestialFrame.EME2000``.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     header = bh.ITCHeader(ephemeris_source="blend")
///     assert header.state_frame == bh.CelestialFrame.EME2000
///     assert header.covariance_frame == bh.ITCCovarianceFrame.RTN
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ITCHeader")]
#[derive(Clone)]
pub struct PyITCHeader {
    pub(crate) inner: itc::ITCHeader,
}

#[pymethods]
impl PyITCHeader {
    /// Create a header.
    ///
    /// Args:
    ///     created (Epoch, optional): File creation time.
    ///     ephemeris_start (Epoch, optional): Declared first epoch.
    ///     ephemeris_stop (Epoch, optional): Declared last epoch.
    ///     step_size (float, optional): Declared step between records, seconds.
    ///     ephemeris_source (str, optional): Free-text source label.
    ///     state_frame (CelestialFrame, optional): Frame of the states; defaults to EME2000.
    ///     covariance_frame (ITCCovarianceFrame, optional): Frame of the covariance; defaults to RTN.
    ///
    /// Returns:
    ///     ITCHeader: The header.
    #[new]
    #[pyo3(signature = (created=None, ephemeris_start=None, ephemeris_stop=None, step_size=None, ephemeris_source=None, state_frame=None, covariance_frame=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        created: Option<PyEpoch>,
        ephemeris_start: Option<PyEpoch>,
        ephemeris_stop: Option<PyEpoch>,
        step_size: Option<f64>,
        ephemeris_source: Option<String>,
        state_frame: Option<PyCelestialFrame>,
        covariance_frame: Option<PyITCCovarianceFrame>,
    ) -> Self {
        let mut inner = itc::ITCHeader::new();
        inner.created = created.map(|e| e.obj);
        inner.ephemeris_start = ephemeris_start.map(|e| e.obj);
        inner.ephemeris_stop = ephemeris_stop.map(|e| e.obj);
        inner.step_size = step_size;
        inner.ephemeris_source = ephemeris_source;
        if let Some(frame) = state_frame {
            inner.state_frame = frame.frame;
        }
        if let Some(frame) = covariance_frame {
            inner.covariance_frame = frame.inner;
        }
        Self { inner }
    }

    /// File creation time.
    ///
    /// Returns:
    ///     Epoch | None: Creation epoch, if present.
    #[getter]
    fn created(&self) -> Option<PyEpoch> {
        self.inner.created.map(|obj| PyEpoch { obj })
    }

    #[setter]
    fn set_created(&mut self, value: Option<PyEpoch>) {
        self.inner.created = value.map(|e| e.obj);
    }

    /// Declared first epoch.
    ///
    /// Returns:
    ///     Epoch | None: Start epoch, if present.
    #[getter]
    fn ephemeris_start(&self) -> Option<PyEpoch> {
        self.inner.ephemeris_start.map(|obj| PyEpoch { obj })
    }

    #[setter]
    fn set_ephemeris_start(&mut self, value: Option<PyEpoch>) {
        self.inner.ephemeris_start = value.map(|e| e.obj);
    }

    /// Declared last epoch.
    ///
    /// Returns:
    ///     Epoch | None: Stop epoch, if present.
    #[getter]
    fn ephemeris_stop(&self) -> Option<PyEpoch> {
        self.inner.ephemeris_stop.map(|obj| PyEpoch { obj })
    }

    #[setter]
    fn set_ephemeris_stop(&mut self, value: Option<PyEpoch>) {
        self.inner.ephemeris_stop = value.map(|e| e.obj);
    }

    /// Declared step between records.
    ///
    /// Returns:
    ///     float | None: Step in seconds, if present.
    #[getter]
    fn step_size(&self) -> Option<f64> {
        self.inner.step_size
    }

    #[setter]
    fn set_step_size(&mut self, value: Option<f64>) {
        self.inner.step_size = value;
    }

    /// Free-text source label.
    ///
    /// Returns:
    ///     str | None: Source label, if present.
    #[getter]
    fn ephemeris_source(&self) -> Option<String> {
        self.inner.ephemeris_source.clone()
    }

    #[setter]
    fn set_ephemeris_source(&mut self, value: Option<String>) {
        self.inner.ephemeris_source = value;
    }

    /// Frame of the state vectors.
    ///
    /// Returns:
    ///     CelestialFrame: State frame.
    #[getter]
    fn state_frame(&self) -> PyCelestialFrame {
        PyCelestialFrame { frame: self.inner.state_frame }
    }

    #[setter]
    fn set_state_frame(&mut self, value: PyCelestialFrame) {
        self.inner.state_frame = value.frame;
    }

    /// Frame of the covariance matrices.
    ///
    /// Returns:
    ///     ITCCovarianceFrame: Covariance frame.
    #[getter]
    fn covariance_frame(&self) -> PyITCCovarianceFrame {
        PyITCCovarianceFrame { inner: self.inner.covariance_frame }
    }

    #[setter]
    fn set_covariance_frame(&mut self, value: PyITCCovarianceFrame) {
        self.inner.covariance_frame = value.inner;
    }

    fn __repr__(&self) -> String {
        let step_size = match self.inner.step_size {
            Some(value) => format!("{}", value),
            None => "None".to_string(),
        };
        let ephemeris_source = match &self.inner.ephemeris_source {
            Some(value) => format!("'{}'", value),
            None => "None".to_string(),
        };
        format!(
            "ITCHeader(state_frame={}, covariance_frame={}, step_size={}, ephemeris_source={})",
            self.inner.state_frame, self.inner.covariance_frame, step_size, ephemeris_source
        )
    }
}

/// One Modified ITC record: epoch, position and velocity in SI units.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epoch = bh.Epoch(2026, 9, 11, 1, 42, 42.0, 0.0, time_system=bh.TimeSystem.UTC)
///     state = bh.ITCStateVector(epoch, np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]))
///     assert state.position[0] == 7.0e6
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ITCStateVector")]
#[derive(Clone)]
pub struct PyITCStateVector {
    pub(crate) inner: itc::ITCStateVector,
}

#[pymethods]
impl PyITCStateVector {
    /// Create a record.
    ///
    /// Args:
    ///     epoch (Epoch): Record epoch.
    ///     position (numpy.ndarray): Position, meters, shape (3,).
    ///     velocity (numpy.ndarray): Velocity, meters per second, shape (3,).
    ///
    /// Returns:
    ///     ITCStateVector: The record.
    #[new]
    fn new(epoch: PyEpoch, position: PyReadonlyArray1<f64>, velocity: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let p = position.as_slice()?;
        let v = velocity.as_slice()?;
        if p.len() != 3 || v.len() != 3 {
            return Err(exceptions::PyValueError::new_err("position and velocity must each have three components"));
        }
        Ok(Self { inner: itc::ITCStateVector::new(epoch.obj, [p[0], p[1], p[2]], [v[0], v[1], v[2]]) })
    }

    /// Record epoch.
    ///
    /// Returns:
    ///     Epoch: The epoch.
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.inner.epoch }
    }

    /// Position in the header's state frame.
    ///
    /// Returns:
    ///     numpy.ndarray: Position, meters, shape (3,).
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.position.to_vec().into_pyarray(py)
    }

    /// Velocity in the header's state frame.
    ///
    /// Returns:
    ///     numpy.ndarray: Velocity, meters per second, shape (3,).
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.velocity.to_vec().into_pyarray(py)
    }

    /// The record's position and velocity as a single Cartesian state.
    ///
    /// Returns:
    ///     numpy.ndarray: ``[x, y, z, vx, vy, vz]`` in the header's state frame, meters and meters per second, shape (6,).
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epoch = bh.Epoch(2026, 9, 11, 1, 42, 42.0, 0.0, time_system=bh.TimeSystem.UTC)
    ///     state = bh.ITCStateVector(epoch, np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0]))
    ///     assert state.to_vector()[4] == 7.5e3
    ///     ```
    fn to_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.to_vector().as_slice().to_vec().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!("ITCStateVector(epoch={}, position={:?}, velocity={:?})", self.inner.epoch, self.inner.position, self.inner.velocity)
    }
}

/// A Modified ITC ephemeris message.
///
/// Read Starlink and other Space-Track Modified ITC files, build messages
/// record by record, and write them back in the same text layout. All
/// values are SI: meters, meters per second, and m², m²/s, m²/s² for the
/// covariance in the header's covariance frame. Properties return copies;
/// to change the header, modify a copy and assign it back.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     itc = bh.ITC.from_file(
///         "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt"
///     )
///     assert len(itc) == 50
///     assert itc.has_covariance
///     assert itc.source_name.object_name == "STARLINK-37711"
///     text = itc.to_string()
///
///     traj = itc.to_trajectory()
///     station = bh.PointLocation(-122.4194, 37.7749, 0.0)
///     windows = bh.location_accesses([station], [traj], itc.start_epoch, itc.end_epoch, bh.ElevationConstraint(min_elevation_deg=10.0))
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ITC")]
#[derive(Clone)]
pub struct PyITC {
    pub(crate) inner: itc::ITC,
}

#[pymethods]
impl PyITC {
    /// Create an empty message with the given header.
    ///
    /// Args:
    ///     header (ITCHeader): Header fields.
    ///
    /// Returns:
    ///     ITC: A message with no records.
    #[new]
    fn new(header: PyITCHeader) -> Self {
        Self { inner: itc::ITC::new(header.inner) }
    }

    /// Parse a message from text.
    ///
    /// The state frame stays at the header default (EME2000); use
    /// ``from_file`` to infer it from a compliant file name.
    ///
    /// Args:
    ///     content (str): Full file text.
    ///
    /// Returns:
    ///     ITC: The parsed message.
    ///
    /// Raises:
    ///     BraheError: If the header or any record is malformed.
    #[staticmethod]
    fn from_str(content: &str) -> PyResult<Self> {
        itc::ITC::from_str(content).map(|inner| Self { inner }).map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Read and parse a file.
    ///
    /// When the file name follows the Space-Track convention its DataType
    /// sets the state frame and the parsed name is kept in ``source_name``.
    ///
    /// Args:
    ///     path (str): Path to the file.
    ///
    /// Returns:
    ///     ITC: The parsed message.
    ///
    /// Raises:
    ///     BraheError: If the file cannot be read or parsed, or names an unsupported DataType.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        itc::ITC::from_file(path).map(|inner| Self { inner }).map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Render the message as Modified ITC text.
    ///
    /// Returns:
    ///     str: The file text.
    ///
    /// Raises:
    ///     BraheError: If the message has no records.
    fn to_string(&self) -> PyResult<String> {
        self.inner.to_string().map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Write the message to a file.
    ///
    /// Args:
    ///     path (str): Destination path.
    ///
    /// Raises:
    ///     BraheError: If the message has no records or the file cannot be written.
    fn to_file(&self, path: &str) -> PyResult<()> {
        self.inner.to_file(path).map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Header fields.
    ///
    /// Returns:
    ///     ITCHeader: A copy of the header; assign it back (``itc.header = h``) to apply changes.
    #[getter]
    fn header(&self) -> PyITCHeader {
        PyITCHeader { inner: self.inner.header.clone() }
    }

    #[setter]
    fn set_header(&mut self, value: PyITCHeader) {
        self.inner.header = value.inner;
    }

    /// Ephemeris records in epoch order.
    ///
    /// Returns:
    ///     list[ITCStateVector]: Copies of the records.
    #[getter]
    fn states(&self) -> Vec<PyITCStateVector> {
        self.inner.states.iter().map(|s| PyITCStateVector { inner: s.clone() }).collect()
    }

    /// Covariance per record in the header's covariance frame.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: One 6x6 array per record, or an empty list when the file has no covariance.
    #[getter]
    fn covariances<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray<f64, Ix2>>> {
        self.inner
            .covariances
            .iter()
            .map(|m| {
                let mut array = ndarray::Array2::<f64>::zeros((6, 6));
                for i in 0..6 {
                    for j in 0..6 {
                        array[[i, j]] = m[(i, j)];
                    }
                }
                array.to_pyarray(py)
            })
            .collect()
    }

    /// Whether the message carries covariance.
    ///
    /// Returns:
    ///     bool: True when every record has a covariance matrix.
    #[getter]
    fn has_covariance(&self) -> bool {
        self.inner.has_covariance()
    }

    /// Parsed file name, when loaded from a compliant file name.
    ///
    /// Returns:
    ///     SpaceTrackEphemerisFileName | None: The parsed name.
    #[getter]
    fn source_name(&self) -> Option<PySpaceTrackEphemerisFileName> {
        self.inner.source_name.clone().map(|inner| PySpaceTrackEphemerisFileName { inner })
    }

    /// Epoch of the first record.
    ///
    /// Returns:
    ///     Epoch | None: First epoch, or None when empty.
    #[getter]
    fn start_epoch(&self) -> Option<PyEpoch> {
        self.inner.start_epoch().map(|obj| PyEpoch { obj })
    }

    /// Epoch of the last record.
    ///
    /// Returns:
    ///     Epoch | None: Last epoch, or None when empty.
    #[getter]
    fn end_epoch(&self) -> Option<PyEpoch> {
        self.inner.end_epoch().map(|obj| PyEpoch { obj })
    }

    /// Append a record without covariance.
    ///
    /// Args:
    ///     state (ITCStateVector): Record with an epoch later than the last one.
    ///
    /// Raises:
    ///     BraheError: If the epoch is not increasing or the message already carries covariance.
    fn push_state(&mut self, state: PyITCStateVector) -> PyResult<()> {
        self.inner.push_state(state.inner).map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Append a record with its covariance.
    ///
    /// Args:
    ///     state (ITCStateVector): Record with an epoch later than the last one.
    ///     covariance (numpy.ndarray): 6x6 covariance in the header's covariance frame, SI units.
    ///
    /// Raises:
    ///     BraheError: If the epoch is not increasing or earlier records have no covariance.
    fn push_state_with_covariance(&mut self, state: PyITCStateVector, covariance: PyReadonlyArray2<f64>) -> PyResult<()> {
        let view = covariance.as_array();
        if view.shape() != [6, 6] {
            return Err(exceptions::PyValueError::new_err("covariance must be a 6x6 array"));
        }
        let mut matrix = nalgebra::SMatrix::<f64, 6, 6>::zeros();
        for i in 0..6 {
            for j in 0..6 {
                matrix[(i, j)] = view[[i, j]];
            }
        }
        self.inner
            .push_state_with_covariance(state.inner, matrix)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Build a Space-Track compliant file name for this message.
    ///
    /// Args:
    ///     norad_cat_id (int): NORAD catalog number or analyst number.
    ///     object_name (str): Common name of the object.
    ///     category (SpaceTrackEphemerisFileCategory): Operational or Special.
    ///     metadata (str): Operator-defined metadata; may be empty.
    ///
    /// Returns:
    ///     SpaceTrackEphemerisFileName: The name; ``str()`` gives the text.
    ///
    /// Raises:
    ///     BraheError: If the message has no start epoch or its state frame has no DataType.
    fn file_name(&self, norad_cat_id: u32, object_name: &str, category: PySpaceTrackEphemerisFileCategory, metadata: &str) -> PyResult<PySpaceTrackEphemerisFileName> {
        self.inner
            .file_name(norad_cat_id, object_name, category.inner, metadata)
            .map(|inner| PySpaceTrackEphemerisFileName { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let covariance = if self.inner.has_covariance() { "True" } else { "False" };
        format!("ITC(records={}, covariance={}, state_frame={})", self.inner.len(), covariance, self.inner.header.state_frame)
    }

    /// Convert the message to an ``OrbitTrajectory`` in the header's state frame.
    ///
    /// Covariance is attached in the state frame whatever frame the header
    /// names for it: RTN with each record's own state, block-diagonally by
    /// default, EME2000 and ITRF with the state-transform Jacobian between the
    /// covariance frame and the state frame.
    ///
    /// Args:
    ///     covariance_variant (OrbitRelativeFrameVariant, optional): ``INERTIAL`` (default, block-diagonal ``[[R, 0], [0, R]]``) or ``ROTATING`` (adds the frame-rate coupling block).
    ///
    /// Returns:
    ///     OrbitTrajectory: Six-dimensional Cartesian trajectory, named after the source file's object when known.
    ///
    /// Raises:
    ///     BraheError: If the message is empty, or the frame router cannot relate the covariance frame and the state frame at a record's epoch.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     itc = bh.ITC.from_file("test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt")
    ///     traj = itc.to_trajectory()
    ///     cov = traj.covariance(itc.start_epoch)
    ///     ```
    ///
    /// References:
    ///     1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense Squadron, Space-Track.org, https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf
    ///     2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13), https://ntrs.nasa.gov/citations/20205011318
    ///     3. NASA CARA Analysis Tools, ``RIC2ECI.m``, https://github.com/nasa/CARA_Analysis_Tools
    ///     4. D. A. Vallado, "Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003, https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf
    #[pyo3(signature = (covariance_variant=None))]
    fn to_trajectory(&self, covariance_variant: Option<PyOrbitRelativeFrameVariant>) -> PyResult<PyOrbitalTrajectory> {
        let variant = covariance_variant.map(|v| v.variant).unwrap_or(frames::OrbitRelativeFrameVariant::Inertial);
        self.inner
            .to_trajectory_with_covariance_variant(variant)
            .map(|trajectory| PyOrbitalTrajectory { trajectory })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Convert the message to an ``OrbitTrajectory`` with an explicit RTN covariance convention.
    ///
    /// The covariance is attached in the state frame whatever frame the header names for it.
    /// An ``RTN`` covariance is rotated with the record's own state expressed in GCRF (the geocentric RTN basis, whatever center the
    /// source used); ``INERTIAL`` uses the block-diagonal rotation and ``ROTATING`` adds
    /// the frame-rate coupling. An ``EME2000`` or ``ITRF`` covariance is rotated by the
    /// state-transform Jacobian from that frame to the state frame at the record epoch, which is
    /// the identity when the two agree.
    ///
    /// Args:
    ///     variant (OrbitRelativeFrameVariant): ``INERTIAL`` or ``ROTATING``.
    ///
    /// Returns:
    ///     OrbitTrajectory: The trajectory.
    ///
    /// Raises:
    ///     BraheError: If the message is empty, or the frame router cannot relate the covariance frame and the state frame at a record's epoch.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     itc = bh.ITC.from_file("test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt")
    ///     traj = itc.to_trajectory_with_covariance_variant(bh.OrbitRelativeFrameVariant.ROTATING)
    ///     cov = traj.covariance(itc.start_epoch)
    ///     ```
    ///
    /// References:
    ///     1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense Squadron, Space-Track.org, https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf
    ///     2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13), https://ntrs.nasa.gov/citations/20205011318
    ///     3. NASA CARA Analysis Tools, ``RIC2ECI.m``, https://github.com/nasa/CARA_Analysis_Tools
    ///     4. D. A. Vallado, "Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003, https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf
    fn to_trajectory_with_covariance_variant(&self, variant: PyOrbitRelativeFrameVariant) -> PyResult<PyOrbitalTrajectory> {
        self.inner
            .to_trajectory_with_covariance_variant(variant.variant)
            .map(|trajectory| PyOrbitalTrajectory { trajectory })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Build a message from a trajectory's stored samples.
    ///
    /// Args:
    ///     trajectory (OrbitTrajectory): Six-dimensional Cartesian trajectory.
    ///     header (ITCHeader): Header template; start, stop and step are filled from the samples.
    ///     covariance_variant (OrbitRelativeFrameVariant, optional): RTN convention for the covariance; ``INERTIAL`` by default.
    ///
    /// Returns:
    ///     ITC: The message.
    ///
    /// Raises:
    ///     BraheError: If the trajectory is empty, not six-dimensional Cartesian, a covariance is smaller than 6x6, or the frame router cannot reach the header's frames.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     itc = bh.ITC.from_file("test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt")
    ///     traj = itc.to_trajectory()
    ///     back = bh.ITC.from_trajectory(traj, bh.ITCHeader(ephemeris_source="brahe"))
    ///     ```
    ///
    /// References:
    ///     1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense Squadron, Space-Track.org, https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf
    ///     2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13), https://ntrs.nasa.gov/citations/20205011318
    ///     3. NASA CARA Analysis Tools, ``RIC2ECI.m``, https://github.com/nasa/CARA_Analysis_Tools
    ///     4. D. A. Vallado, "Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003, https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf
    #[staticmethod]
    #[pyo3(signature = (trajectory, header, covariance_variant=None))]
    fn from_trajectory(trajectory: PyRef<PyOrbitalTrajectory>, header: PyITCHeader, covariance_variant: Option<PyOrbitRelativeFrameVariant>) -> PyResult<Self> {
        let variant = covariance_variant.map(|v| v.variant).unwrap_or(frames::OrbitRelativeFrameVariant::Inertial);
        itc::ITC::from_trajectory_with_covariance_variant(&trajectory.trajectory, header.inner, variant)
            .map(|inner| Self { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Build a message from a trajectory with an explicit RTN covariance convention.
    ///
    /// A covariance is rotated from the trajectory frame into the header's
    /// covariance frame: EME2000 and ITRF through the state-transform Jacobian
    /// between the two frames, RTN through ICRF axes and the RTN frame of the
    /// sample's own state.
    ///
    /// Args:
    ///     trajectory (OrbitTrajectory): Six-dimensional Cartesian trajectory.
    ///     header (ITCHeader): Header template.
    ///     variant (OrbitRelativeFrameVariant): ``INERTIAL`` or ``ROTATING``.
    ///
    /// Returns:
    ///     ITC: The message.
    ///
    /// Raises:
    ///     BraheError: If the trajectory is empty, not six-dimensional Cartesian, a covariance is smaller than 6x6, or the frame router cannot reach the header's frames.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     itc = bh.ITC.from_file("test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt")
    ///     traj = itc.to_trajectory_with_covariance_variant(bh.OrbitRelativeFrameVariant.ROTATING)
    ///     back = bh.ITC.from_trajectory_with_covariance_variant(traj, bh.ITCHeader(), bh.OrbitRelativeFrameVariant.ROTATING)
    ///     ```
    ///
    /// References:
    ///     1. *Spaceflight Safety Handbook for Satellite Operators*, Version 1.7, 18th Space Defense Squadron, Space-Track.org, https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf
    ///     2. NASA Conjunction Assessment Risk Analysis (CARA), *Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13), https://ntrs.nasa.gov/citations/20205011318
    ///     3. NASA CARA Analysis Tools, ``RIC2ECI.m``, https://github.com/nasa/CARA_Analysis_Tools
    ///     4. D. A. Vallado, "Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003, https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf
    #[staticmethod]
    fn from_trajectory_with_covariance_variant(trajectory: PyRef<PyOrbitalTrajectory>, header: PyITCHeader, variant: PyOrbitRelativeFrameVariant) -> PyResult<Self> {
        itc::ITC::from_trajectory_with_covariance_variant(&trajectory.trajectory, header.inner, variant.variant)
            .map(|inner| Self { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }
}
