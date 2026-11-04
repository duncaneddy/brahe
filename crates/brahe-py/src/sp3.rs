// Python bindings for the SP3 precise orbit file format.

/// Hash of a value, used for the `__hash__` of SP3 value classes.
fn sp3_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Convert a Rust error into the Python `BraheError`.
fn sp3_py_err(error: RustBraheError) -> PyErr {
    BraheError::new_err(error.to_string())
}

/// Parse a string that must hold exactly one character.
fn sp3_single_char(text: &str, what: &str) -> PyResult<char> {
    let mut chars = text.trim().chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(exceptions::PyValueError::new_err(format!(
            "{what} must be a single character, got '{text}'"
        ))),
    }
}

const SP3_ALL_SYSTEMS: [sp3::GNSSSystem; 8] = [
    sp3::GNSSSystem::GPS,
    sp3::GNSSSystem::GLONASS,
    sp3::GNSSSystem::Galileo,
    sp3::GNSSSystem::BeiDou,
    sp3::GNSSSystem::QZSS,
    sp3::GNSSSystem::IRNSS,
    sp3::GNSSSystem::SBAS,
    sp3::GNSSSystem::LEO,
];

/// SP3 format version.
///
/// Versions a through d are described by the NGS SP3 documentation and the
/// IGS format specifications.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.SP3Version.from_char("D") == bh.SP3Version.D
///     assert str(bh.SP3Version.D) == "d"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3Version")]
#[derive(Clone, PartialEq)]
pub struct PySP3Version {
    pub(crate) inner: sp3::SP3Version,
}

#[pymethods]
impl PySP3Version {
    /// SP3-a, GPS satellites only.
    #[classattr]
    #[allow(non_snake_case)]
    fn A() -> Self {
        Self { inner: sp3::SP3Version::A }
    }

    /// SP3-b, adds multi-GNSS satellite identifiers.
    #[classattr]
    #[allow(non_snake_case)]
    fn B() -> Self {
        Self { inner: sp3::SP3Version::B }
    }

    /// SP3-c, adds clock event flags and correlation records.
    #[classattr]
    #[allow(non_snake_case)]
    fn C() -> Self {
        Self { inner: sp3::SP3Version::C }
    }

    /// SP3-d, adds extended satellite counts and long comments.
    #[classattr]
    #[allow(non_snake_case)]
    fn D() -> Self {
        Self { inner: sp3::SP3Version::D }
    }

    /// Parse a version letter, ignoring case.
    ///
    /// Args:
    ///     c (str): Version letter, one of ``a``, ``b``, ``c``, ``d`` in either case.
    ///
    /// Returns:
    ///     SP3Version: The version.
    ///
    /// Raises:
    ///     BraheError: If the letter is not a known version.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3Version.from_char("c") == bh.SP3Version.C
    ///     ```
    #[staticmethod]
    fn from_char(c: &str) -> PyResult<Self> {
        let c = sp3_single_char(c, "version letter")?;
        sp3::SP3Version::from_char(c)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Lowercase version letter as written in the first header line.
    ///
    /// Returns:
    ///     str: One of ``a``, ``b``, ``c``, ``d``.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3Version.B.as_char() == "b"
    ///     ```
    fn as_char(&self) -> String {
        self.inner.as_char().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SP3Version.{}", self.inner.as_char().to_ascii_uppercase())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        sp3_hash(&self.inner)
    }
}

/// Satellite navigation system of an SP3 satellite identifier.
///
/// Each system has a one-letter code used as the first character of the
/// satellite identifier (``G05``, ``R12``, ...).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.GNSSSystem.from_code("E") == bh.GNSSSystem.GALILEO
///     assert bh.GNSSSystem.BEIDOU.code == "C"
///     assert str(bh.GNSSSystem.GLONASS) == "GLONASS"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "GNSSSystem")]
#[derive(Clone, PartialEq)]
pub struct PyGNSSSystem {
    pub(crate) inner: sp3::GNSSSystem,
}

#[pymethods]
impl PyGNSSSystem {
    /// Global Positioning System (code ``G``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GPS() -> Self {
        Self { inner: sp3::GNSSSystem::GPS }
    }

    /// GLONASS (code ``R``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GLONASS() -> Self {
        Self { inner: sp3::GNSSSystem::GLONASS }
    }

    /// Galileo (code ``E``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GALILEO() -> Self {
        Self { inner: sp3::GNSSSystem::Galileo }
    }

    /// BeiDou (code ``C``).
    #[classattr]
    #[allow(non_snake_case)]
    fn BEIDOU() -> Self {
        Self { inner: sp3::GNSSSystem::BeiDou }
    }

    /// Quasi-Zenith Satellite System (code ``J``).
    #[classattr]
    #[allow(non_snake_case)]
    fn QZSS() -> Self {
        Self { inner: sp3::GNSSSystem::QZSS }
    }

    /// Indian Regional Navigation Satellite System (code ``I``).
    #[classattr]
    #[allow(non_snake_case)]
    fn IRNSS() -> Self {
        Self { inner: sp3::GNSSSystem::IRNSS }
    }

    /// Satellite-based augmentation systems (code ``S``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SBAS() -> Self {
        Self { inner: sp3::GNSSSystem::SBAS }
    }

    /// Low Earth orbiters (code ``L``).
    #[classattr]
    #[allow(non_snake_case)]
    fn LEO() -> Self {
        Self { inner: sp3::GNSSSystem::LEO }
    }

    /// Parse a system code letter.
    ///
    /// Args:
    ///     code (str): One of ``G``, ``R``, ``E``, ``C``, ``J``, ``I``, ``S``, ``L``.
    ///
    /// Returns:
    ///     GNSSSystem: The system.
    ///
    /// Raises:
    ///     BraheError: If the code is not a known system letter.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.GNSSSystem.from_code("R") == bh.GNSSSystem.GLONASS
    ///     ```
    #[staticmethod]
    fn from_code(code: &str) -> PyResult<Self> {
        let c = sp3_single_char(code, "system code")?;
        sp3::GNSSSystem::from_code(c)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// One-letter system code used in satellite identifiers.
    ///
    /// Returns:
    ///     str: The code letter.
    #[getter]
    fn code(&self) -> String {
        self.inner.code().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        let name = match self.inner {
            sp3::GNSSSystem::GPS => "GPS",
            sp3::GNSSSystem::GLONASS => "GLONASS",
            sp3::GNSSSystem::Galileo => "GALILEO",
            sp3::GNSSSystem::BeiDou => "BEIDOU",
            sp3::GNSSSystem::QZSS => "QZSS",
            sp3::GNSSSystem::IRNSS => "IRNSS",
            sp3::GNSSSystem::SBAS => "SBAS",
            sp3::GNSSSystem::LEO => "LEO",
        };
        format!("GNSSSystem.{}", name)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        sp3_hash(&self.inner)
    }
}

/// Satellite system covered by an SP3 file.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.SP3FileType.from_code("M") == bh.SP3FileType.MIXED
///     assert bh.SP3FileType.GPS.code == "G"
///     assert str(bh.SP3FileType.MIXED) == "Mixed"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3FileType")]
#[derive(Clone, PartialEq)]
pub struct PySP3FileType {
    pub(crate) inner: sp3::SP3FileType,
}

#[pymethods]
impl PySP3FileType {
    /// GPS satellites only (code ``G``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GPS() -> Self {
        Self { inner: sp3::SP3FileType::GPS }
    }

    /// GLONASS satellites only (code ``R``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GLONASS() -> Self {
        Self { inner: sp3::SP3FileType::GLONASS }
    }

    /// Galileo satellites only (code ``E``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GALILEO() -> Self {
        Self { inner: sp3::SP3FileType::Galileo }
    }

    /// BeiDou satellites only (code ``C``).
    #[classattr]
    #[allow(non_snake_case)]
    fn BEIDOU() -> Self {
        Self { inner: sp3::SP3FileType::BeiDou }
    }

    /// QZSS satellites only (code ``J``).
    #[classattr]
    #[allow(non_snake_case)]
    fn QZSS() -> Self {
        Self { inner: sp3::SP3FileType::QZSS }
    }

    /// IRNSS satellites only (code ``I``).
    #[classattr]
    #[allow(non_snake_case)]
    fn IRNSS() -> Self {
        Self { inner: sp3::SP3FileType::IRNSS }
    }

    /// SBAS satellites only (code ``S``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SBAS() -> Self {
        Self { inner: sp3::SP3FileType::SBAS }
    }

    /// Low Earth orbiters only (code ``L``).
    #[classattr]
    #[allow(non_snake_case)]
    fn LEO() -> Self {
        Self { inner: sp3::SP3FileType::LEO }
    }

    /// More than one system (code ``M``).
    #[classattr]
    #[allow(non_snake_case)]
    fn MIXED() -> Self {
        Self { inner: sp3::SP3FileType::Mixed }
    }

    /// Parse a file type code letter.
    ///
    /// Args:
    ///     code (str): One of ``G``, ``R``, ``E``, ``C``, ``J``, ``I``, ``S``, ``L``, ``M``.
    ///
    /// Returns:
    ///     SP3FileType: The file type.
    ///
    /// Raises:
    ///     BraheError: If the code is not a known file type letter.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3FileType.from_code("E") == bh.SP3FileType.GALILEO
    ///     ```
    #[staticmethod]
    fn from_code(code: &str) -> PyResult<Self> {
        let c = sp3_single_char(code, "file type code")?;
        sp3::SP3FileType::from_code(c)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// One-letter file type code written in the header.
    ///
    /// Returns:
    ///     str: The code letter.
    #[getter]
    fn code(&self) -> String {
        self.inner.code().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        let name = match self.inner {
            sp3::SP3FileType::GPS => "GPS",
            sp3::SP3FileType::GLONASS => "GLONASS",
            sp3::SP3FileType::Galileo => "GALILEO",
            sp3::SP3FileType::BeiDou => "BEIDOU",
            sp3::SP3FileType::QZSS => "QZSS",
            sp3::SP3FileType::IRNSS => "IRNSS",
            sp3::SP3FileType::SBAS => "SBAS",
            sp3::SP3FileType::LEO => "LEO",
            sp3::SP3FileType::Mixed => "MIXED",
        };
        format!("SP3FileType.{}", name)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        sp3_hash(&self.inner)
    }
}

/// Time system of the epochs in an SP3 file.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.SP3TimeSystem.parse("GAL") == bh.SP3TimeSystem.GAL
///     assert bh.SP3TimeSystem.GAL.to_time_system() == bh.TimeSystem.GST
///     assert bh.SP3TimeSystem.GLO.to_time_system() is None
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3TimeSystem")]
#[derive(Clone, PartialEq)]
pub struct PySP3TimeSystem {
    pub(crate) inner: sp3::SP3TimeSystem,
}

#[pymethods]
impl PySP3TimeSystem {
    /// GPS time.
    #[classattr]
    #[allow(non_snake_case)]
    fn GPS() -> Self {
        Self { inner: sp3::SP3TimeSystem::GPS }
    }

    /// GLONASS time.
    #[classattr]
    #[allow(non_snake_case)]
    fn GLO() -> Self {
        Self { inner: sp3::SP3TimeSystem::GLO }
    }

    /// Galileo System Time.
    #[classattr]
    #[allow(non_snake_case)]
    fn GAL() -> Self {
        Self { inner: sp3::SP3TimeSystem::GAL }
    }

    /// BeiDou Time.
    #[classattr]
    #[allow(non_snake_case)]
    fn BDT() -> Self {
        Self { inner: sp3::SP3TimeSystem::BDT }
    }

    /// International Atomic Time.
    #[classattr]
    #[allow(non_snake_case)]
    fn TAI() -> Self {
        Self { inner: sp3::SP3TimeSystem::TAI }
    }

    /// Coordinated Universal Time.
    #[classattr]
    #[allow(non_snake_case)]
    fn UTC() -> Self {
        Self { inner: sp3::SP3TimeSystem::UTC }
    }

    /// QZSS time.
    #[classattr]
    #[allow(non_snake_case)]
    fn QZS() -> Self {
        Self { inner: sp3::SP3TimeSystem::QZS }
    }

    /// IRNSS time.
    #[classattr]
    #[allow(non_snake_case)]
    fn IRN() -> Self {
        Self { inner: sp3::SP3TimeSystem::IRN }
    }

    /// Parse a time system label.
    ///
    /// Surrounding whitespace is ignored. The ``ccc`` placeholder that
    /// SP3-a and SP3-b files carry maps to GPS time.
    ///
    /// Args:
    ///     s (str): Time system label, for example ``GPS`` or ``UTC``.
    ///
    /// Returns:
    ///     SP3TimeSystem: The time system.
    ///
    /// Raises:
    ///     BraheError: If the label is not a known time system.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3TimeSystem.parse(" UTC ") == bh.SP3TimeSystem.UTC
    ///     ```
    #[staticmethod]
    fn parse(s: &str) -> PyResult<Self> {
        sp3::SP3TimeSystem::parse(s)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Convert a Brahe time system to the SP3 label.
    ///
    /// Args:
    ///     time_system (TimeSystem): Brahe time system.
    ///
    /// Returns:
    ///     SP3TimeSystem | None: The SP3 time system, or None if SP3 has no label for it.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3TimeSystem.from_time_system(bh.TimeSystem.GST) == bh.SP3TimeSystem.GAL
    ///     assert bh.SP3TimeSystem.from_time_system(bh.TimeSystem.TT) is None
    ///     ```
    #[staticmethod]
    fn from_time_system(time_system: PyTimeSystem) -> Option<Self> {
        sp3::SP3TimeSystem::from_time_system(time_system.ts).map(|inner| Self { inner })
    }

    /// Brahe time system equivalent to this SP3 time system.
    ///
    /// Returns:
    ///     TimeSystem | None: The Brahe time system, or None for GLO, QZS and IRN, which Brahe does not implement.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3TimeSystem.BDT.to_time_system() == bh.TimeSystem.BDT
    ///     ```
    fn to_time_system(&self) -> Option<PyTimeSystem> {
        self.inner.to_time_system().map(|ts| PyTimeSystem { ts })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SP3TimeSystem.{}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        sp3_hash(&self.inner)
    }
}

/// Orbit type label of an SP3 file.
///
/// The four standard labels are class attributes. Any other label found in
/// a file is preserved verbatim by ``parse``.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.SP3OrbitType.parse("FIT") == bh.SP3OrbitType.FIT
///     assert str(bh.SP3OrbitType.parse("BHN")) == "BHN"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3OrbitType")]
#[derive(Clone, PartialEq)]
pub struct PySP3OrbitType {
    pub(crate) inner: sp3::SP3OrbitType,
}

#[pymethods]
impl PySP3OrbitType {
    /// Fitted orbit.
    #[classattr]
    #[allow(non_snake_case)]
    fn FIT() -> Self {
        Self { inner: sp3::SP3OrbitType::FIT }
    }

    /// Extrapolated or predicted orbit.
    #[classattr]
    #[allow(non_snake_case)]
    fn EXT() -> Self {
        Self { inner: sp3::SP3OrbitType::EXT }
    }

    /// Broadcast orbit.
    #[classattr]
    #[allow(non_snake_case)]
    fn BCT() -> Self {
        Self { inner: sp3::SP3OrbitType::BCT }
    }

    /// Fit after Helmert transformation.
    #[classattr]
    #[allow(non_snake_case)]
    fn HLM() -> Self {
        Self { inner: sp3::SP3OrbitType::HLM }
    }

    /// Parse an orbit type label.
    ///
    /// Surrounding whitespace is ignored. Labels other than the four
    /// standard ones are kept as written.
    ///
    /// Args:
    ///     s (str): Orbit type label, for example ``FIT`` or ``EXT``.
    ///
    /// Returns:
    ///     SP3OrbitType: The orbit type.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3OrbitType.parse(" EXT ") == bh.SP3OrbitType.EXT
    ///     assert bh.SP3OrbitType.parse("XYZ") != bh.SP3OrbitType.EXT
    ///     ```
    #[staticmethod]
    fn parse(s: &str) -> Self {
        Self { inner: sp3::SP3OrbitType::parse(s) }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SP3OrbitType.parse('{}')", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        sp3_hash(&self.inner)
    }
}

/// Extract an `sp3::GNSSSystem` from a `GNSSSystem`, a one-letter code, or a system name.
fn extract_sp3_gnss_system(obj: &Bound<'_, PyAny>) -> PyResult<sp3::GNSSSystem> {
    if let Ok(system) = obj.extract::<PyGNSSSystem>() {
        return Ok(system.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        let text = text.trim();
        if text.chars().count() == 1 {
            return sp3::GNSSSystem::from_code(text.chars().next().unwrap()).map_err(sp3_py_err);
        }
        if let Some(system) = SP3_ALL_SYSTEMS
            .iter()
            .find(|s| s.to_string().eq_ignore_ascii_case(text))
        {
            return Ok(*system);
        }
        return Err(BraheError::new_err(format!(
            "SP3: unknown satellite system '{text}'"
        )));
    }
    Err(exceptions::PyTypeError::new_err(
        "expected a GNSSSystem or a system code string",
    ))
}

/// Extract an `sp3::SP3Version` from an `SP3Version` or its version letter.
fn extract_sp3_version(obj: &Bound<'_, PyAny>) -> PyResult<sp3::SP3Version> {
    if let Ok(version) = obj.extract::<PySP3Version>() {
        return Ok(version.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return sp3::SP3Version::from_char(sp3_single_char(&text, "version letter")?)
            .map_err(sp3_py_err);
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an SP3Version or a version letter string",
    ))
}

/// Extract an `sp3::SP3FileType` from an `SP3FileType` or its code letter.
fn extract_sp3_file_type(obj: &Bound<'_, PyAny>) -> PyResult<sp3::SP3FileType> {
    if let Ok(file_type) = obj.extract::<PySP3FileType>() {
        return Ok(file_type.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return sp3::SP3FileType::from_code(sp3_single_char(&text, "file type code")?)
            .map_err(sp3_py_err);
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an SP3FileType or a file type code string",
    ))
}

/// Extract an `sp3::SP3TimeSystem` from an `SP3TimeSystem` or its label.
fn extract_sp3_time_system(obj: &Bound<'_, PyAny>) -> PyResult<sp3::SP3TimeSystem> {
    if let Ok(time_system) = obj.extract::<PySP3TimeSystem>() {
        return Ok(time_system.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return sp3::SP3TimeSystem::parse(&text).map_err(sp3_py_err);
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an SP3TimeSystem or a time system label string",
    ))
}

/// Extract an `sp3::SP3OrbitType` from an `SP3OrbitType` or its label.
fn extract_sp3_orbit_type(obj: &Bound<'_, PyAny>) -> PyResult<sp3::SP3OrbitType> {
    if let Ok(orbit_type) = obj.extract::<PySP3OrbitType>() {
        return Ok(orbit_type.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return Ok(sp3::SP3OrbitType::parse(&text));
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an SP3OrbitType or an orbit type label string",
    ))
}

/// Extract an `sp3::SP3SatelliteId` from an `SP3SatelliteId` or an identifier string such as `G05`.
fn extract_sp3_satellite_id(obj: &Bound<'_, PyAny>) -> PyResult<sp3::SP3SatelliteId> {
    if let Ok(id) = obj.extract::<PySP3SatelliteId>() {
        return Ok(id.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return sp3::SP3SatelliteId::parse(&text).map_err(sp3_py_err);
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an SP3SatelliteId or a satellite identifier string such as 'G05'",
    ))
}

/// Build the header options of a new SP3 file from Python arguments.
fn sp3_write_options(
    agency: &str,
    time_system: &Bound<'_, PyAny>,
    coordinate_system: &str,
    orbit_type: &Bound<'_, PyAny>,
    data_used: &str,
    include_velocity: bool,
    comments: Option<Vec<String>>,
) -> PyResult<sp3::SP3WriteOptions> {
    Ok(sp3::SP3WriteOptions::new()
        .with_agency(agency)
        .with_time_system(extract_sp3_time_system(time_system)?)
        .with_coordinate_system(coordinate_system)
        .with_orbit_type(extract_sp3_orbit_type(orbit_type)?)
        .with_data_used(data_used)
        .with_velocity(include_velocity)
        .with_comments(comments.unwrap_or_default()))
}

/// Extract three finite-or-not floats from a numpy array or sequence.
fn extract_sp3_vector3(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = match obj.extract::<PyReadonlyArray1<f64>>() {
        Ok(array) => array.as_array().to_vec(),
        Err(_) => obj.extract::<Vec<f64>>()?,
    };
    <[f64; 3]>::try_from(values).map_err(|v| {
        exceptions::PyValueError::new_err(format!(
            "{name} must have three components, got {}",
            v.len()
        ))
    })
}

/// Extract three optional floats from a sequence.
fn extract_sp3_optional_triple(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<[Option<f64>; 3]> {
    let values: Vec<Option<f64>> = obj.extract()?;
    <[Option<f64>; 3]>::try_from(values).map_err(|v| {
        exceptions::PyValueError::new_err(format!(
            "{name} must have three entries, got {}",
            v.len()
        ))
    })
}

/// Treat NaN values as absent, matching the NaN-for-None convention of the array accessors.
fn sp3_nan_to_none(values: Vec<Option<f64>>) -> Vec<Option<f64>> {
    values
        .into_iter()
        .map(|v| v.filter(|x| !x.is_nan()))
        .collect()
}

fn sp3_vector_to_numpy<'py>(py: Python<'py>, v: [f64; 3]) -> Bound<'py, PyArray<f64, Ix1>> {
    v.to_vec().into_pyarray(py)
}

/// Identifier of a satellite in an SP3 file: a system and a number.
///
/// Identifiers print as the system code followed by a two-digit number
/// (``G05``) and order by system, then number.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     sat = bh.SP3SatelliteId(bh.GNSSSystem.GPS, 5)
///     assert str(sat) == "G05"
///     assert bh.SP3SatelliteId.parse("G 5") == sat
///     assert bh.SP3SatelliteId("G", 5) == sat
///     assert sat == "G05"
///     assert hash(sat) == hash("G05")
///     ```
///
/// An identifier also equals any string that ``parse`` reads as the same
/// identifier. Its hash equals the hash of the canonical string ``str(sat)``
/// only, so a non-canonical spelling such as ``"G 5"`` compares equal but
/// hashes differently.
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3SatelliteId")]
#[derive(Clone, PartialEq)]
pub struct PySP3SatelliteId {
    pub(crate) inner: sp3::SP3SatelliteId,
}

#[pymethods]
impl PySP3SatelliteId {
    /// Create a satellite identifier.
    ///
    /// Args:
    ///     system (GNSSSystem | str): Satellite system, or its one-letter code or name.
    ///     number (int): Satellite number within the system, 1 to 99 inclusive.
    ///
    /// Returns:
    ///     SP3SatelliteId: The identifier.
    ///
    /// Raises:
    ///     BraheError: If the system is unknown or the number is outside 1 to 99.
    #[new]
    fn new(system: &Bound<'_, PyAny>, number: i64) -> PyResult<Self> {
        let system = extract_sp3_gnss_system(system)?;
        let number = u16::try_from(number).map_err(|_| {
            BraheError::new_err(format!("SP3: satellite number {number} out of range 1..=99"))
        })?;
        sp3::SP3SatelliteId::new(system, number)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Parse an identifier as written in an SP3 file.
    ///
    /// The identifier has three characters: a system letter (a space means
    /// GPS) and a two-digit number, which may be space padded.
    ///
    /// Args:
    ///     s (str): Identifier such as ``G05``, ``G 5`` or ``  5``.
    ///
    /// Returns:
    ///     SP3SatelliteId: The identifier.
    ///
    /// Raises:
    ///     BraheError: If the text is not a valid identifier.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.SP3SatelliteId.parse("R12").system == bh.GNSSSystem.GLONASS
    ///     ```
    #[staticmethod]
    fn parse(s: &str) -> PyResult<Self> {
        sp3::SP3SatelliteId::parse(s)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Satellite system.
    ///
    /// Returns:
    ///     GNSSSystem: The system.
    #[getter]
    fn system(&self) -> PyGNSSSystem {
        PyGNSSSystem { inner: self.inner.system }
    }

    /// Satellite number within the system.
    ///
    /// Returns:
    ///     int: Number from 1 to 99.
    #[getter]
    fn number(&self) -> u16 {
        self.inner.number
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SP3SatelliteId('{}')", self.inner)
    }

    /// Compare with another `SP3SatelliteId`, or for `==` and `!=` with an
    /// identifier string read by `SP3SatelliteId.parse`. A string that does
    /// not parse is unequal.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let other_id = if let Ok(id) = other.extract::<PySP3SatelliteId>() {
            Some(id.inner)
        } else if let Ok(text) = other.extract::<String>() {
            if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
                return Ok(py.NotImplemented());
            }
            sp3::SP3SatelliteId::parse(&text).ok()
        } else {
            return Ok(py.NotImplemented());
        };
        let result = match other_id {
            Some(id) => op.matches(self.inner.cmp(&id)),
            None => matches!(op, CompareOp::Ne),
        };
        result.into_py_any(py)
    }

    /// Hash of the canonical identifier string, so `hash(sat) == hash(str(sat))`.
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        PyString::new(py, &self.inner.to_string()).hash()
    }
}

/// Standard deviations and correlations of an SP3-c/d correlation record.
///
/// ``EP`` records carry position (m) and clock (s) values and ``EV`` records
/// carry velocity (m/s) and clock rate (s/s) values. A correlation
/// coefficient whose column is blank in the file is unknown and is None,
/// distinct from an explicit 0. Properties return copies; assign the object
/// back to its owner to change it.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     corr = bh.SP3Correlation(std=(0.01, 0.01, 0.02), clock_std=1e-11, xy=0.1)
///     assert corr.std[2] == 0.02
///     assert corr.xy == 0.1
///     assert corr.yz is None
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3Correlation")]
#[derive(Clone)]
pub struct PySP3Correlation {
    pub(crate) inner: sp3::SP3Correlation,
}

#[pymethods]
impl PySP3Correlation {
    /// Create a correlation record.
    ///
    /// Args:
    ///     std (Sequence[float | None], optional): Standard deviations of x, y, z; defaults to all unknown.
    ///     clock_std (float, optional): Standard deviation of the clock.
    ///     xy (float, optional): Correlation between x and y; defaults to None (unknown).
    ///     xz (float, optional): Correlation between x and z; defaults to None (unknown).
    ///     xc (float, optional): Correlation between x and the clock; defaults to None (unknown).
    ///     yz (float, optional): Correlation between y and z; defaults to None (unknown).
    ///     yc (float, optional): Correlation between y and the clock; defaults to None (unknown).
    ///     zc (float, optional): Correlation between z and the clock; defaults to None (unknown).
    ///
    /// Returns:
    ///     SP3Correlation: The correlation record.
    #[new]
    #[pyo3(signature = (std=None, clock_std=None, xy=None, xz=None, xc=None, yz=None, yc=None, zc=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        std: Option<&Bound<'_, PyAny>>,
        clock_std: Option<f64>,
        xy: Option<f64>,
        xz: Option<f64>,
        xc: Option<f64>,
        yz: Option<f64>,
        yc: Option<f64>,
        zc: Option<f64>,
    ) -> PyResult<Self> {
        let std = match std {
            Some(obj) => extract_sp3_optional_triple(obj, "std")?,
            None => [None; 3],
        };
        Ok(Self {
            inner: sp3::SP3Correlation { std, clock_std, xy, xz, xc, yz, yc, zc },
        })
    }

    /// Standard deviations of x, y, z.
    ///
    /// Returns:
    ///     tuple[float | None, float | None, float | None]: Standard deviations in m (EP) or m/s (EV); None where unknown.
    #[getter]
    fn std(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let s = self.inner.std;
        (s[0], s[1], s[2])
    }

    #[setter]
    fn set_std(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.std = extract_sp3_optional_triple(value, "std")?;
        Ok(())
    }

    /// Standard deviation of the clock.
    ///
    /// Returns:
    ///     float | None: Standard deviation in s (EP) or s/s (EV); None where unknown.
    #[getter]
    fn clock_std(&self) -> Option<f64> {
        self.inner.clock_std
    }

    #[setter]
    fn set_clock_std(&mut self, value: Option<f64>) {
        self.inner.clock_std = value;
    }

    /// Correlation between x and y.
    ///
    /// Returns:
    ///     float | None: Correlation coefficient; None if unknown.
    #[getter]
    fn xy(&self) -> Option<f64> {
        self.inner.xy
    }

    #[setter]
    fn set_xy(&mut self, value: Option<f64>) {
        self.inner.xy = value;
    }

    /// Correlation between x and z.
    ///
    /// Returns:
    ///     float | None: Correlation coefficient; None if unknown.
    #[getter]
    fn xz(&self) -> Option<f64> {
        self.inner.xz
    }

    #[setter]
    fn set_xz(&mut self, value: Option<f64>) {
        self.inner.xz = value;
    }

    /// Correlation between x and the clock.
    ///
    /// Returns:
    ///     float | None: Correlation coefficient; None if unknown.
    #[getter]
    fn xc(&self) -> Option<f64> {
        self.inner.xc
    }

    #[setter]
    fn set_xc(&mut self, value: Option<f64>) {
        self.inner.xc = value;
    }

    /// Correlation between y and z.
    ///
    /// Returns:
    ///     float | None: Correlation coefficient; None if unknown.
    #[getter]
    fn yz(&self) -> Option<f64> {
        self.inner.yz
    }

    #[setter]
    fn set_yz(&mut self, value: Option<f64>) {
        self.inner.yz = value;
    }

    /// Correlation between y and the clock.
    ///
    /// Returns:
    ///     float | None: Correlation coefficient; None if unknown.
    #[getter]
    fn yc(&self) -> Option<f64> {
        self.inner.yc
    }

    #[setter]
    fn set_yc(&mut self, value: Option<f64>) {
        self.inner.yc = value;
    }

    /// Correlation between z and the clock.
    ///
    /// Returns:
    ///     float | None: Correlation coefficient; None if unknown.
    #[getter]
    fn zc(&self) -> Option<f64> {
        self.inner.zc
    }

    #[setter]
    fn set_zc(&mut self, value: Option<f64>) {
        self.inner.zc = value;
    }

    fn __repr__(&self) -> String {
        format!("SP3Correlation(std={:?}, clock_std={:?})", self.inner.std, self.inner.clock_std)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// One satellite state at one epoch of an SP3 file.
///
/// All quantities are SI: meters, meters per second, seconds and seconds per
/// second. Optional fields are None when the file leaves them blank or does
/// not carry them. Properties return copies; to change a record held by an
/// ``SP3``, modify a copy and pass it to ``SP3.set_record``.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     record = bh.SP3Record(
///         np.array([15.0e6, 20.0e6, 5.0e6]),
///         velocity=np.array([1000.0, -2000.0, 500.0]),
///         clock=2.5e-5,
///     )
///     assert record.clock == 2.5e-5
///     assert record.clock_rate is None
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3Record")]
#[derive(Clone)]
pub struct PySP3Record {
    pub(crate) inner: sp3::SP3Record,
}

#[pymethods]
impl PySP3Record {
    /// Create a record.
    ///
    /// Args:
    ///     position (numpy.ndarray): Position in the file's coordinate system, meters, shape (3,).
    ///     clock (float, optional): Clock offset, seconds.
    ///     velocity (numpy.ndarray, optional): Velocity, meters per second, shape (3,).
    ///     clock_rate (float, optional): Clock rate of change, seconds per second.
    ///     position_std (Sequence[float | None], optional): Standard deviations of the position, meters.
    ///     clock_std (float, optional): Standard deviation of the clock, seconds.
    ///     velocity_std (Sequence[float | None], optional): Standard deviations of the velocity, meters per second.
    ///     clock_rate_std (float, optional): Standard deviation of the clock rate, seconds per second.
    ///     clock_event (bool, optional): Whether a clock discontinuity occurred; defaults to False.
    ///     clock_predicted (bool, optional): Whether the clock is predicted; defaults to False.
    ///     maneuver (bool, optional): Whether a maneuver occurred; defaults to False.
    ///     orbit_predicted (bool, optional): Whether the orbit is predicted; defaults to False.
    ///     position_correlation (SP3Correlation, optional): Position and clock correlation record.
    ///     velocity_correlation (SP3Correlation, optional): Velocity and clock rate correlation record.
    ///
    /// Returns:
    ///     SP3Record: The record.
    ///
    /// Raises:
    ///     ValueError: If a vector argument does not have three components.
    #[new]
    #[pyo3(signature = (
        position, *,
        clock=None, velocity=None, clock_rate=None,
        position_std=None, clock_std=None, velocity_std=None, clock_rate_std=None,
        clock_event=false, clock_predicted=false, maneuver=false, orbit_predicted=false,
        position_correlation=None, velocity_correlation=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        position: &Bound<'_, PyAny>,
        clock: Option<f64>,
        velocity: Option<&Bound<'_, PyAny>>,
        clock_rate: Option<f64>,
        position_std: Option<&Bound<'_, PyAny>>,
        clock_std: Option<f64>,
        velocity_std: Option<&Bound<'_, PyAny>>,
        clock_rate_std: Option<f64>,
        clock_event: bool,
        clock_predicted: bool,
        maneuver: bool,
        orbit_predicted: bool,
        position_correlation: Option<PySP3Correlation>,
        velocity_correlation: Option<PySP3Correlation>,
    ) -> PyResult<Self> {
        let mut inner = sp3::SP3Record::new(extract_sp3_vector3(position, "position")?);
        inner.clock = clock;
        inner.velocity = velocity
            .map(|v| extract_sp3_vector3(v, "velocity"))
            .transpose()?;
        inner.clock_rate = clock_rate;
        if let Some(std) = position_std {
            inner.position_std = extract_sp3_optional_triple(std, "position_std")?;
        }
        inner.clock_std = clock_std;
        if let Some(std) = velocity_std {
            inner.velocity_std = extract_sp3_optional_triple(std, "velocity_std")?;
        }
        inner.clock_rate_std = clock_rate_std;
        inner.clock_event = clock_event;
        inner.clock_predicted = clock_predicted;
        inner.maneuver = maneuver;
        inner.orbit_predicted = orbit_predicted;
        inner.position_correlation = position_correlation.map(|c| c.inner);
        inner.velocity_correlation = velocity_correlation.map(|c| c.inner);
        Ok(Self { inner })
    }

    /// Position in the file's coordinate system.
    ///
    /// Returns:
    ///     numpy.ndarray: Position, meters, shape (3,).
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        sp3_vector_to_numpy(py, self.inner.position)
    }

    #[setter]
    fn set_position(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.position = extract_sp3_vector3(value, "position")?;
        Ok(())
    }

    /// Clock offset.
    ///
    /// Returns:
    ///     float | None: Clock offset, seconds; None if absent.
    #[getter]
    fn clock(&self) -> Option<f64> {
        self.inner.clock
    }

    #[setter]
    fn set_clock(&mut self, value: Option<f64>) {
        self.inner.clock = value;
    }

    /// Standard deviations of the position.
    ///
    /// Returns:
    ///     tuple[float | None, float | None, float | None]: Standard deviations, meters; None where unknown.
    #[getter]
    fn position_std(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let s = self.inner.position_std;
        (s[0], s[1], s[2])
    }

    #[setter]
    fn set_position_std(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.position_std = extract_sp3_optional_triple(value, "position_std")?;
        Ok(())
    }

    /// Standard deviation of the clock.
    ///
    /// Returns:
    ///     float | None: Standard deviation, seconds; None where unknown.
    #[getter]
    fn clock_std(&self) -> Option<f64> {
        self.inner.clock_std
    }

    #[setter]
    fn set_clock_std(&mut self, value: Option<f64>) {
        self.inner.clock_std = value;
    }

    /// Velocity in the file's coordinate system.
    ///
    /// Returns:
    ///     numpy.ndarray | None: Velocity, meters per second, shape (3,); None if absent.
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray<f64, Ix1>>> {
        self.inner.velocity.map(|v| sp3_vector_to_numpy(py, v))
    }

    #[setter]
    fn set_velocity(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.velocity = value
            .map(|v| extract_sp3_vector3(v, "velocity"))
            .transpose()?;
        Ok(())
    }

    /// Clock rate of change.
    ///
    /// Returns:
    ///     float | None: Clock rate, seconds per second; None if absent.
    #[getter]
    fn clock_rate(&self) -> Option<f64> {
        self.inner.clock_rate
    }

    #[setter]
    fn set_clock_rate(&mut self, value: Option<f64>) {
        self.inner.clock_rate = value;
    }

    /// Standard deviations of the velocity.
    ///
    /// Returns:
    ///     tuple[float | None, float | None, float | None]: Standard deviations, meters per second; None where unknown.
    #[getter]
    fn velocity_std(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let s = self.inner.velocity_std;
        (s[0], s[1], s[2])
    }

    #[setter]
    fn set_velocity_std(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.velocity_std = extract_sp3_optional_triple(value, "velocity_std")?;
        Ok(())
    }

    /// Standard deviation of the clock rate.
    ///
    /// Returns:
    ///     float | None: Standard deviation, seconds per second; None where unknown.
    #[getter]
    fn clock_rate_std(&self) -> Option<f64> {
        self.inner.clock_rate_std
    }

    #[setter]
    fn set_clock_rate_std(&mut self, value: Option<f64>) {
        self.inner.clock_rate_std = value;
    }

    /// Clock event flag.
    ///
    /// Returns:
    ///     bool: True if a clock discontinuity occurred at this epoch.
    #[getter]
    fn clock_event(&self) -> bool {
        self.inner.clock_event
    }

    #[setter]
    fn set_clock_event(&mut self, value: bool) {
        self.inner.clock_event = value;
    }

    /// Clock prediction flag.
    ///
    /// Returns:
    ///     bool: True if the clock value is predicted.
    #[getter]
    fn clock_predicted(&self) -> bool {
        self.inner.clock_predicted
    }

    #[setter]
    fn set_clock_predicted(&mut self, value: bool) {
        self.inner.clock_predicted = value;
    }

    /// Maneuver flag.
    ///
    /// Returns:
    ///     bool: True if the satellite maneuvered since the previous epoch.
    #[getter]
    fn maneuver(&self) -> bool {
        self.inner.maneuver
    }

    #[setter]
    fn set_maneuver(&mut self, value: bool) {
        self.inner.maneuver = value;
    }

    /// Orbit prediction flag.
    ///
    /// Returns:
    ///     bool: True if the orbit is predicted.
    #[getter]
    fn orbit_predicted(&self) -> bool {
        self.inner.orbit_predicted
    }

    #[setter]
    fn set_orbit_predicted(&mut self, value: bool) {
        self.inner.orbit_predicted = value;
    }

    /// Position and clock correlation record (``EP``).
    ///
    /// Returns:
    ///     SP3Correlation | None: A copy of the record; None if absent.
    #[getter]
    fn position_correlation(&self) -> Option<PySP3Correlation> {
        self.inner
            .position_correlation
            .clone()
            .map(|inner| PySP3Correlation { inner })
    }

    #[setter]
    fn set_position_correlation(&mut self, value: Option<PySP3Correlation>) {
        self.inner.position_correlation = value.map(|c| c.inner);
    }

    /// Velocity and clock rate correlation record (``EV``).
    ///
    /// Returns:
    ///     SP3Correlation | None: A copy of the record; None if absent.
    #[getter]
    fn velocity_correlation(&self) -> Option<PySP3Correlation> {
        self.inner
            .velocity_correlation
            .clone()
            .map(|inner| PySP3Correlation { inner })
    }

    #[setter]
    fn set_velocity_correlation(&mut self, value: Option<PySP3Correlation>) {
        self.inner.velocity_correlation = value.map(|c| c.inner);
    }

    fn __repr__(&self) -> String {
        format!(
            "SP3Record(position={:?}, clock={:?}, velocity={:?}, clock_rate={:?})",
            self.inner.position, self.inner.clock, self.inner.velocity, self.inner.clock_rate
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Header of an SP3 file.
///
/// Mirrors the header fields of the file. ``SP3.header`` returns a copy;
/// modify the copy and assign it back with ``sp3.header = header``.
/// ``start_epoch``, ``gps_week``, ``seconds_of_week``, ``mjd`` and
/// ``fractional_day`` describe the first epoch and ``number_of_epochs`` the
/// epoch count; the writer recomputes them from the epoch table.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     sp3 = bh.SP3(["G01"], 900.0)
///     header = sp3.header
///     assert header.version == bh.SP3Version.D
///     assert header.agency == "BRHE"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3Header")]
#[derive(Clone)]
pub struct PySP3Header {
    pub(crate) inner: sp3::SP3Header,
}

#[pymethods]
impl PySP3Header {
    /// SP3 format version. Assignment accepts an ``SP3Version`` or its letter.
    ///
    /// Returns:
    ///     SP3Version: The version.
    #[getter]
    fn version(&self) -> PySP3Version {
        PySP3Version { inner: self.inner.version }
    }

    #[setter]
    fn set_version(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.version = extract_sp3_version(value)?;
        Ok(())
    }

    /// Whether the file carries velocity records.
    ///
    /// Returns:
    ///     bool: True if velocity records are present.
    #[getter]
    fn has_velocity(&self) -> bool {
        self.inner.has_velocity
    }

    #[setter]
    fn set_has_velocity(&mut self, value: bool) {
        self.inner.has_velocity = value;
    }

    /// First epoch of the file.
    ///
    /// Returns:
    ///     Epoch: The first epoch, in the file's time system.
    #[getter]
    fn start_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.inner.start_epoch }
    }

    #[setter]
    fn set_start_epoch(&mut self, value: PyEpoch) {
        self.inner.start_epoch = value.obj;
    }

    /// Number of epochs declared by the header.
    ///
    /// Returns:
    ///     int: Epoch count.
    #[getter]
    fn number_of_epochs(&self) -> usize {
        self.inner.number_of_epochs
    }

    #[setter]
    fn set_number_of_epochs(&mut self, value: usize) {
        self.inner.number_of_epochs = value;
    }

    /// Data used descriptor; at most 5 characters to be written.
    ///
    /// Returns:
    ///     str: Descriptor, for example ``ORBIT``.
    #[getter]
    fn data_used(&self) -> String {
        self.inner.data_used.clone()
    }

    #[setter]
    fn set_data_used(&mut self, value: String) {
        self.inner.data_used = value;
    }

    /// Coordinate system label; at most 5 characters to be written.
    ///
    /// Returns:
    ///     str: Label, for example ``IGS20``.
    #[getter]
    fn coordinate_system(&self) -> String {
        self.inner.coordinate_system.clone()
    }

    #[setter]
    fn set_coordinate_system(&mut self, value: String) {
        self.inner.coordinate_system = value;
    }

    /// Orbit type; at most 3 characters to be written.
    ///
    /// Returns:
    ///     SP3OrbitType: The orbit type.
    #[getter]
    fn orbit_type(&self) -> PySP3OrbitType {
        PySP3OrbitType { inner: self.inner.orbit_type.clone() }
    }

    #[setter]
    fn set_orbit_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.orbit_type = extract_sp3_orbit_type(value)?;
        Ok(())
    }

    /// Producing agency; at most 4 characters to be written.
    ///
    /// Returns:
    ///     str: Agency label.
    #[getter]
    fn agency(&self) -> String {
        self.inner.agency.clone()
    }

    #[setter]
    fn set_agency(&mut self, value: String) {
        self.inner.agency = value;
    }

    /// GPS week of the first epoch.
    ///
    /// Returns:
    ///     int: GPS week number.
    #[getter]
    fn gps_week(&self) -> u32 {
        self.inner.gps_week
    }

    #[setter]
    fn set_gps_week(&mut self, value: u32) {
        self.inner.gps_week = value;
    }

    /// Seconds of the GPS week of the first epoch.
    ///
    /// Returns:
    ///     float: Seconds of week.
    #[getter]
    fn seconds_of_week(&self) -> f64 {
        self.inner.seconds_of_week
    }

    #[setter]
    fn set_seconds_of_week(&mut self, value: f64) {
        self.inner.seconds_of_week = value;
    }

    /// Spacing of the epoch grid.
    ///
    /// Returns:
    ///     float: Epoch interval, seconds.
    #[getter]
    fn epoch_interval(&self) -> f64 {
        self.inner.epoch_interval
    }

    #[setter]
    fn set_epoch_interval(&mut self, value: f64) {
        self.inner.epoch_interval = value;
    }

    /// Modified Julian day of the first epoch.
    ///
    /// Returns:
    ///     int: Integer MJD.
    #[getter]
    fn mjd(&self) -> u32 {
        self.inner.mjd
    }

    #[setter]
    fn set_mjd(&mut self, value: u32) {
        self.inner.mjd = value;
    }

    /// Fraction of the day of the first epoch.
    ///
    /// Returns:
    ///     float: Fractional day in [0, 1).
    #[getter]
    fn fractional_day(&self) -> f64 {
        self.inner.fractional_day
    }

    #[setter]
    fn set_fractional_day(&mut self, value: f64) {
        self.inner.fractional_day = value;
    }

    /// Satellite system covered by the file. Assignment accepts an
    /// ``SP3FileType`` or its code letter.
    ///
    /// Returns:
    ///     SP3FileType: The file type.
    #[getter]
    fn file_type(&self) -> PySP3FileType {
        PySP3FileType { inner: self.inner.file_type }
    }

    #[setter]
    fn set_file_type(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.file_type = extract_sp3_file_type(value)?;
        Ok(())
    }

    /// Time system of the epochs.
    ///
    /// Returns:
    ///     SP3TimeSystem: The time system.
    #[getter]
    fn time_system(&self) -> PySP3TimeSystem {
        PySP3TimeSystem { inner: self.inner.time_system }
    }

    #[setter]
    fn set_time_system(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.time_system = extract_sp3_time_system(value)?;
        Ok(())
    }

    /// Base of the position and velocity standard deviation exponents.
    ///
    /// Returns:
    ///     float | None: Base in mm (or 1e-4 mm/s); None if the file declares none.
    #[getter]
    fn position_base(&self) -> Option<f64> {
        self.inner.position_base
    }

    #[setter]
    fn set_position_base(&mut self, value: Option<f64>) {
        self.inner.position_base = value;
    }

    /// Base of the clock and clock rate standard deviation exponents.
    ///
    /// Returns:
    ///     float | None: Base in ps (or 1e-4 ps/s); None if the file declares none.
    #[getter]
    fn clock_base(&self) -> Option<f64> {
        self.inner.clock_base
    }

    #[setter]
    fn set_clock_base(&mut self, value: Option<f64>) {
        self.inner.clock_base = value;
    }

    /// Accuracy of each satellite in header order.
    ///
    /// Returns:
    ///     list[float | None]: Accuracies in meters; None for an unknown accuracy.
    #[getter]
    fn accuracies(&self) -> Vec<Option<f64>> {
        self.inner.accuracies.clone()
    }

    #[setter]
    fn set_accuracies(&mut self, value: Vec<Option<f64>>) {
        self.inner.accuracies = value;
    }

    /// Comment lines.
    ///
    /// Returns:
    ///     list[str]: Comments in file order.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    #[setter]
    fn set_comments(&mut self, value: Vec<String>) {
        self.inner.comments = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "SP3Header(version={}, agency='{}', time_system={}, file_type={}, epoch_interval={})",
            self.inner.version,
            self.inner.agency,
            self.inner.time_system,
            self.inner.file_type,
            self.inner.epoch_interval
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// An SP3 precise orbit file held in memory.
///
/// Read SP3-a through SP3-d files, build files epoch by epoch, and write
/// SP3-d text. All values are SI. The file stores satellites in header
/// order and records on a fixed epoch grid. Properties and accessors return
/// copies; to change the header or a record, modify the copy and assign it
/// back with ``sp3.header = header`` or ``sp3.set_record(...)``.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sp3 = bh.SP3(["G01", "G02"], 900.0, agency="TEST")
///     t0 = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
///     k = sp3.push_epoch(t0)
///     sp3.set_record("G01", k, bh.SP3Record(np.array([15.0e6, 20.0e6, 5.0e6]), clock=2.5e-5))
///
///     parsed = bh.SP3.from_str(sp3.to_string())
///     assert parsed.num_epochs == 1
///     assert abs(parsed.record("G01", 0).clock - 2.5e-5) < 1e-12
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SP3")]
#[derive(Clone)]
pub struct PySP3 {
    pub(crate) inner: sp3::SP3,
}

#[pymethods]
impl PySP3 {
    /// Create an empty file with a fixed epoch grid.
    ///
    /// The header is SP3-d with a file type derived from the satellite
    /// systems. The start epoch and GPS week fields are set by the first
    /// ``push_epoch`` call.
    ///
    /// Args:
    ///     satellites (list[str | SP3SatelliteId]): Satellites in header order, at most 999, no duplicates.
    ///     epoch_interval (float): Spacing of the epoch grid, seconds.
    ///     agency (str, optional): Producing agency, at most 4 characters; defaults to ``BRHE``.
    ///     time_system (SP3TimeSystem | str, optional): Time system of the epochs; defaults to ``GPS``. Must have a Brahe equivalent (GPS, GAL, BDT, TAI, UTC).
    ///     coordinate_system (str, optional): Coordinate system label, at most 5 characters; defaults to ``IGS20``.
    ///     orbit_type (SP3OrbitType | str, optional): Orbit type, at most 3 characters; defaults to ``EXT``.
    ///     data_used (str, optional): Data used descriptor, at most 5 characters; defaults to ``ORBIT``.
    ///     include_velocity (bool, optional): Whether velocity records are written; defaults to True.
    ///     comments (list[str], optional): Comment lines, at most 77 characters each.
    ///
    /// Returns:
    ///     SP3: The empty file.
    ///
    /// Raises:
    ///     BraheError: If a satellite is invalid or duplicated, there are more than 999 satellites, the interval is not positive and finite, the time system has no Brahe equivalent, or a header label is longer than its field (agency 4, coordinate system 5, data used 5, orbit type 3 characters), empty (except the agency), or contains whitespace.
    #[new]
    #[pyo3(
        signature = (
            satellites, epoch_interval, *,
            agency="BRHE", time_system=None, coordinate_system="IGS20", orbit_type=None,
            data_used="ORBIT", include_velocity=true, comments=None
        ),
        text_signature = "(satellites, epoch_interval, *, agency='BRHE', time_system='GPS', coordinate_system='IGS20', orbit_type='EXT', data_used='ORBIT', include_velocity=True, comments=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        satellites: Vec<Bound<'_, PyAny>>,
        epoch_interval: f64,
        agency: &str,
        time_system: Option<Bound<'_, PyAny>>,
        coordinate_system: &str,
        orbit_type: Option<Bound<'_, PyAny>>,
        data_used: &str,
        include_velocity: bool,
        comments: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let ids = satellites
            .iter()
            .map(extract_sp3_satellite_id)
            .collect::<PyResult<Vec<_>>>()?;
        let time_system = match time_system {
            Some(obj) => obj,
            None => PyString::new(py, "GPS").into_any(),
        };
        let orbit_type = match orbit_type {
            Some(obj) => obj,
            None => PyString::new(py, "EXT").into_any(),
        };
        let options = sp3_write_options(
            agency,
            &time_system,
            coordinate_system,
            &orbit_type,
            data_used,
            include_velocity,
            comments,
        )?;
        sp3::SP3::new(ids, epoch_interval, options)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Parse an SP3 file from text.
    ///
    /// Args:
    ///     content (str): Full file text.
    ///
    /// Returns:
    ///     SP3: The parsed file.
    ///
    /// Raises:
    ///     BraheError: If the header or any record is malformed. Messages carry the one-based line number.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     text = open("test_assets/sp3/example-d-2.sp3").read()
    ///     sp3 = bh.SP3.from_str(text)
    ///     assert sp3.num_satellites == 26
    ///     ```
    #[staticmethod]
    fn from_str(content: &str) -> PyResult<Self> {
        sp3::SP3::from_str(content)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Read and parse an SP3 file.
    ///
    /// Args:
    ///     path (str): Path to the file; any ``os.PathLike`` is accepted.
    ///
    /// Returns:
    ///     SP3: The parsed file.
    ///
    /// Raises:
    ///     BraheError: If the file cannot be read or parsed.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    ///     assert sp3.header.version == bh.SP3Version.D
    ///     ```
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        sp3::SP3::from_file(path)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Render the file as SP3-d text.
    ///
    /// The version, epoch count, start epoch, GPS week, and date fields are
    /// derived from the epoch table rather than taken from the header.
    ///
    /// Returns:
    ///     str: The file text.
    ///
    /// Raises:
    ///     BraheError: If the file cannot be represented in SP3-d, for example an off-grid epoch, an epoch interval that rounds to zero at 8 decimal places, a comment longer than 77 characters, a header label longer than its field (agency 4, coordinate system 5, data used 5, orbit type 3 characters; shorten it by assigning a modified ``header``), or a value that does not fit its column.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    ///     assert sp3.to_string().startswith("#dV2001")
    ///     ```
    fn to_string(&self) -> PyResult<String> {
        self.inner.to_string().map_err(sp3_py_err)
    }

    /// Write the file as SP3-d text.
    ///
    /// Args:
    ///     path (str): Destination path; any ``os.PathLike`` is accepted.
    ///
    /// Raises:
    ///     BraheError: If the file cannot be represented in SP3-d or cannot be written.
    ///
    /// Example:
    ///     ```python
    ///     import os
    ///     import tempfile
    ///
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    ///     with tempfile.TemporaryDirectory() as directory:
    ///         sp3.to_file(os.path.join(directory, "example-d-2-copy.sp3"))
    ///     ```
    fn to_file(&self, path: PathBuf) -> PyResult<()> {
        self.inner.to_file(path).map_err(sp3_py_err)
    }

    /// Append an epoch to the grid.
    ///
    /// The first epoch sets the header start epoch, GPS week and date fields.
    /// Later epochs must lie after the last epoch on the epoch grid.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to append.
    ///
    /// Returns:
    ///     int: Index of the new epoch.
    ///
    /// Raises:
    ///     BraheError: If the epoch is not after the last epoch or is off the grid.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3(["G01"], 900.0)
    ///     t0 = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    ///     assert sp3.push_epoch(t0) == 0
    ///     assert sp3.push_epoch(t0 + 2700.0) == 1
    ///     ```
    fn push_epoch(&mut self, epoch: PyEpoch) -> PyResult<usize> {
        self.inner.push_epoch(epoch.obj).map_err(sp3_py_err)
    }

    /// Set the record of a satellite at an epoch.
    ///
    /// A record whose position is exactly zero marks the satellite as absent
    /// at that epoch.
    ///
    /// Args:
    ///     satellite (str | SP3SatelliteId): Satellite.
    ///     epoch_index (int): Index of the epoch.
    ///     record (SP3Record): Record to store.
    ///
    /// Raises:
    ///     BraheError: If the satellite is not in the file or the epoch index is out of range.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     sp3 = bh.SP3(["G01"], 900.0)
    ///     k = sp3.push_epoch(bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS))
    ///     sp3.set_record("G01", k, bh.SP3Record(np.array([2.0e7, 1.0e7, 1.0e7])))
    ///     assert sp3.record("G01", k) is not None
    ///     ```
    fn set_record(
        &mut self,
        satellite: &Bound<'_, PyAny>,
        epoch_index: usize,
        record: PySP3Record,
    ) -> PyResult<()> {
        let id = extract_sp3_satellite_id(satellite)?;
        self.inner
            .set_record(id, epoch_index, record.inner)
            .map_err(sp3_py_err)
    }

    /// Record of a satellite at an epoch.
    ///
    /// Args:
    ///     satellite (str | SP3SatelliteId): Satellite.
    ///     epoch_index (int): Index of the epoch.
    ///
    /// Returns:
    ///     SP3Record | None: A copy of the record, or None if the satellite is absent at that epoch.
    ///
    /// Raises:
    ///     BraheError: If the satellite is not in the file or the epoch index is out of range.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    ///     record = sp3.record("G01", 0)
    ///     print(record.position, record.clock)
    ///     ```
    fn record(
        &self,
        satellite: &Bound<'_, PyAny>,
        epoch_index: usize,
    ) -> PyResult<Option<PySP3Record>> {
        let id = extract_sp3_satellite_id(satellite)?;
        self.inner
            .record(id, epoch_index)
            .map(|r| r.cloned().map(|inner| PySP3Record { inner }))
            .map_err(sp3_py_err)
    }

    /// File header.
    ///
    /// Returns:
    ///     SP3Header: A copy of the header. Assign a modified copy back to replace the header.
    #[getter]
    fn header(&self) -> PySP3Header {
        PySP3Header { inner: self.inner.header().clone() }
    }

    #[setter]
    fn set_header(&mut self, value: PySP3Header) {
        *self.inner.header_mut() = value.inner;
    }

    /// Epochs of the file.
    ///
    /// Returns:
    ///     list[Epoch]: Epochs in ascending order, in the file's time system.
    #[getter]
    fn epochs(&self) -> Vec<PyEpoch> {
        self.inner
            .epochs()
            .iter()
            .map(|&obj| PyEpoch { obj })
            .collect()
    }

    /// Satellites in header order.
    ///
    /// Returns:
    ///     list[SP3SatelliteId]: Satellite identifiers.
    #[getter]
    fn satellite_ids(&self) -> Vec<PySP3SatelliteId> {
        self.inner
            .satellite_ids()
            .iter()
            .map(|&inner| PySP3SatelliteId { inner })
            .collect()
    }

    /// Number of epochs.
    ///
    /// Returns:
    ///     int: Epoch count.
    #[getter]
    fn num_epochs(&self) -> usize {
        self.inner.num_epochs()
    }

    /// Number of satellites.
    ///
    /// Returns:
    ///     int: Satellite count.
    #[getter]
    fn num_satellites(&self) -> usize {
        self.inner.num_satellites()
    }

    /// First epoch of the file.
    ///
    /// Returns:
    ///     Epoch: First epoch, or GPS epoch zero if the file has none.
    #[getter]
    fn start_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.inner.start_epoch() }
    }

    /// Last epoch of the file.
    ///
    /// Returns:
    ///     Epoch: Last epoch, or GPS epoch zero if the file has none.
    #[getter]
    fn end_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.inner.end_epoch() }
    }

    /// Set the clock correction of a satellite at every epoch.
    ///
    /// ``None`` or NaN clears the clock at that epoch, so the ``clocks()``
    /// array of an ``SP3Satellite`` can be passed back. Nothing is modified
    /// when the call fails.
    ///
    /// Args:
    ///     satellite (str | SP3SatelliteId): Satellite.
    ///     clocks (list[float | None]): One clock correction per epoch of the file, seconds. NaN is treated as None.
    ///
    /// Raises:
    ///     BraheError: If the satellite is not in the file, ``clocks`` does not hold one value per epoch, or a value is given at an epoch where the satellite has no record.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     sp3 = bh.SP3(["G01"], 900.0)
    ///     t0 = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    ///     sp3.push_epoch(t0)
    ///     sp3.push_epoch(t0 + 900.0)
    ///     sp3.set_record("G01", 0, bh.SP3Record(np.array([2.0e7, 1.0e7, 1.0e7])))
    ///
    ///     sp3.set_clocks("G01", [1e-5, None])
    ///     assert sp3.record("G01", 0).clock == 1e-5
    ///     ```
    fn set_clocks(
        &mut self,
        satellite: &Bound<'_, PyAny>,
        clocks: Vec<Option<f64>>,
    ) -> PyResult<()> {
        let id = extract_sp3_satellite_id(satellite)?;
        self.inner
            .set_clocks(id, &sp3_nan_to_none(clocks))
            .map_err(sp3_py_err)
    }

    /// Set the clock rate of a satellite at every epoch.
    ///
    /// Clock rates are stored on velocity records, so the file must have
    /// velocity records. ``None`` or NaN clears the rate at that epoch.
    /// Nothing is modified when the call fails.
    ///
    /// Args:
    ///     satellite (str | SP3SatelliteId): Satellite.
    ///     rates (list[float | None]): One clock rate per epoch of the file, seconds per second. NaN is treated as None.
    ///
    /// Raises:
    ///     BraheError: If the file has no velocity records, the satellite is not in the file, ``rates`` does not hold one value per epoch, or a value is given at an epoch where the satellite has no record.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     sp3 = bh.SP3(["G01"], 900.0)
    ///     sp3.push_epoch(bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS))
    ///     record = bh.SP3Record(np.array([2.0e7, 1.0e7, 1.0e7]), velocity=np.array([0.0, 3000.0, 0.0]))
    ///     sp3.set_record("G01", 0, record)
    ///
    ///     sp3.set_clock_rates("G01", [1e-12])
    ///     assert sp3.record("G01", 0).clock_rate == 1e-12
    ///     ```
    fn set_clock_rates(
        &mut self,
        satellite: &Bound<'_, PyAny>,
        rates: Vec<Option<f64>>,
    ) -> PyResult<()> {
        let id = extract_sp3_satellite_id(satellite)?;
        self.inner
            .set_clock_rates(id, &sp3_nan_to_none(rates))
            .map_err(sp3_py_err)
    }

    /// Build an SP3 file by sampling orbit state providers on a regular grid.
    ///
    /// Each provider is sampled with ``state_itrf`` at ``start + k * step``
    /// for ``k = 0, 1, ...`` up to the last grid point not after ``end``.
    /// Velocities are stored when ``include_velocity`` is set. Clocks and
    /// clock rates are absent and are written as the format's bad-clock value
    /// unless set with ``set_clocks`` and ``set_clock_rates``. The file type
    /// follows the satellite systems.
    ///
    /// Args:
    ///     propagators (list[tuple[str | SP3SatelliteId, KeplerianPropagator | SGPPropagator | NumericalOrbitPropagator | OrbitTrajectory | SP3Satellite]]): Satellite identifiers paired with the providers that supply their states; provider types may be mixed.
    ///     start (Epoch): First epoch of the file.
    ///     end (Epoch): Last epoch to include.
    ///     step (float): Epoch spacing, seconds.
    ///     agency (str, optional): Producing agency, at most 4 characters; defaults to ``BRHE``.
    ///     time_system (SP3TimeSystem | str, optional): Time system of the epochs; defaults to ``GPS``. Must have a Brahe equivalent (GPS, GAL, BDT, TAI, UTC).
    ///     coordinate_system (str, optional): Coordinate system label, at most 5 characters; defaults to ``IGS20``.
    ///     orbit_type (SP3OrbitType | str, optional): Orbit type, at most 3 characters; defaults to ``EXT``.
    ///     data_used (str, optional): Data used descriptor, at most 5 characters; defaults to ``ORBIT``.
    ///     include_velocity (bool, optional): Whether velocity records are stored; defaults to True.
    ///     comments (list[str], optional): Comment lines, at most 77 characters each.
    ///
    /// Returns:
    ///     SP3: The file.
    ///
    /// Raises:
    ///     TypeError: If an item is not a ``(satellite, provider)`` tuple or a provider is not one of the accepted types.
    ///     BraheError: If ``end`` precedes ``start``, the header options or satellites are invalid, or a provider cannot supply a state at a grid epoch. A provider error names the satellite and the epoch.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     t0 = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    ///     oe = np.array([bh.R_EARTH + 800e3, 0.001, 98.0, 0.0, 0.0, 0.0])
    ///     prop = bh.KeplerianPropagator.from_keplerian(t0, oe, bh.AngleFormat.DEGREES, 60.0)
    ///
    ///     sp3 = bh.SP3.from_propagators([("L01", prop)], t0, t0 + 3600.0, 60.0, agency="DEMO")
    ///     assert sp3.num_epochs == 61
    ///     ```
    #[staticmethod]
    #[pyo3(
        signature = (
            propagators, start, end, step, *,
            agency="BRHE", time_system=None, coordinate_system="IGS20", orbit_type=None,
            data_used="ORBIT", include_velocity=true, comments=None
        ),
        text_signature = "(propagators, start, end, step, *, agency='BRHE', time_system='GPS', coordinate_system='IGS20', orbit_type='EXT', data_used='ORBIT', include_velocity=True, comments=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    fn from_propagators<'py>(
        py: Python<'py>,
        propagators: Vec<Bound<'py, PyAny>>,
        start: PyEpoch,
        end: PyEpoch,
        step: f64,
        agency: &str,
        time_system: Option<Bound<'py, PyAny>>,
        coordinate_system: &str,
        orbit_type: Option<Bound<'py, PyAny>>,
        data_used: &str,
        include_velocity: bool,
        comments: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut entries: Vec<(sp3::SP3SatelliteId, SP3ProviderRef<'py>)> =
            Vec::with_capacity(propagators.len());
        for item in &propagators {
            let (satellite, provider): (Bound<'py, PyAny>, Bound<'py, PyAny>) =
                item.extract().map_err(|_| {
                    exceptions::PyTypeError::new_err(
                        "each item of propagators must be a (satellite, provider) tuple",
                    )
                })?;
            entries.push((
                extract_sp3_satellite_id(&satellite)?,
                SP3ProviderRef::extract(&provider)?,
            ));
        }
        let providers: Vec<(sp3::SP3SatelliteId, &dyn DOrbitStateProvider)> = entries
            .iter()
            .map(|(id, guard)| (*id, guard.provider()))
            .collect();

        let time_system = match time_system {
            Some(obj) => obj,
            None => PyString::new(py, "GPS").into_any(),
        };
        let orbit_type = match orbit_type {
            Some(obj) => obj,
            None => PyString::new(py, "EXT").into_any(),
        };
        let options = sp3_write_options(
            agency,
            &time_system,
            coordinate_system,
            &orbit_type,
            data_used,
            include_velocity,
            comments,
        )?;
        sp3::SP3::from_propagators(&providers, start.obj, end.obj, step, options)
            .map(|inner| Self { inner })
            .map_err(sp3_py_err)
    }

    /// Extract one satellite.
    ///
    /// The result keeps the epochs where the satellite has a record with a
    /// non-zero position, derives velocities when the file has none, and
    /// splits the samples into contiguous runs at gaps.
    ///
    /// Args:
    ///     satellite (str | SP3SatelliteId): Satellite.
    ///
    /// Returns:
    ///     SP3Satellite: The satellite.
    ///
    /// Raises:
    ///     BraheError: If the satellite is not in the file or has no valid sample.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    ///     g01 = sp3.satellite("G01")
    ///     print(len(g01), g01.start_epoch, g01.end_epoch)
    ///     ```
    fn satellite(&self, satellite: &Bound<'_, PyAny>) -> PyResult<PySP3Satellite> {
        let id = extract_sp3_satellite_id(satellite)?;
        self.inner
            .satellite(id)
            .map(|inner| PySP3Satellite { inner })
            .map_err(sp3_py_err)
    }

    /// Extract every satellite that has at least one valid sample.
    ///
    /// Returns:
    ///     list[SP3Satellite]: Satellites in header order, omitting satellites without a valid sample.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    ///     for sat in sp3.satellites():
    ///         print(sat.id, len(sat))
    ///     ```
    fn satellites(&self) -> PyResult<Vec<PySP3Satellite>> {
        self.inner
            .satellites()
            .map(|sats| {
                sats.into_iter()
                    .map(|inner| PySP3Satellite { inner })
                    .collect()
            })
            .map_err(sp3_py_err)
    }

    fn __len__(&self) -> usize {
        self.inner.num_epochs()
    }

    fn __contains__(&self, satellite: &Bound<'_, PyAny>) -> PyResult<bool> {
        let id = extract_sp3_satellite_id(satellite)?;
        Ok(self.inner.contains(id))
    }

    fn __repr__(&self) -> String {
        format!(
            "SP3(satellites={}, epochs={}, agency='{}')",
            self.inner.num_satellites(),
            self.inner.num_epochs(),
            self.inner.header().agency
        )
    }
}

/// Borrow of a Python orbit state provider accepted by `SP3.from_propagators`.
#[allow(clippy::upper_case_acronyms)]
enum SP3ProviderRef<'py> {
    Keplerian(PyRef<'py, PyKeplerianPropagator>),
    SGP(PyRef<'py, PySGPPropagator>),
    Numerical(PyRef<'py, PyNumericalOrbitPropagator>),
    Trajectory(PyRef<'py, PyOrbitalTrajectory>),
    Satellite(PyRef<'py, PySP3Satellite>),
}

impl<'py> SP3ProviderRef<'py> {
    /// Borrow `obj` as one of the accepted provider types.
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(p) = obj.cast::<PyKeplerianPropagator>() {
            return Ok(Self::Keplerian(p.try_borrow()?));
        }
        if let Ok(p) = obj.cast::<PySGPPropagator>() {
            return Ok(Self::SGP(p.try_borrow()?));
        }
        if let Ok(p) = obj.cast::<PyNumericalOrbitPropagator>() {
            return Ok(Self::Numerical(p.try_borrow()?));
        }
        if let Ok(t) = obj.cast::<PyOrbitalTrajectory>() {
            return Ok(Self::Trajectory(t.try_borrow()?));
        }
        if let Ok(s) = obj.cast::<PySP3Satellite>() {
            return Ok(Self::Satellite(s.try_borrow()?));
        }
        Err(exceptions::PyTypeError::new_err(format!(
            "provider must be a KeplerianPropagator, SGPPropagator, NumericalOrbitPropagator, \
             OrbitTrajectory, or SP3Satellite, got {}",
            obj.get_type().name()?
        )))
    }

    fn provider(&self) -> &dyn DOrbitStateProvider {
        match self {
            Self::Keplerian(p) => &p.propagator,
            Self::SGP(p) => &p.propagator,
            Self::Numerical(p) => &p.propagator,
            Self::Trajectory(t) => &t.trajectory,
            Self::Satellite(s) => &s.inner,
        }
    }
}

/// Convert per-sample 3-vectors into an (N, 3) array.
fn sp3_vectors_to_numpy<'py>(py: Python<'py>, rows: &[[f64; 3]]) -> Bound<'py, PyArray<f64, Ix2>> {
    ndarray::Array2::from_shape_fn((rows.len(), 3), |(i, j)| rows[i][j]).into_pyarray(py)
}

/// Convert optional per-sample values into an (N,) array with NaN for None.
fn sp3_optionals_to_numpy<'py>(
    py: Python<'py>,
    values: &[Option<f64>],
) -> Bound<'py, PyArray<f64, Ix1>> {
    values
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect::<Vec<f64>>()
        .into_pyarray(py)
}

/// Convert a list of states into Python arrays.
fn sp3_states_to_numpy<'py>(
    py: Python<'py>,
    states: &[Vector6<f64>],
) -> Vec<Bound<'py, PyArray<f64, Ix1>>> {
    states
        .iter()
        .map(|s| s.as_slice().to_pyarray(py).to_owned())
        .collect()
}

/// Parse an optional UUID string.
fn sp3_parse_uuid(uuid_str: Option<String>) -> PyResult<Option<uuid::Uuid>> {
    uuid_str
        .map(|s| {
            uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))
        })
        .transpose()
}

/// Orbit and clock samples of one satellite from SP3 data.
///
/// Only epochs where the satellite has a valid position are kept; every data
/// accessor returns one entry per sample, aligned with ``epochs()``.
/// Positions and velocities are in the file's Earth-fixed frame (ITRF), in m
/// and m/s.
///
/// Velocities come from the file when every sample has one. Otherwise each
/// sample's velocity is the derivative of a Lagrange polynomial of degree
/// ``SP3_DEFAULT_LAGRANGE_DEGREE`` fitted through neighboring samples of the
/// same contiguous run, as centered on the sample as the run allows. A sample
/// that forms a run on its own has no derived velocity and reports NaN.
///
/// Two consecutive samples spaced more than 1.5 times the larger of their
/// epoch intervals apart start a new run, so the satellite has a gap between
/// them. Each run of at least two samples becomes one ITRF ``OrbitTrajectory``
/// named after the satellite.
///
/// The satellite is an orbit state provider with the same ``state_*`` and
/// ``states_*`` methods as ``OrbitTrajectory``, and is named after its
/// identifier (such as ``"G01"``), so it can be passed to
/// ``location_accesses`` directly. States are interpolated in ITRF from the
/// run containing the requested epoch, both ends of a run included, and
/// converted to other frames on request. A request before the first sample,
/// after the last sample, inside a gap, or at a sample that forms a run on
/// its own raises ``BraheError`` naming the satellite.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sp3 = bh.SP3(["G01"], 900.0, include_velocity=False)
///     t0 = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
///     for k in range(4):
///         index = sp3.push_epoch(t0 + 900.0 * k)
///         sp3.set_record("G01", index, bh.SP3Record(np.array([2.0e7 + 900.0 * k, 1.0e7, 1.0e7])))
///
///     sat = sp3.satellite("G01")
///     assert len(sat) == 4
///     assert abs(sat.velocities()[0, 0] - 1.0) < 1e-9
///     assert sat.get_name() == "G01"
///     state = sat.state_itrf(t0 + 900.0)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SP3Satellite")]
#[derive(Clone)]
pub struct PySP3Satellite {
    pub(crate) inner: sp3::SP3Satellite,
}

#[pymethods]
impl PySP3Satellite {
    /// Chain every satellite across several SP3 files.
    ///
    /// The files are sorted by start epoch, then by end epoch, so they may be
    /// passed in any order; files without epochs are ignored. For each
    /// satellite in the union of the files' satellite lists, the valid
    /// samples of every file are joined in time order before velocities are
    /// derived, so samples next to a file boundary get centered fits. Epochs
    /// compare as absolute instants, so files in different time systems
    /// chain directly.
    ///
    /// Two consecutive files may share their boundary epoch. The later file's
    /// sample is kept there when it is valid, and the earlier file's
    /// otherwise. A file that lacks a satellite contributes no samples, which
    /// leaves a gap in that satellite. Velocities come from the files only
    /// when every sample comes from a file with velocity records.
    ///
    /// Args:
    ///     files (list[SP3]): SP3 files covering consecutive, non-overlapping spans.
    ///
    /// Returns:
    ///     list[SP3Satellite]: One satellite for each identifier with at least one valid sample, in ascending identifier order (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS, SBAS, LEO, then by number).
    ///
    /// Raises:
    ///     BraheError: If two files overlap by more than a shared boundary epoch; the message names the agency and span of both.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     files = [
    ///         bh.SP3.from_file("test_assets/sp3/gbm19500_truncated.sp3"),
    ///         bh.SP3.from_file("test_assets/sp3/gbm19500_after_drop.sp3"),
    ///     ]
    ///     sats = bh.SP3Satellite.from_files(files)
    ///     print(len(sats), sats[0].id, sats[0].segments())
    ///     ```
    #[staticmethod]
    fn from_files(files: Vec<PySP3>) -> PyResult<Vec<PySP3Satellite>> {
        let files: Vec<sp3::SP3> = files.into_iter().map(|f| f.inner).collect();
        sp3::SP3Satellite::from_files(&files)
            .map(|sats| {
                sats.into_iter()
                    .map(|inner| PySP3Satellite { inner })
                    .collect()
            })
            .map_err(sp3_py_err)
    }

    /// Satellite identifier.
    ///
    /// Returns:
    ///     SP3SatelliteId: The identifier.
    #[getter(id)]
    fn satellite_id(&self) -> PySP3SatelliteId {
        PySP3SatelliteId {
            inner: self.inner.id(),
        }
    }

    /// Epoch of the first sample.
    ///
    /// Returns:
    ///     Epoch: First sample epoch.
    #[getter]
    fn start_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.start_epoch(),
        }
    }

    /// Epoch of the last sample.
    ///
    /// Returns:
    ///     Epoch: Last sample epoch.
    #[getter]
    fn end_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.end_epoch(),
        }
    }

    /// Dimension of the provider state.
    ///
    /// Returns:
    ///     int: Always 6 (position then velocity).
    #[getter]
    fn state_dim(&self) -> usize {
        DStateProvider::state_dim(&self.inner)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SP3Satellite(id={}, samples={}, segments={})",
            self.inner.id(),
            self.inner.len(),
            self.inner.segments().len()
        )
    }

    /// Epochs of the valid samples, in increasing order.
    ///
    /// Returns:
    ///     list[Epoch]: Sample epochs.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     epochs = sat.epochs()
    ///     ```
    fn epochs(&self) -> Vec<PyEpoch> {
        self.inner
            .epochs()
            .iter()
            .map(|&obj| PyEpoch { obj })
            .collect()
    }

    /// Positions of the valid samples.
    ///
    /// Returns:
    ///     numpy.ndarray: Array of shape (N, 3) with the ITRF position of each sample, meters.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     assert sat.positions().shape == (len(sat), 3)
    ///     ```
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        sp3_vectors_to_numpy(py, &self.inner.positions())
    }

    /// Velocities of the valid samples.
    ///
    /// The values come from the file when every sample has a file velocity,
    /// and are derived from the positions otherwise. A derived velocity is
    /// NaN for a sample that forms a run on its own.
    ///
    /// Returns:
    ///     numpy.ndarray: Array of shape (N, 3) with the ITRF velocity of each sample, meters per second.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     assert sat.velocities().shape == (len(sat), 3)
    ///     ```
    fn velocities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        sp3_vectors_to_numpy(py, self.inner.velocities())
    }

    /// Whether the velocities come from the file.
    ///
    /// Returns:
    ///     bool: True if every sample carries a file velocity, False if the velocities are derived from the positions.
    fn has_file_velocities(&self) -> bool {
        self.inner.has_file_velocities()
    }

    /// Clock corrections of the valid samples.
    ///
    /// Returns:
    ///     numpy.ndarray: Array of shape (N,) with the clock correction of each sample in seconds, NaN where the file has a bad or missing clock.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     valid = ~np.isnan(sat.clocks())
    ///     ```
    fn clocks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        sp3_optionals_to_numpy(py, &self.inner.clocks())
    }

    /// Clock rates of the valid samples.
    ///
    /// Returns:
    ///     numpy.ndarray: Array of shape (N,) with the clock rate of each sample in seconds per second, NaN where the file has a bad or missing rate.
    fn clock_rates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        sp3_optionals_to_numpy(py, &self.inner.clock_rates())
    }

    /// Full file records of the valid samples.
    ///
    /// The records hold the file values unchanged, including standard
    /// deviations, flags, and correlations; derived velocities are not
    /// written into them.
    ///
    /// Returns:
    ///     list[SP3Record]: A copy of the record of each sample.
    fn records(&self) -> Vec<PySP3Record> {
        self.inner
            .records()
            .iter()
            .map(|r| PySP3Record { inner: r.clone() })
            .collect()
    }

    /// Spans of the contiguous runs of samples.
    ///
    /// A run that holds a single sample has equal start and end epochs.
    ///
    /// Returns:
    ///     list[tuple[Epoch, Epoch]]: First and last sample epoch of each run, in time order.
    fn segments(&self) -> Vec<(PyEpoch, PyEpoch)> {
        self.inner
            .segments()
            .into_iter()
            .map(|(start, end)| (PyEpoch { obj: start }, PyEpoch { obj: end }))
            .collect()
    }

    /// Whether the samples form more than one contiguous run.
    ///
    /// Returns:
    ///     bool: True if the satellite has at least one gap.
    fn has_gaps(&self) -> bool {
        self.inner.has_gaps()
    }

    /// Get the interpolation method.
    ///
    /// Returns:
    ///     InterpolationMethod: The method as set, before any per-run Lagrange degree adjustment; defaults to Lagrange of degree ``SP3_DEFAULT_LAGRANGE_DEGREE``.
    fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod {
            method: self.inner.get_interpolation_method(),
        }
    }

    /// Set the interpolation method.
    ///
    /// The method applies to state queries and to the trajectories returned
    /// afterwards. For each run's trajectory a Lagrange degree is reduced to
    /// at most the run length minus one. The fit that derives velocities is
    /// not affected.
    ///
    /// Args:
    ///     method (InterpolationMethod): Interpolation method.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     sat.set_interpolation_method(bh.InterpolationMethod.HERMITE_CUBIC)
    ///     ```
    fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.inner.set_interpolation_method(method.method);
    }

    /// Set the interpolation method and return self (builder pattern).
    ///
    /// Args:
    ///     method (InterpolationMethod): Interpolation method.
    ///
    /// Returns:
    ///     SP3Satellite: Self with the method set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     sat = sat.with_interpolation_method(bh.InterpolationMethod.lagrange(5))
    ///     ```
    fn with_interpolation_method<'a>(
        mut slf: PyRefMut<'a, Self>,
        method: PyRef<'_, PyInterpolationMethod>,
    ) -> PyRefMut<'a, Self> {
        slf.inner.set_interpolation_method(method.method);
        slf
    }

    /// Trajectory over all samples.
    ///
    /// The trajectory is a Cartesian ITRF ``OrbitTrajectory`` named after the
    /// satellite, using the satellite's interpolation method. A Lagrange
    /// degree is raised to at least 1 and reduced to at most the number of
    /// samples minus one.
    ///
    /// Returns:
    ///     OrbitTrajectory: The trajectory.
    ///
    /// Raises:
    ///     BraheError: If the satellite has a gap, naming the satellite and each gap and pointing to ``to_trajectory_segments``; or if it has a single sample, naming the satellite and epoch.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sat = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3").satellite("G05")
    ///     traj = sat.to_trajectory()
    ///     gcrf = traj.to_gcrf()
    ///     ```
    fn to_trajectory(&self) -> PyResult<PyOrbitalTrajectory> {
        self.inner
            .to_trajectory()
            .map(|trajectory| PyOrbitalTrajectory { trajectory })
            .map_err(sp3_py_err)
    }

    /// One trajectory per contiguous run of at least two samples.
    ///
    /// Each trajectory is a Cartesian ITRF ``OrbitTrajectory`` named after
    /// the satellite, using the satellite's interpolation method with a
    /// Lagrange degree raised to at least 1 and reduced to at most the run
    /// length minus one. Runs of a single sample are omitted.
    ///
    /// Returns:
    ///     list[OrbitTrajectory]: Trajectories in time order.
    fn to_trajectory_segments(&self) -> PyResult<Vec<PyOrbitalTrajectory>> {
        self.inner
            .to_trajectory_segments()
            .map(|trajs| {
                trajs
                    .into_iter()
                    .map(|trajectory| PyOrbitalTrajectory { trajectory })
                    .collect()
            })
            .map_err(sp3_py_err)
    }

    /// State at an epoch in ITRF.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: ITRF state [x, y, z, vx, vy, vz] (meters, m/s).
    ///
    /// Raises:
    ///     BraheError: If the epoch is outside the samples, inside a gap, or at a sample that forms a run on its own.
    #[pyo3(text_signature = "(epoch)")]
    fn state<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DStateProvider::state(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in ECI (GCRF) Cartesian coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: ECI state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_eci<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_eci(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in ECEF (ITRF) Cartesian coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: ECEF state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_ecef<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_ecef(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in GCRF Cartesian coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: GCRF state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_gcrf<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_gcrf(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in ITRF Cartesian coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: ITRF state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_itrf<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_itrf(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in EME2000 Cartesian coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: EME2000 state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_eme2000<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_eme2000(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in the Earth's body-centered inertial frame (GCRF).
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: BCI state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_bci<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_bci(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch in the Earth's body-centered body-fixed frame (ITRF).
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: BCBF state [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    fn state_bcbf<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_bcbf(&self.inner, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// State at an epoch expressed in an arbitrary reference frame.
    ///
    /// Args:
    ///     frame (CelestialFrame): Reference frame to express the state in.
    ///     epoch (Epoch): Epoch to evaluate.
    ///
    /// Returns:
    ///     numpy.ndarray: State [x, y, z, vx, vy, vz] in ``frame`` (meters, m/s).
    #[pyo3(text_signature = "(frame, epoch)")]
    fn state_in_frame<'py>(
        &self,
        py: Python<'py>,
        frame: &PyCelestialFrame,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_in_frame(&self.inner, frame.frame, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Osculating Keplerian elements at an epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///     angle_format (AngleFormat): Angle format of the angular elements.
    ///
    /// Returns:
    ///     numpy.ndarray: Osculating elements [a, e, i, raan, argp, mean_anomaly].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    fn state_koe_osc<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_koe_osc(&self.inner, epoch.obj, angle_format.value)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Mean Keplerian elements at an epoch.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch to evaluate.
    ///     angle_format (AngleFormat): Angle format of the angular elements.
    ///
    /// Returns:
    ///     numpy.ndarray: Mean elements [a, e, i, raan, argp, mean_anomaly].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    fn state_koe_mean<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let state =
            DOrbitStateProvider::state_koe_mean(&self.inner, epoch.obj, angle_format.value)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// ITRF states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: ITRF state [x, y, z, vx, vy, vz] at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DStateProvider::states(&self.inner, &epochs)?;
        Ok(states
            .iter()
            .map(|s| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// ECI (GCRF) states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: ECI state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_eci<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_eci(&self.inner, &epochs)?,
        ))
    }

    /// ECEF (ITRF) states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: ECEF state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_ecef<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_ecef(&self.inner, &epochs)?,
        ))
    }

    /// GCRF states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: GCRF state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_gcrf<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_gcrf(&self.inner, &epochs)?,
        ))
    }

    /// ITRF states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: ITRF state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_itrf<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_itrf(&self.inner, &epochs)?,
        ))
    }

    /// EME2000 states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: EME2000 state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_eme2000<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_eme2000(&self.inner, &epochs)?,
        ))
    }

    /// Body-centered inertial (GCRF) states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: BCI state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_bci<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_bci(&self.inner, &epochs)?,
        ))
    }

    /// Body-centered body-fixed (ITRF) states at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: BCBF state at each epoch (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    fn states_bcbf<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_bcbf(&self.inner, &epochs)?,
        ))
    }

    /// States at several epochs expressed in an arbitrary reference frame.
    ///
    /// Args:
    ///     frame (CelestialFrame): Reference frame to express the states in.
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: State in ``frame`` at each epoch (meters, m/s).
    #[pyo3(text_signature = "(frame, epochs)")]
    fn states_in_frame<'py>(
        &self,
        py: Python<'py>,
        frame: &PyCelestialFrame,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_in_frame(&self.inner, frame.frame, &epochs)?,
        ))
    }

    /// Osculating Keplerian elements at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///     angle_format (AngleFormat): Angle format of the angular elements.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: Osculating elements [a, e, i, raan, argp, mean_anomaly] at each epoch.
    #[pyo3(text_signature = "(epochs, angle_format)")]
    fn states_koe_osc<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_koe_osc(&self.inner, &epochs, angle_format.value)?,
        ))
    }

    /// Mean Keplerian elements at several epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): Epochs to evaluate.
    ///     angle_format (AngleFormat): Angle format of the angular elements.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: Mean elements [a, e, i, raan, argp, mean_anomaly] at each epoch.
    #[pyo3(text_signature = "(epochs, angle_format)")]
    fn states_koe_mean<'py>(
        &self,
        py: Python<'py>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'py, PyArray<f64, Ix1>>>> {
        let epochs: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        Ok(sp3_states_to_numpy(
            py,
            &DOrbitStateProvider::states_koe_mean(&self.inner, &epochs, angle_format.value)?,
        ))
    }

    /// Get the name.
    ///
    /// Returns:
    ///     str | None: The name; defaults to the satellite identifier, such as ``"G01"``.
    fn get_name(&self) -> Option<String> {
        Identifiable::get_name(&self.inner).map(str::to_string)
    }

    /// Set the name in place.
    ///
    /// The trajectories returned afterwards carry the same name.
    ///
    /// Args:
    ///     name (str | None): Name to assign, or None to clear.
    fn set_name(&mut self, name: Option<String>) {
        Identifiable::set_name(&mut self.inner, name.as_deref());
    }

    /// Set the name and return self (builder pattern).
    ///
    /// Args:
    ///     name (str): Name to assign.
    ///
    /// Returns:
    ///     SP3Satellite: Self with the name set.
    fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        Identifiable::set_name(&mut slf.inner, Some(&name));
        slf
    }

    /// Get the numeric ID.
    ///
    /// Returns:
    ///     int | None: The numeric ID, or None if not set.
    fn get_id(&self) -> Option<u64> {
        Identifiable::get_id(&self.inner)
    }

    /// Set the numeric ID in place.
    ///
    /// Args:
    ///     id (int | None): Numeric ID to assign, or None to clear.
    fn set_id(&mut self, id: Option<u64>) {
        Identifiable::set_id(&mut self.inner, id);
    }

    /// Set the numeric ID and return self (builder pattern).
    ///
    /// Args:
    ///     id (int): Numeric ID to assign.
    ///
    /// Returns:
    ///     SP3Satellite: Self with the ID set.
    fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {
        Identifiable::set_id(&mut slf.inner, Some(id));
        slf
    }

    /// Get the UUID.
    ///
    /// Returns:
    ///     str | None: The UUID as a string, or None if not set.
    fn get_uuid(&self) -> Option<String> {
        Identifiable::get_uuid(&self.inner).map(|u| u.to_string())
    }

    /// Set the UUID and return self (builder pattern).
    ///
    /// Args:
    ///     uuid_str (str): UUID string to assign.
    ///
    /// Returns:
    ///     SP3Satellite: Self with the UUID set.
    ///
    /// Raises:
    ///     ValueError: If ``uuid_str`` is not a valid UUID.
    fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = sp3_parse_uuid(Some(uuid_str))?;
        let (name, id) = (
            Identifiable::get_name(&slf.inner).map(str::to_string),
            Identifiable::get_id(&slf.inner),
        );
        Identifiable::set_identity(&mut slf.inner, name.as_deref(), uuid, id);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self (builder pattern).
    ///
    /// Returns:
    ///     SP3Satellite: Self with a new UUID set.
    fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        Identifiable::generate_uuid(&mut slf.inner);
        slf
    }

    /// Generate a new UUID and set it in place.
    fn generate_uuid(&mut self) {
        Identifiable::generate_uuid(&mut self.inner);
    }

    /// Set every identity field and return self (builder pattern).
    ///
    /// Args:
    ///     name (str | None): Name to assign.
    ///     uuid_str (str | None): UUID string to assign.
    ///     id (int | None): Numeric ID to assign.
    ///
    /// Returns:
    ///     SP3Satellite: Self with the identity set.
    ///
    /// Raises:
    ///     ValueError: If ``uuid_str`` is not a valid UUID.
    fn with_identity(
        mut slf: PyRefMut<'_, Self>,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = sp3_parse_uuid(uuid_str)?;
        Identifiable::set_identity(&mut slf.inner, name.as_deref(), uuid, id);
        Ok(slf)
    }

    /// Set every identity field in place.
    ///
    /// Args:
    ///     name (str | None): Name to assign.
    ///     uuid_str (str | None): UUID string to assign.
    ///     id (int | None): Numeric ID to assign.
    ///
    /// Raises:
    ///     ValueError: If ``uuid_str`` is not a valid UUID.
    fn set_identity(
        &mut self,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<()> {
        let uuid = sp3_parse_uuid(uuid_str)?;
        Identifiable::set_identity(&mut self.inner, name.as_deref(), uuid, id);
        Ok(())
    }
}
