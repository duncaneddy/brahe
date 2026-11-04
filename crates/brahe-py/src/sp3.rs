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
///     ```
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

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        op.matches(self.inner.cmp(&other.inner))
    }

    fn __hash__(&self) -> u64 {
        sp3_hash(&self.inner)
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
    /// SP3 format version.
    ///
    /// Returns:
    ///     SP3Version: The version.
    #[getter]
    fn version(&self) -> PySP3Version {
        PySP3Version { inner: self.inner.version }
    }

    #[setter]
    fn set_version(&mut self, value: PySP3Version) {
        self.inner.version = value.inner;
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

    /// Data used descriptor.
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

    /// Coordinate system label.
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

    /// Orbit type.
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

    /// Producing agency.
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

    /// Satellite system covered by the file.
    ///
    /// Returns:
    ///     SP3FileType: The file type.
    #[getter]
    fn file_type(&self) -> PySP3FileType {
        PySP3FileType { inner: self.inner.file_type }
    }

    #[setter]
    fn set_file_type(&mut self, value: PySP3FileType) {
        self.inner.file_type = value.inner;
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
    ///     orbit_type (SP3OrbitType | str, optional): Orbit type; defaults to ``EXT``.
    ///     data_used (str, optional): Data used descriptor, at most 5 characters; defaults to ``ORBIT``.
    ///     include_velocity (bool, optional): Whether velocity records are written; defaults to True.
    ///     comments (list[str], optional): Comment lines, at most 77 characters each.
    ///
    /// Returns:
    ///     SP3: The empty file.
    ///
    /// Raises:
    ///     BraheError: If a satellite is invalid or duplicated, there are more than 999 satellites, the interval is not positive and finite, or the time system has no Brahe equivalent.
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
    ///     BraheError: If the file cannot be represented in SP3-d, for example an off-grid epoch, a comment longer than 77 characters, or a value that does not fit its column.
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
