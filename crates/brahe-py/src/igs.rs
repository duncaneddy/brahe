// Python bindings for the IGS product names module.

/// Hash a value that implements `Hash` with the standard hasher.
fn igs_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Solution type of an IGS product: final, rapid, ultra-rapid or predicted.
///
/// Codes other than ``FIN``, ``RAP``, ``ULT`` and ``PRD`` are kept as they
/// appear, uppercased, and compare equal to other solutions parsed from the
/// same code.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.IGSSolutionType.parse("fin") == bh.IGSSolutionType.FINAL
///     assert bh.IGSSolutionType.ULTRA_RAPID.code == "ULT"
///     assert bh.IGSSolutionType.parse("abc").code == "ABC"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSSolutionType")]
#[derive(Clone, PartialEq)]
pub struct PyIGSSolutionType {
    pub(crate) inner: igs::IGSSolutionType,
}

#[pymethods]
impl PyIGSSolutionType {
    /// Final solution (``FIN``).
    #[classattr]
    #[allow(non_snake_case)]
    fn FINAL() -> Self {
        Self { inner: igs::IGSSolutionType::Final }
    }

    /// Rapid solution (``RAP``).
    #[classattr]
    #[allow(non_snake_case)]
    fn RAPID() -> Self {
        Self { inner: igs::IGSSolutionType::Rapid }
    }

    /// Ultra-rapid solution (``ULT``).
    #[classattr]
    #[allow(non_snake_case)]
    fn ULTRA_RAPID() -> Self {
        Self { inner: igs::IGSSolutionType::UltraRapid }
    }

    /// Predicted solution (``PRD``).
    #[classattr]
    #[allow(non_snake_case)]
    fn PREDICTED() -> Self {
        Self { inner: igs::IGSSolutionType::Predicted }
    }

    /// Parse a solution code.
    ///
    /// Matching is case-insensitive. Unrecognized codes are kept, uppercased.
    ///
    /// Args:
    ///     code (str): Three-letter solution code, for example ``"FIN"``.
    ///
    /// Returns:
    ///     IGSSolutionType: The parsed solution type.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.IGSSolutionType.parse("fin") == bh.IGSSolutionType.FINAL
    ///     assert bh.IGSSolutionType.parse("xyz").code == "XYZ"
    ///     ```
    #[staticmethod]
    fn parse(code: &str) -> Self {
        Self { inner: igs::IGSSolutionType::parse(code) }
    }

    /// Three-letter code as it appears in a long product file name.
    ///
    /// Returns:
    ///     str: ``FIN``, ``RAP``, ``ULT``, ``PRD`` or the uppercased unrecognized code.
    #[getter]
    fn code(&self) -> String {
        self.inner.code().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.code().to_string()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            igs::IGSSolutionType::Final => "IGSSolutionType.FINAL".to_string(),
            igs::IGSSolutionType::Rapid => "IGSSolutionType.RAPID".to_string(),
            igs::IGSSolutionType::UltraRapid => "IGSSolutionType.ULTRA_RAPID".to_string(),
            igs::IGSSolutionType::Predicted => "IGSSolutionType.PREDICTED".to_string(),
            igs::IGSSolutionType::Other(ref code) => format!("IGSSolutionType.parse('{}')", code),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

/// Content type of an IGS product: orbit, clock, Earth rotation and so on.
///
/// Codes other than the listed ones are kept as they appear, uppercased.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.IGSContentType.parse("orb") == bh.IGSContentType.ORBIT
///     assert bh.IGSContentType.EARTH_ROTATION.code == "ERP"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSContentType")]
#[derive(Clone, PartialEq)]
pub struct PyIGSContentType {
    pub(crate) inner: igs::IGSContentType,
}

#[pymethods]
impl PyIGSContentType {
    /// Orbit (``ORB``).
    #[classattr]
    #[allow(non_snake_case)]
    fn ORBIT() -> Self {
        Self { inner: igs::IGSContentType::Orbit }
    }

    /// Clock (``CLK``).
    #[classattr]
    #[allow(non_snake_case)]
    fn CLOCK() -> Self {
        Self { inner: igs::IGSContentType::Clock }
    }

    /// Earth rotation parameters (``ERP``).
    #[classattr]
    #[allow(non_snake_case)]
    fn EARTH_ROTATION() -> Self {
        Self { inner: igs::IGSContentType::EarthRotation }
    }

    /// Station coordinate solution (``SOL``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SOLUTION() -> Self {
        Self { inner: igs::IGSContentType::Solution }
    }

    /// Satellite attitude (``ATT``).
    #[classattr]
    #[allow(non_snake_case)]
    fn ATTITUDE() -> Self {
        Self { inner: igs::IGSContentType::Attitude }
    }

    /// Signal biases (``BIA``).
    #[classattr]
    #[allow(non_snake_case)]
    fn BIAS() -> Self {
        Self { inner: igs::IGSContentType::Bias }
    }

    /// Global ionosphere maps (``GIM``).
    #[classattr]
    #[allow(non_snake_case)]
    fn IONOSPHERE() -> Self {
        Self { inner: igs::IGSContentType::Ionosphere }
    }

    /// Troposphere (``TRO``).
    #[classattr]
    #[allow(non_snake_case)]
    fn TROPOSPHERE() -> Self {
        Self { inner: igs::IGSContentType::Troposphere }
    }

    /// Product summary (``SUM``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SUMMARY() -> Self {
        Self { inner: igs::IGSContentType::Summary }
    }

    /// Parse a content-type code.
    ///
    /// Matching is case-insensitive. Unrecognized codes are kept, uppercased.
    ///
    /// Args:
    ///     code (str): Three-letter content code, for example ``"ORB"``.
    ///
    /// Returns:
    ///     IGSContentType: The parsed content type.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.IGSContentType.parse("orb") == bh.IGSContentType.ORBIT
    ///     ```
    #[staticmethod]
    fn parse(code: &str) -> Self {
        Self { inner: igs::IGSContentType::parse(code) }
    }

    /// Three-letter code as it appears in a long product file name.
    ///
    /// Returns:
    ///     str: The content code, uppercase.
    #[getter]
    fn code(&self) -> String {
        self.inner.code().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.code().to_string()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            igs::IGSContentType::Orbit => "IGSContentType.ORBIT".to_string(),
            igs::IGSContentType::Clock => "IGSContentType.CLOCK".to_string(),
            igs::IGSContentType::EarthRotation => "IGSContentType.EARTH_ROTATION".to_string(),
            igs::IGSContentType::Solution => "IGSContentType.SOLUTION".to_string(),
            igs::IGSContentType::Attitude => "IGSContentType.ATTITUDE".to_string(),
            igs::IGSContentType::Bias => "IGSContentType.BIAS".to_string(),
            igs::IGSContentType::Ionosphere => "IGSContentType.IONOSPHERE".to_string(),
            igs::IGSContentType::Troposphere => "IGSContentType.TROPOSPHERE".to_string(),
            igs::IGSContentType::Summary => "IGSContentType.SUMMARY".to_string(),
            igs::IGSContentType::Other(ref code) => format!("IGSContentType.parse('{}')", code),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

/// File format of an IGS product: SP3, CLK, ERP and so on.
///
/// Codes other than the listed ones are kept as they appear, uppercased.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.IGSFileFormat.parse("sp3") == bh.IGSFileFormat.SP3
///     assert bh.IGSFileFormat.SNX.code == "SNX"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSFileFormat")]
#[derive(Clone, PartialEq)]
pub struct PyIGSFileFormat {
    pub(crate) inner: igs::IGSFileFormat,
}

#[pymethods]
impl PyIGSFileFormat {
    /// SP3 precise ephemeris (``SP3``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SP3() -> Self {
        Self { inner: igs::IGSFileFormat::SP3 }
    }

    /// RINEX clock (``CLK``).
    #[classattr]
    #[allow(non_snake_case)]
    fn CLK() -> Self {
        Self { inner: igs::IGSFileFormat::CLK }
    }

    /// Earth rotation parameters (``ERP``).
    #[classattr]
    #[allow(non_snake_case)]
    fn ERP() -> Self {
        Self { inner: igs::IGSFileFormat::ERP }
    }

    /// SINEX (``SNX``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SNX() -> Self {
        Self { inner: igs::IGSFileFormat::SNX }
    }

    /// Orbit exchange, ORBEX (``OBX``).
    #[classattr]
    #[allow(non_snake_case)]
    fn OBX() -> Self {
        Self { inner: igs::IGSFileFormat::OBX }
    }

    /// Bias SINEX (``BIA``).
    #[classattr]
    #[allow(non_snake_case)]
    fn BIA() -> Self {
        Self { inner: igs::IGSFileFormat::BIA }
    }

    /// IONEX (``INX``).
    #[classattr]
    #[allow(non_snake_case)]
    fn INX() -> Self {
        Self { inner: igs::IGSFileFormat::INX }
    }

    /// Troposphere SINEX (``TRO``).
    #[classattr]
    #[allow(non_snake_case)]
    fn TRO() -> Self {
        Self { inner: igs::IGSFileFormat::TRO }
    }

    /// Summary text (``SUM``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SUM() -> Self {
        Self { inner: igs::IGSFileFormat::SUM }
    }

    /// Parse a file-format code.
    ///
    /// Matching is case-insensitive. Unrecognized codes are kept, uppercased.
    ///
    /// Args:
    ///     code (str): File-format code, for example ``"SP3"``.
    ///
    /// Returns:
    ///     IGSFileFormat: The parsed file format.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.IGSFileFormat.parse("sp3") == bh.IGSFileFormat.SP3
    ///     ```
    #[staticmethod]
    fn parse(code: &str) -> Self {
        Self { inner: igs::IGSFileFormat::parse(code) }
    }

    /// Format code as it appears in a long product file name.
    ///
    /// Returns:
    ///     str: The format code, uppercase.
    #[getter]
    fn code(&self) -> String {
        self.inner.code().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.code().to_string()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            igs::IGSFileFormat::SP3 => "IGSFileFormat.SP3".to_string(),
            igs::IGSFileFormat::CLK => "IGSFileFormat.CLK".to_string(),
            igs::IGSFileFormat::ERP => "IGSFileFormat.ERP".to_string(),
            igs::IGSFileFormat::SNX => "IGSFileFormat.SNX".to_string(),
            igs::IGSFileFormat::OBX => "IGSFileFormat.OBX".to_string(),
            igs::IGSFileFormat::BIA => "IGSFileFormat.BIA".to_string(),
            igs::IGSFileFormat::INX => "IGSFileFormat.INX".to_string(),
            igs::IGSFileFormat::TRO => "IGSFileFormat.TRO".to_string(),
            igs::IGSFileFormat::SUM => "IGSFileFormat.SUM".to_string(),
            igs::IGSFileFormat::Other(ref code) => format!("IGSFileFormat.parse('{}')", code),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

/// Compression applied to an IGS product file.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     name = bh.IGSProductFileName.parse("igs21000.sp3.Z")
///     assert name.compression == bh.IGSCompression.UNIX_COMPRESS
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSCompression")]
#[derive(Clone, PartialEq)]
pub struct PyIGSCompression {
    pub(crate) inner: igs::IGSCompression,
}

#[pymethods]
impl PyIGSCompression {
    /// Uncompressed.
    #[classattr]
    #[allow(non_snake_case)]
    fn NONE() -> Self {
        Self { inner: igs::IGSCompression::None }
    }

    /// gzip (``.gz``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GZIP() -> Self {
        Self { inner: igs::IGSCompression::Gzip }
    }

    /// Unix ``compress`` (``.Z``).
    #[classattr]
    #[allow(non_snake_case)]
    fn UNIX_COMPRESS() -> Self {
        Self { inner: igs::IGSCompression::UnixCompress }
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("IGSCompression.{}", self.name())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

impl PyIGSCompression {
    fn name(&self) -> &'static str {
        match self.inner {
            igs::IGSCompression::None => "NONE",
            igs::IGSCompression::Gzip => "GZIP",
            igs::IGSCompression::UnixCompress => "UNIX_COMPRESS",
        }
    }
}

/// Which of the two IGS archive file-name conventions a name follows.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     name = bh.IGSProductFileName.parse("igs21000.sp3.Z")
///     assert name.name_style == bh.IGSNameStyle.SHORT
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSNameStyle")]
#[derive(Clone, PartialEq)]
pub struct PyIGSNameStyle {
    pub(crate) inner: igs::IGSNameStyle,
}

#[pymethods]
impl PyIGSNameStyle {
    /// ``CCCVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT`` names, used from GPS week 2238.
    #[classattr]
    #[allow(non_snake_case)]
    fn LONG() -> Self {
        Self { inner: igs::IGSNameStyle::Long }
    }

    /// ``cccwwwwd[_hh].ext`` names, used before GPS week 2238.
    #[classattr]
    #[allow(non_snake_case)]
    fn SHORT() -> Self {
        Self { inner: igs::IGSNameStyle::Short }
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("IGSNameStyle.{}", self.name())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

impl PyIGSNameStyle {
    fn name(&self) -> &'static str {
        match self.inner {
            igs::IGSNameStyle::Long => "LONG",
            igs::IGSNameStyle::Short => "SHORT",
        }
    }
}

/// A parsed IGS product file name.
///
/// Parses both the long names used from GPS week 2238
/// (``IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz``) and the short names used
/// before it (``igs21000.sp3.Z``). ``str()`` returns the name exactly as it
/// was parsed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     name = bh.IGSProductFileName.parse("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz")
///     assert name.center == "COD"
///     assert name.solution == bh.IGSSolutionType.FINAL
///     assert name.content == bh.IGSContentType.ORBIT
///     assert name.gps_week == 2340
///     assert name.span == 86400.0
///     assert name.sampling == 300.0
///     assert name.is_sp3
///     assert str(name) == "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSProductFileName")]
#[derive(Clone)]
pub struct PyIGSProductFileName {
    pub(crate) inner: igs::IGSProductFileName,
}

#[pymethods]
impl PyIGSProductFileName {
    /// Parse a long or short IGS product file name.
    ///
    /// Args:
    ///     name (str): File name without directory components.
    ///
    /// Returns:
    ///     IGSProductFileName: The parsed name.
    ///
    /// Raises:
    ///     BraheError: If the name matches neither convention or a field is out of range.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     name = bh.IGSProductFileName.parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz")
    ///     assert name.center == "IGS"
    ///     assert name.gps_week == 2340
    ///     assert name.is_sp3
    ///
    ///     short = bh.IGSProductFileName.parse("igs23400.sp3.Z")
    ///     assert short.name_style == bh.IGSNameStyle.SHORT
    ///     ```
    #[staticmethod]
    fn parse(name: &str) -> PyResult<Self> {
        igs::IGSProductFileName::parse(name)
            .map(|inner| Self { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Which naming convention the name follows.
    ///
    /// Returns:
    ///     IGSNameStyle: ``LONG`` or ``SHORT``.
    #[getter]
    fn name_style(&self) -> PyIGSNameStyle {
        PyIGSNameStyle { inner: self.inner.name_style() }
    }

    /// Analysis-center code in uppercase, for example ``COD``.
    ///
    /// Returns:
    ///     str: Center code.
    #[getter]
    fn center(&self) -> String {
        self.inner.center().to_string()
    }

    /// Product version digit. Only long names carry one.
    ///
    /// Returns:
    ///     int | None: Version, or None for short names.
    #[getter]
    fn version(&self) -> Option<u8> {
        self.inner.version()
    }

    /// Campaign code, for example ``OPS`` or ``MGX``. Only long names carry one.
    ///
    /// Returns:
    ///     str | None: Campaign code, or None for short names.
    #[getter]
    fn campaign(&self) -> Option<String> {
        self.inner.campaign().map(str::to_string)
    }

    /// Solution type of the product.
    ///
    /// Returns:
    ///     IGSSolutionType: Solution type.
    #[getter]
    fn solution(&self) -> PyIGSSolutionType {
        PyIGSSolutionType { inner: self.inner.solution().clone() }
    }

    /// First epoch covered by the product, in GPS time.
    ///
    /// For long names this is the start time in the name. For short ultra-rapid
    /// names (igu, igv, siu, ...) the named week, day and hour mark the start of
    /// the predicted half of the two-day file, so coverage starts 24 hours
    /// earlier and can fall in the previous GPS week, whereas gps_week stays the
    /// named week.
    ///
    /// Returns:
    ///     Epoch: Start epoch.
    #[getter]
    fn start_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.inner.start_epoch() }
    }

    /// GPS week of the archive directory that holds the product.
    ///
    /// This is the week in the file name. It equals the week of start_epoch
    /// except for short ultra-rapid names issued on the first day of a week,
    /// whose coverage starts in the previous week.
    ///
    /// Returns:
    ///     int: GPS week number.
    #[getter]
    fn gps_week(&self) -> u32 {
        self.inner.gps_week()
    }

    /// Time span covered by the product, in seconds.
    ///
    /// Returns:
    ///     float | None: Span in seconds, or None for weekly short-name products.
    #[getter]
    fn span(&self) -> Option<f64> {
        self.inner.span()
    }

    /// Sampling interval of the product, in seconds. Only long names carry one.
    ///
    /// Returns:
    ///     float | None: Sampling interval in seconds, or None for short names.
    #[getter]
    fn sampling(&self) -> Option<f64> {
        self.inner.sampling()
    }

    /// Content type of the product.
    ///
    /// Returns:
    ///     IGSContentType: Content type.
    #[getter]
    fn content(&self) -> PyIGSContentType {
        PyIGSContentType { inner: self.inner.content().clone() }
    }

    /// File format of the product.
    ///
    /// Returns:
    ///     IGSFileFormat: File format.
    #[getter]
    fn format(&self) -> PyIGSFileFormat {
        PyIGSFileFormat { inner: self.inner.format().clone() }
    }

    /// Compression applied to the file.
    ///
    /// Returns:
    ///     IGSCompression: Compression.
    #[getter]
    fn compression(&self) -> PyIGSCompression {
        PyIGSCompression { inner: self.inner.compression() }
    }

    /// Whether the product is an SP3 orbit file.
    ///
    /// Returns:
    ///     bool: True for orbit products in SP3 format.
    #[getter]
    fn is_sp3(&self) -> bool {
        self.inner.is_sp3()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("IGSProductFileName('{}')", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner.to_string())
    }
}

/// An IGS analysis center: its product file-name code and institution name.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     center = bh.IGSAnalysisCenter.from_code("cod")
///     assert center.code == "COD"
///     assert center.name == "Center for Orbit Determination in Europe"
///     assert bh.IGSAnalysisCenter.from_code("XYZ") is None
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSAnalysisCenter")]
#[derive(Clone)]
pub struct PyIGSAnalysisCenter {
    pub(crate) inner: igs::IGSAnalysisCenter,
}

#[pymethods]
impl PyIGSAnalysisCenter {
    /// Look up an analysis center by its product file-name code.
    ///
    /// Short-name combination and variant codes such as IGL, IGV and COF are not
    /// analysis centers in the table and return None.
    ///
    /// Args:
    ///     code (str): Three-letter center code, case-insensitive.
    ///
    /// Returns:
    ///     IGSAnalysisCenter | None: The matching center, or None if the code is not in the table.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     center = bh.IGSAnalysisCenter.from_code("cod")
    ///     assert center.code == "COD"
    ///     assert bh.IGSAnalysisCenter.from_code("IGV") is None
    ///     ```
    #[staticmethod]
    fn from_code(code: &str) -> Option<Self> {
        igs::IGSAnalysisCenter::from_code(code).map(|center| Self { inner: *center })
    }

    /// Three-letter code used in product file names, uppercase.
    ///
    /// Returns:
    ///     str: Center code.
    #[getter]
    fn code(&self) -> String {
        self.inner.code.to_string()
    }

    /// Full name of the institution.
    ///
    /// Returns:
    ///     str: Institution name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.to_string()
    }

    fn __str__(&self) -> String {
        self.inner.code.to_string()
    }

    fn __repr__(&self) -> String {
        format!("IGSAnalysisCenter(code='{}', name='{}')", self.inner.code, self.inner.name)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

/// List every IGS analysis center in the table.
///
/// Returns:
///     list[IGSAnalysisCenter]: The centers, ordered by code with ``IGS`` last.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     codes = [center.code for center in bh.list_analysis_centers()]
///     assert "JPL" in codes
///     ```
#[pyfunction]
#[pyo3(name = "list_analysis_centers")]
fn py_list_analysis_centers() -> Vec<PyIGSAnalysisCenter> {
    igs::list_analysis_centers()
        .iter()
        .map(|center| PyIGSAnalysisCenter { inner: *center })
        .collect()
}

/// Convert a Rust error into the Python `BraheError`.
fn igs_py_err(error: RustBraheError) -> PyErr {
    BraheError::new_err(error.to_string())
}

/// Extract an `igs::IGSProductSource` from an `IGSProductSource` or its name.
fn extract_igs_product_source(obj: &Bound<'_, PyAny>) -> PyResult<igs::IGSProductSource> {
    if let Ok(source) = obj.extract::<PyIGSProductSource>() {
        return Ok(source.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return [
            igs::IGSProductSource::CDDIS,
            igs::IGSProductSource::Garner,
            igs::IGSProductSource::BKG,
        ]
        .into_iter()
        .find(|source| source.to_string().eq_ignore_ascii_case(text.trim()))
        .ok_or_else(|| {
            exceptions::PyValueError::new_err(format!(
                "unknown IGS product source '{text}', expected CDDIS, GARNER or BKG"
            ))
        });
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an IGSProductSource or a source name string",
    ))
}

/// Extract an `igs::IGSSolutionType` from an `IGSSolutionType` or its code.
fn extract_igs_solution_type(obj: &Bound<'_, PyAny>) -> PyResult<igs::IGSSolutionType> {
    if let Ok(solution) = obj.extract::<PyIGSSolutionType>() {
        return Ok(solution.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return Ok(igs::IGSSolutionType::parse(&text));
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an IGSSolutionType or a solution code string",
    ))
}

/// Extract an `igs::IGSContentType` from an `IGSContentType` or its code.
fn extract_igs_content_type(obj: &Bound<'_, PyAny>) -> PyResult<igs::IGSContentType> {
    if let Ok(content) = obj.extract::<PyIGSContentType>() {
        return Ok(content.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return Ok(igs::IGSContentType::parse(&text));
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an IGSContentType or a content code string",
    ))
}

/// Extract an `igs::IGSFileFormat` from an `IGSFileFormat` or its code.
fn extract_igs_file_format(obj: &Bound<'_, PyAny>) -> PyResult<igs::IGSFileFormat> {
    if let Ok(format) = obj.extract::<PyIGSFileFormat>() {
        return Ok(format.inner);
    }
    if let Ok(text) = obj.extract::<String>() {
        return Ok(igs::IGSFileFormat::parse(&text));
    }
    Err(exceptions::PyTypeError::new_err(
        "expected an IGSFileFormat or a file format code string",
    ))
}

/// Build an `igs::IGSProductFilter` from the keyword arguments of a listing call.
fn igs_product_filter(
    center: Option<&str>,
    solution: Option<&Bound<'_, PyAny>>,
    content: Option<&Bound<'_, PyAny>>,
    format: Option<&Bound<'_, PyAny>>,
) -> PyResult<igs::IGSProductFilter> {
    let mut filter = igs::IGSProductFilter::new();
    if let Some(center) = center {
        filter = filter.with_center(center);
    }
    if let Some(solution) = solution {
        filter = filter.with_solution(extract_igs_solution_type(solution)?);
    }
    if let Some(content) = content {
        filter = filter.with_content(extract_igs_content_type(content)?);
    }
    if let Some(format) = format {
        filter = filter.with_format(extract_igs_file_format(format)?);
    }
    Ok(filter)
}

/// Convert cached or written paths to strings.
fn igs_path_strings(paths: Vec<PathBuf>) -> Vec<String> {
    paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

/// Public archive that serves IGS products.
///
/// ``CDDIS`` holds the full archive and requires NASA Earthdata credentials.
/// ``GARNER`` (the Scripps Institution of Oceanography mirror) holds the full
/// archive from GPS week 576 and is served over unencrypted HTTP. ``BKG``
/// holds GPS weeks 2238 and later, and only the IGS combination and CAS
/// products.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     source = bh.IGSProductSource.BKG
///     assert source.default_base_url == "https://igs.bkg.bund.de/root_ftp/IGS/products/"
///     assert str(source) == "BKG"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSProductSource")]
#[derive(Clone, PartialEq)]
pub struct PyIGSProductSource {
    pub(crate) inner: igs::IGSProductSource,
}

#[pymethods]
impl PyIGSProductSource {
    /// NASA Crustal Dynamics Data Information System; requires Earthdata credentials.
    #[classattr]
    #[allow(non_snake_case)]
    fn CDDIS() -> Self {
        Self { inner: igs::IGSProductSource::CDDIS }
    }

    /// Scripps Institution of Oceanography Garner archive, served over plain HTTP.
    #[classattr]
    #[allow(non_snake_case)]
    fn GARNER() -> Self {
        Self { inner: igs::IGSProductSource::Garner }
    }

    /// German Federal Agency for Cartography and Geodesy archive.
    #[classattr]
    #[allow(non_snake_case)]
    fn BKG() -> Self {
        Self { inner: igs::IGSProductSource::BKG }
    }

    /// Base URL of the source's product directory, with a trailing slash.
    ///
    /// Returns:
    ///     str: Directory URL that contains one subdirectory per GPS week.
    #[getter]
    fn default_base_url(&self) -> &'static str {
        self.inner.default_base_url()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            igs::IGSProductSource::CDDIS => "IGSProductSource.CDDIS".to_string(),
            igs::IGSProductSource::Garner => "IGSProductSource.GARNER".to_string(),
            igs::IGSProductSource::BKG => "IGSProductSource.BKG".to_string(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}

/// A product file listed in an IGS archive's GPS week directory.
///
/// Products are returned by ``IGSProductsClient`` listings and passed back to
/// its download methods. All properties are read-only.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.IGSProductsClient(source=bh.IGSProductSource.GARNER)
///     epoch = bh.Epoch(2024, 11, 10, 0, 0, 0.0, 0.0, time_system=bh.GPS)
///     product = client.list_products(epoch, center="COD", solution="FIN", content="ORB")[0]
///     print(product.name, product.url, product.gps_week)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSProduct")]
#[derive(Clone)]
pub struct PyIGSProduct {
    pub(crate) inner: igs::IGSProduct,
}

#[pymethods]
impl PyIGSProduct {
    /// Parsed fields of the file name.
    ///
    /// Returns:
    ///     IGSProductFileName: The parsed name.
    #[getter]
    fn file_name(&self) -> PyIGSProductFileName {
        PyIGSProductFileName { inner: self.inner.file_name.clone() }
    }

    /// File name as published.
    ///
    /// Returns:
    ///     str: File name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// URL the file is served from.
    ///
    /// Returns:
    ///     str: File URL.
    #[getter]
    fn url(&self) -> String {
        self.inner.url.clone()
    }

    /// File size in bytes, when the listing gives it.
    ///
    /// Returns:
    ///     int | None: Size in bytes, or None when the listing has no sizes.
    #[getter]
    fn size(&self) -> Option<u64> {
        self.inner.size
    }

    /// GPS week of the archive directory the file is listed in.
    ///
    /// Returns:
    ///     int: GPS week number.
    #[getter]
    fn gps_week(&self) -> u32 {
        self.inner.gps_week
    }

    fn __str__(&self) -> String {
        self.inner.name.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "IGSProduct(name='{}', url='{}', gps_week={})",
            self.inner.name, self.inner.url, self.inner.gps_week
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Client for IGS precise-product archives.
///
/// The client reads from one ``IGSProductSource``. With no ``source``, it
/// uses CDDIS when the ``CDDIS_EARTHDATA_USERNAME`` and
/// ``CDDIS_EARTHDATA_PASSWORD`` environment variables are both non-empty.
/// Otherwise it prints a warning to standard error, once per process, and
/// uses the Garner mirror, which is served over unencrypted HTTP. A client
/// built with an explicit source never falls back to another source.
///
/// Passing ``cddis_username`` and ``cddis_password`` selects CDDIS with those
/// credentials. A CDDIS client built without them reads the environment
/// variables. Credentials are sent only to the NASA Earthdata login service
/// and are never shown by ``repr``.
///
/// Requests are blocking, retried on transient failures, and honour
/// ``BRAHE_NETWORK_MODE``. Listings and downloaded files are cached under
/// ``get_igs_cache_dir()``. Listings are cached per source, so a client with
/// a different ``base_url`` shares the listing cache of the source it stands
/// in for. The cache directory is not coordinated across processes or across
/// clients sharing it.
///
/// Args:
///     source (IGSProductSource | str, optional): Archive to read from, or its name (``"CDDIS"``, ``"GARNER"``, ``"BKG"``, case-insensitive). Default: chosen from the environment as above.
///     base_url (str, optional): Directory URL that contains one subdirectory per GPS week, for mirrors and tests. Requires ``source``. A trailing ``/`` is added when missing.
///     cddis_username (str, optional): NASA Earthdata username. Requires ``cddis_password``, a CDDIS or unset ``source``, and no ``base_url``.
///     cddis_password (str, optional): NASA Earthdata password. Requires ``cddis_username``.
///     cache_max_age (float, optional): Seconds a cached directory listing is served before it is refreshed. Default: 3600.0.
///     max_retries (int, optional): Extra attempts after the first on transient failures. Default: 3.
///
/// Raises:
///     ValueError: If ``base_url`` is given without ``source``, only one credential is given, credentials are given with a non-CDDIS source or with ``base_url``, or ``source`` is not a known source name.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.IGSProductsClient(source="GARNER")
///     start = bh.Epoch(2024, 11, 11, 0, 0, 0.0, 0.0, time_system=bh.GPS)
///     end = bh.Epoch(2024, 11, 12, 0, 0, 0.0, 0.0, time_system=bh.GPS)
///     files = client.download_sp3_range(start, end, "COD", bh.IGSSolutionType.FINAL)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "IGSProductsClient")]
pub struct PyIGSProductsClient {
    inner: igs::IGSProductsClient,
}

#[pymethods]
impl PyIGSProductsClient {
    #[new]
    #[pyo3(signature = (
        *, source=None, base_url=None, cddis_username=None, cddis_password=None,
        cache_max_age=None, max_retries=None
    ))]
    fn new(
        source: Option<&Bound<'_, PyAny>>,
        base_url: Option<&str>,
        cddis_username: Option<&str>,
        cddis_password: Option<&str>,
        cache_max_age: Option<f64>,
        max_retries: Option<u32>,
    ) -> PyResult<Self> {
        let source = source.map(extract_igs_product_source).transpose()?;
        let credentials = match (cddis_username, cddis_password) {
            (Some(username), Some(password)) => Some((username, password)),
            (None, None) => None,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "cddis_username and cddis_password must be given together",
                ));
            }
        };
        let mut client = match (source, base_url, credentials) {
            (None, Some(_), _) => {
                return Err(exceptions::PyValueError::new_err(
                    "base_url requires source: name the archive the URL stands in for",
                ));
            }
            (_, Some(_), Some(_)) => {
                return Err(exceptions::PyValueError::new_err(
                    "cddis_username and cddis_password cannot be combined with base_url; a CDDIS client with a base_url reads CDDIS_EARTHDATA_USERNAME and CDDIS_EARTHDATA_PASSWORD",
                ));
            }
            (Some(source), _, Some(_)) if source != igs::IGSProductSource::CDDIS => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "cddis_username and cddis_password apply only to the CDDIS source, not {source}"
                )));
            }
            (_, None, Some((username, password))) => {
                igs::IGSProductsClient::with_cddis_credentials(username, password)
            }
            (Some(source), Some(url), None) => igs::IGSProductsClient::with_base_url(source, url),
            (Some(source), None, None) => igs::IGSProductsClient::with_source(source),
            (None, None, None) => igs::IGSProductsClient::new(),
        };
        if let Some(age) = cache_max_age {
            client = client.with_cache_age(age);
        }
        if let Some(n) = max_retries {
            client = client.max_retries(n);
        }
        Ok(Self { inner: client })
    }

    /// Source the client reads from.
    ///
    /// Returns:
    ///     IGSProductSource: The archive.
    #[getter]
    fn source(&self) -> PyIGSProductSource {
        PyIGSProductSource { inner: self.inner.source() }
    }

    /// Base URL of the product directory, with a trailing slash.
    ///
    /// Returns:
    ///     str: The directory URL.
    #[getter]
    fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    /// Directory holding cached listings and product files.
    ///
    /// Returns:
    ///     str: ``$BRAHE_CACHE/igs``, created if missing.
    ///
    /// Raises:
    ///     BraheError: If the cache directory cannot be created.
    fn cache_dir(&self) -> PyResult<String> {
        self.inner
            .cache_dir()
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(igs_py_err)
    }

    /// Lists the GPS weeks that have a directory on the source.
    ///
    /// Reads the source's top-level listing and keeps the entries named by a
    /// four-digit week number, such as ``0576/`` or ``2340/``.
    ///
    /// Returns:
    ///     list[int]: GPS week numbers in ascending order.
    ///
    /// Raises:
    ///     BraheError: If the listing cannot be served from the cache or fetched.
    fn list_weeks(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        py.detach(|| self.inner.list_weeks()).map_err(igs_py_err)
    }

    /// Lists the file names in the GPS week directory containing ``epoch``.
    ///
    /// Every file is returned as published, including names that are not IGS
    /// products such as ``readme.txt``. Subdirectories, sort links, and the
    /// parent-directory link are left out.
    ///
    /// Args:
    ///     epoch (Epoch): Any instant in the GPS week to list.
    ///
    /// Returns:
    ///     list[str]: File names in listing order.
    ///
    /// Raises:
    ///     BraheError: If the week directory does not exist on the source, or the listing cannot be served from the cache or fetched.
    fn list_files(&self, py: Python<'_>, epoch: &PyEpoch) -> PyResult<Vec<String>> {
        let epoch = epoch.obj;
        py.detach(|| self.inner.list_files(epoch)).map_err(igs_py_err)
    }

    /// Lists the products in the GPS week directory containing ``epoch``.
    ///
    /// Files whose names parse as IGS product names are returned; other files
    /// are skipped. Each filter left as None matches every product.
    ///
    /// Args:
    ///     epoch (Epoch): Any instant in the GPS week to list.
    ///     center (str, optional): Analysis center code, such as ``"COD"``, in any case.
    ///     solution (IGSSolutionType | str, optional): Solution type or its code, such as ``"FIN"``.
    ///     content (IGSContentType | str, optional): Content type or its code, such as ``"ORB"``.
    ///     format (IGSFileFormat | str, optional): File format or its code, such as ``"SP3"``.
    ///
    /// Returns:
    ///     list[IGSProduct]: Matching products in listing order.
    ///
    /// Raises:
    ///     BraheError: If the week directory does not exist on the source, or the listing cannot be served from the cache or fetched.
    #[pyo3(signature = (epoch, *, center=None, solution=None, content=None, format=None))]
    fn list_products(
        &self,
        py: Python<'_>,
        epoch: &PyEpoch,
        center: Option<&str>,
        solution: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyIGSProduct>> {
        let epoch = epoch.obj;
        let filter = igs_product_filter(center, solution, content, format)?;
        py.detach(|| self.inner.list_products(epoch, &filter))
            .map(|products| {
                products
                    .into_iter()
                    .map(|inner| PyIGSProduct { inner })
                    .collect()
            })
            .map_err(igs_py_err)
    }

    /// Lists the products whose coverage overlaps ``[start, end]``.
    ///
    /// A product covers ``[start_epoch, start_epoch + span)``. A product with
    /// no span, such as a short-name weekly file, covers the GPS week of its
    /// archive directory. The week directories containing ``start`` through
    /// ``end`` are listed, plus those holding products that begin up to three
    /// days before ``start`` or are filed up to one day after ``end``. Such
    /// an extra week that does not exist on the source is skipped. When
    /// ``BRAHE_NETWORK_MODE`` forbids fetching a listing, the extra weeks are
    /// served only from the cache and skipped when they are not cached; a
    /// week containing part of the range must still be cached.
    ///
    /// Args:
    ///     start (Epoch): Start of the time range.
    ///     end (Epoch): End of the time range, inclusive.
    ///     center (str, optional): Analysis center code, such as ``"COD"``, in any case.
    ///     solution (IGSSolutionType | str, optional): Solution type or its code, such as ``"FIN"``.
    ///     content (IGSContentType | str, optional): Content type or its code, such as ``"ORB"``.
    ///     format (IGSFileFormat | str, optional): File format or its code, such as ``"SP3"``.
    ///
    /// Returns:
    ///     list[IGSProduct]: Matching products ordered by start epoch, then name.
    ///
    /// Raises:
    ///     BraheError: If ``end`` is before ``start``, a week directory containing part of the range does not exist on the source, or a listing cannot be served from the cache or fetched.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start, end, *, center=None, solution=None, content=None, format=None))]
    fn list_products_between(
        &self,
        py: Python<'_>,
        start: &PyEpoch,
        end: &PyEpoch,
        center: Option<&str>,
        solution: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyIGSProduct>> {
        let (start, end) = (start.obj, end.obj);
        let filter = igs_product_filter(center, solution, content, format)?;
        py.detach(|| self.inner.list_products_between(start, end, &filter))
            .map(|products| {
                products
                    .into_iter()
                    .map(|inner| PyIGSProduct { inner })
                    .collect()
            })
            .map_err(igs_py_err)
    }

    /// Downloads a product into the cache and returns the cached path.
    ///
    /// Files are cached under ``<cache_dir>/<source>/<week>/<name>``, with the
    /// source in lowercase, the product's four-digit archive week, and the
    /// name as published. A cached file is returned without a request and
    /// without an age check. When the file is not cached,
    /// ``BRAHE_NETWORK_MODE`` must allow the request. A CDDIS download signs
    /// in to NASA Earthdata with the client's credentials.
    ///
    /// Args:
    ///     product (IGSProduct): Product to download, as returned by a listing.
    ///
    /// Returns:
    ///     str: Path of the cached file, compressed as published.
    ///
    /// Raises:
    ///     BraheError: If the file does not exist on the source, the network mode forbids the request, the Earthdata login fails, the body does not match the name's compression, or the cache cannot be written.
    fn download(&self, py: Python<'_>, product: &PyIGSProduct) -> PyResult<String> {
        let product = product.inner.clone();
        py.detach(|| self.inner.download(&product))
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(igs_py_err)
    }

    /// Downloads a product, if it is not cached, and writes it to ``destination``.
    ///
    /// An existing directory, or a path whose last component has no
    /// extension, is treated as a directory (created if missing). A gzip
    /// product is written decompressed, in a directory under its published
    /// name without ``.gz``. Unix-compressed (``.Z``) and uncompressed
    /// products are written unchanged under their published name. A path
    /// with an extension is the file written. With ``keep_cached`` False the
    /// cached file is removed; with ``keep_cached`` True it stays.
    ///
    /// Args:
    ///     product (IGSProduct): Product to save, as returned by a listing.
    ///     destination (str): Directory or file path, outside the cache directory; any ``os.PathLike`` is accepted.
    ///     keep_cached (bool, optional): Leave the cached file in place. Default: False.
    ///
    /// Returns:
    ///     str: Path written.
    ///
    /// Raises:
    ///     BraheError: If ``destination`` resolves inside the cache directory, which is checked before anything is downloaded, the download fails, the gzip data is invalid, or the file cannot be written.
    #[pyo3(signature = (product, destination, keep_cached=false))]
    fn save(
        &self,
        py: Python<'_>,
        product: &PyIGSProduct,
        destination: PathBuf,
        keep_cached: bool,
    ) -> PyResult<String> {
        let product = product.inner.clone();
        py.detach(|| self.inner.save(&product, &destination, keep_cached))
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(igs_py_err)
    }

    /// Downloads every product whose coverage overlaps ``[start, end]``.
    ///
    /// The products are those ``list_products_between`` returns, each
    /// downloaded as ``download`` does.
    ///
    /// Args:
    ///     start (Epoch): Start of the time range.
    ///     end (Epoch): End of the time range, inclusive.
    ///     center (str, optional): Analysis center code, such as ``"COD"``, in any case.
    ///     solution (IGSSolutionType | str, optional): Solution type or its code, such as ``"FIN"``.
    ///     content (IGSContentType | str, optional): Content type or its code, such as ``"ORB"``.
    ///     format (IGSFileFormat | str, optional): File format or its code, such as ``"SP3"``.
    ///
    /// Returns:
    ///     list[str]: Cached paths, ordered by product start epoch, then name.
    ///
    /// Raises:
    ///     BraheError: If the listing fails or any download fails.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start, end, *, center=None, solution=None, content=None, format=None))]
    fn download_between(
        &self,
        py: Python<'_>,
        start: &PyEpoch,
        end: &PyEpoch,
        center: Option<&str>,
        solution: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<String>> {
        let (start, end) = (start.obj, end.obj);
        let filter = igs_product_filter(center, solution, content, format)?;
        py.detach(|| self.inner.download_between(start, end, &filter))
            .map(igs_path_strings)
            .map_err(igs_py_err)
    }

    /// Downloads every product ``download_between`` would and writes each into ``directory``.
    ///
    /// Every product is downloaded before any is written. Each is then
    /// written as ``save`` writes into a directory: gzip products
    /// decompressed under their name without ``.gz``, other products
    /// unchanged. ``directory`` is created if missing. With ``keep_cached``
    /// False the cached files are removed; with ``keep_cached`` True they
    /// stay.
    ///
    /// Args:
    ///     start (Epoch): Start of the time range.
    ///     end (Epoch): End of the time range, inclusive.
    ///     directory (str): Directory to write into, outside the cache directory; any ``os.PathLike`` is accepted.
    ///     center (str, optional): Analysis center code, such as ``"COD"``, in any case.
    ///     solution (IGSSolutionType | str, optional): Solution type or its code, such as ``"FIN"``.
    ///     content (IGSContentType | str, optional): Content type or its code, such as ``"ORB"``.
    ///     format (IGSFileFormat | str, optional): File format or its code, such as ``"SP3"``.
    ///     keep_cached (bool, optional): Leave the cached files in place. Default: False.
    ///
    /// Returns:
    ///     list[str]: Paths written, ordered by product start epoch, then name.
    ///
    /// Raises:
    ///     BraheError: If ``directory`` is an existing file or resolves inside the cache directory, both checked before anything is downloaded, or if the listing, a download, or a write fails.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        start, end, directory, *, center=None, solution=None, content=None, format=None,
        keep_cached=false
    ))]
    fn save_between(
        &self,
        py: Python<'_>,
        start: &PyEpoch,
        end: &PyEpoch,
        directory: PathBuf,
        center: Option<&str>,
        solution: Option<&Bound<'_, PyAny>>,
        content: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
        keep_cached: bool,
    ) -> PyResult<Vec<String>> {
        let (start, end) = (start.obj, end.obj);
        let filter = igs_product_filter(center, solution, content, format)?;
        py.detach(|| {
            self.inner
                .save_between(start, end, &filter, &directory, keep_cached)
        })
        .map(igs_path_strings)
        .map_err(igs_py_err)
    }

    /// Downloads an SP3 orbit product, if it is not cached, and parses it.
    ///
    /// The product is downloaded as ``download`` does and read with
    /// ``SP3.from_file``, which decompresses gzip data. Unix compress
    /// (``.Z``) data cannot be decompressed, so a ``.Z`` product is rejected
    /// before anything is downloaded; download it with ``save`` and
    /// decompress it externally.
    ///
    /// Args:
    ///     product (IGSProduct): SP3 product to download, as returned by a listing.
    ///
    /// Returns:
    ///     SP3: The parsed file.
    ///
    /// Raises:
    ///     BraheError: If the product is not an SP3 file or is Unix-compressed, both checked before anything is downloaded, or if the download or parse fails.
    fn download_sp3(&self, py: Python<'_>, product: &PyIGSProduct) -> PyResult<PySP3> {
        let product = product.inner.clone();
        py.detach(|| self.inner.download_sp3(&product))
            .map(|inner| PySP3 { inner })
            .map_err(igs_py_err)
    }

    /// Downloads and parses the daily SP3 orbit files of one analysis center
    /// and solution that cover ``[start, end]``.
    ///
    /// The files are the daily orbit files of every GPS day from the day
    /// containing ``start`` through the day containing ``end``. With ``pad``
    /// set, the days before and after are added when the source has their
    /// files, so that ``SP3Satellite.from_files`` uses centered fits at the
    /// span edges. From GPS week 2238 each day's file is the long-name
    /// one-day ``ORB`` SP3 file starting at 00:00; when several sampling
    /// intervals are published, the coarsest is used unless ``sampling`` is
    /// set, in which case only that interval is used. Before GPS week 2238 it
    /// is the short-name daily SP3 file, such as ``cod21003.sp3``, and
    /// ``sampling`` is ignored.
    ///
    /// Every file is chosen, and every error is raised, before anything is
    /// downloaded. Ultra-rapid orbits are rejected because successive issues
    /// overlap in time. A day inside the span with no file, or whose file is
    /// only published Unix-compressed, is an error; such a padding day is
    /// skipped.
    ///
    /// Args:
    ///     start (Epoch): Start of the time range.
    ///     end (Epoch): End of the time range, inclusive.
    ///     center (str): Analysis center code, in any case, for example ``"COD"``.
    ///     solution (IGSSolutionType | str): Solution type or its code, for example ``"FIN"``.
    ///     pad (bool, optional): Add the days before and after the span when available. Default: True.
    ///     sampling (float, optional): Sampling interval of the long-name files to use, seconds. Default: the coarsest published.
    ///
    /// Returns:
    ///     list[SP3]: Parsed daily files in time order.
    ///
    /// Raises:
    ///     BraheError: If ``solution`` is ultra-rapid, ``end`` is before ``start``, a week listing containing part of the span cannot be obtained, a day inside the span has no file on the source or only a Unix-compressed one, or a download or parse fails.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.IGSProductsClient(source="GARNER")
    ///     start = bh.Epoch(2024, 11, 11, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    ///     end = bh.Epoch(2024, 11, 12, 0, 0, 0.0, 0.0, time_system=bh.GPS)
    ///     files = client.download_sp3_range(start, end, "COD", "FIN")
    ///     satellites = bh.SP3Satellite.from_files(files)
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start, end, center, solution, *, pad=true, sampling=None))]
    fn download_sp3_range(
        &self,
        py: Python<'_>,
        start: &PyEpoch,
        end: &PyEpoch,
        center: &str,
        solution: &Bound<'_, PyAny>,
        pad: bool,
        sampling: Option<f64>,
    ) -> PyResult<Vec<PySP3>> {
        let (start, end) = (start.obj, end.obj);
        let solution = extract_igs_solution_type(solution)?;
        let mut options = igs::IGSRangeOptions::new().with_pad(pad);
        if let Some(sampling) = sampling {
            options = options.with_sampling(sampling);
        }
        py.detach(|| {
            self.inner
                .download_sp3_range(start, end, center, solution, &options)
        })
        .map(|files| files.into_iter().map(|inner| PySP3 { inner }).collect())
        .map_err(igs_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "IGSProductsClient(source={}, base_url='{}')",
            self.inner.source(),
            self.inner.base_url()
        )
    }
}
