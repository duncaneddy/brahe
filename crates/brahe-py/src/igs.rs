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
///     assert name.center == bh.IGSAnalysisCenter.COD
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
    ///     assert name.center == bh.IGSAnalysisCenter.IGS
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

    /// Analysis center that produced the product.
    ///
    /// Returns:
    ///     IGSAnalysisCenter: The analysis center.
    #[getter]
    fn center(&self) -> PyIGSAnalysisCenter {
        PyIGSAnalysisCenter { inner: self.inner.center().clone() }
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

/// Analysis center that produced an IGS product.
///
/// Codes other than the listed ones, such as the short-name combination codes
/// ``IGL`` and ``IGV``, are kept as they appear, uppercased, and compare equal
/// to other centers parsed from the same code.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     assert bh.IGSAnalysisCenter.parse("cod") == bh.IGSAnalysisCenter.COD
///     assert bh.IGSAnalysisCenter.JPL.code == "JPL"
///     assert bh.IGSAnalysisCenter.JPL.name == "Jet Propulsion Laboratory"
///     assert bh.IGSAnalysisCenter.parse("igl").code == "IGL"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "IGSAnalysisCenter")]
#[derive(Clone, PartialEq)]
pub struct PyIGSAnalysisCenter {
    pub(crate) inner: igs::IGSAnalysisCenter,
}

#[pymethods]
impl PyIGSAnalysisCenter {
    /// Center for Orbit Determination in Europe (``COD``).
    #[classattr]
    #[allow(non_snake_case)]
    fn COD() -> Self {
        Self { inner: igs::IGSAnalysisCenter::COD }
    }

    /// Natural Resources Canada (``EMR``).
    #[classattr]
    #[allow(non_snake_case)]
    fn EMR() -> Self {
        Self { inner: igs::IGSAnalysisCenter::EMR }
    }

    /// European Space Agency/ESOC (``ESA``).
    #[classattr]
    #[allow(non_snake_case)]
    fn ESA() -> Self {
        Self { inner: igs::IGSAnalysisCenter::ESA }
    }

    /// GFZ Helmholtz Centre for Geosciences (``GFZ``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GFZ() -> Self {
        Self { inner: igs::IGSAnalysisCenter::GFZ }
    }

    /// Geodetic Observatory Pecny (``GOP``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GOP() -> Self {
        Self { inner: igs::IGSAnalysisCenter::GOP }
    }

    /// Space geodesy team of the CNES (``GRG``).
    #[classattr]
    #[allow(non_snake_case)]
    fn GRG() -> Self {
        Self { inner: igs::IGSAnalysisCenter::GRG }
    }

    /// Geospatial Information Authority of Japan and Japan Aerospace Exploration Agency (``JGX``).
    #[classattr]
    #[allow(non_snake_case)]
    fn JGX() -> Self {
        Self { inner: igs::IGSAnalysisCenter::JGX }
    }

    /// Jet Propulsion Laboratory (``JPL``).
    #[classattr]
    #[allow(non_snake_case)]
    fn JPL() -> Self {
        Self { inner: igs::IGSAnalysisCenter::JPL }
    }

    /// Massachusetts Institute of Technology (``MIT``).
    #[classattr]
    #[allow(non_snake_case)]
    fn MIT() -> Self {
        Self { inner: igs::IGSAnalysisCenter::MIT }
    }

    /// NOAA/National Geodetic Survey (``NGS``).
    #[classattr]
    #[allow(non_snake_case)]
    fn NGS() -> Self {
        Self { inner: igs::IGSAnalysisCenter::NGS }
    }

    /// Scripps Institution of Oceanography (``SIO``).
    #[classattr]
    #[allow(non_snake_case)]
    fn SIO() -> Self {
        Self { inner: igs::IGSAnalysisCenter::SIO }
    }

    /// U.S. Naval Observatory (``USN``).
    #[classattr]
    #[allow(non_snake_case)]
    fn USN() -> Self {
        Self { inner: igs::IGSAnalysisCenter::USN }
    }

    /// Wuhan University (``WHU``).
    #[classattr]
    #[allow(non_snake_case)]
    fn WHU() -> Self {
        Self { inner: igs::IGSAnalysisCenter::WHU }
    }

    /// International GNSS Service combined products (``IGS``).
    #[classattr]
    #[allow(non_snake_case)]
    fn IGS() -> Self {
        Self { inner: igs::IGSAnalysisCenter::IGS }
    }

    /// Parse a center code.
    ///
    /// Matching is case-insensitive. Unlisted codes are kept, uppercased.
    ///
    /// Args:
    ///     code (str): Three-letter center code, for example ``"COD"``.
    ///
    /// Returns:
    ///     IGSAnalysisCenter: The parsed center.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.IGSAnalysisCenter.parse("gfz") == bh.IGSAnalysisCenter.GFZ
    ///     assert bh.IGSAnalysisCenter.parse("xyz").code == "XYZ"
    ///     ```
    #[staticmethod]
    fn parse(code: &str) -> Self {
        Self { inner: igs::IGSAnalysisCenter::parse(code) }
    }

    /// Three-letter code used in product file names, uppercase.
    ///
    /// Returns:
    ///     str: Center code.
    #[getter]
    fn code(&self) -> String {
        self.inner.code().to_string()
    }

    /// Name of the institution.
    ///
    /// Returns:
    ///     str | None: Institution name, or None for an unlisted code.
    #[getter]
    fn name(&self) -> Option<&'static str> {
        self.inner.name()
    }

    fn __str__(&self) -> String {
        self.inner.code().to_string()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            igs::IGSAnalysisCenter::Other(ref code) => {
                format!("IGSAnalysisCenter.parse('{}')", code)
            }
            ref center => format!("IGSAnalysisCenter.{}", center.code()),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        igs_hash(&self.inner)
    }
}
