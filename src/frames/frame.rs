/*!
 * Unified `Frame` type spanning celestial, orbit-relative, and body frames.
 *
 * [`Frame`] is the top-level frame identity used throughout `brahe`. It
 * covers three kinds of frame:
 *
 * 1. **Celestial** ([`Frame::Celestial`]) — a frame the frames router can
 *    evaluate analytically from an epoch alone ([`CelestialFrame`]).
 * 2. **Orbit-relative** ([`Frame::OrbitRelative`]) — a local orbital frame
 *    (RTN, LVLH, ...) of a specific object, rotating with the orbit or
 *    frozen as an inertial snapshot.
 * 3. **Body** ([`Frame::Body`]) — an object-local frame (spacecraft body,
 *    sensor, actuator, ...) that has no global transformation.
 *
 * Orbit-relative and body frames carry an `object: Option<`[`ObjectId`]`>`.
 * `None` is a pure label — what a data file can express before binding to a
 * specific object; `Some` identifies the object the frame is evaluable
 * against.
 */

use std::fmt;
use std::sync::Arc;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::frames::CelestialFrame;
use crate::utils::errors::BraheError;

/// String-backed object identity (e.g. `"LRO"`, `"2024-123A"`).
///
/// Cheap to clone (`Arc<str>` internally). Serializes as a plain JSON
/// string.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::ObjectId;
///
/// let id: ObjectId = "LRO".into();
/// assert_eq!(id.to_string(), "LRO");
/// assert_eq!(id, ObjectId::from("LRO".to_string()));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ObjectId(Arc<str>);

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ObjectId {
    fn from(value: &str) -> Self {
        ObjectId(Arc::from(value))
    }
}

impl From<String> for ObjectId {
    fn from(value: String) -> Self {
        ObjectId(Arc::from(value.as_str()))
    }
}

impl Serialize for ObjectId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

struct ObjectIdVisitor;

impl Visitor<'_> for ObjectIdVisitor {
    type Value = ObjectId;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string object identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(ObjectId::from(value))
    }
}

impl<'de> Deserialize<'de> for ObjectId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ObjectIdVisitor)
    }
}

/// Local orbital frame axes definitions.
///
/// `RTN` is the frame the SANA registries call `RSW`; brahe uses its
/// existing RTN vocabulary (`state_eci_to_rtn`, `covariance_rtn`).
///
/// # Examples
///
/// ```rust
/// use brahe::frames::OrbitRelativeKind;
///
/// assert_eq!(OrbitRelativeKind::RTN.to_string(), "RTN");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRelativeKind {
    /// Local-Vertical Local-Horizontal.
    LVLH,
    /// Radial / transverse (along-track) / normal (cross-track). SANA: RSW.
    RTN,
    /// Normal / tangential / cross-track.
    NTW,
    /// Tangential / normal / cross-track.
    TNW,
    /// Perifocal. SANA-registered only as an inertial-snapshot frame.
    PQW,
    /// Equinoctial. SANA-registered only as an inertial-snapshot frame.
    EQW,
    /// Topocentric south / east / zenith.
    SEZ,
    /// Velocity / normal / co-normal.
    VNC,
    /// Nadir / Sun / normal.
    NSW,
}

impl fmt::Display for OrbitRelativeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let token = match self {
            Self::LVLH => "LVLH",
            Self::RTN => "RTN",
            Self::NTW => "NTW",
            Self::TNW => "TNW",
            Self::PQW => "PQW",
            Self::EQW => "EQW",
            Self::SEZ => "SEZ",
            Self::VNC => "VNC",
            Self::NSW => "NSW",
        };
        write!(f, "{}", token)
    }
}

/// Rotating vs. quasi-inertial snapshot variant of a local orbital frame.
///
/// - **Rotating**: True local orbital frame, rotating with the orbit.
/// - **Inertial**: Quasi-inertial frame frozen at each evaluation time.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::OrbitRelativeVariant;
///
/// assert_eq!(OrbitRelativeVariant::Rotating.to_string(), "rotating");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRelativeVariant {
    /// True local orbital frame, rotating with the orbit.
    Rotating,
    /// Quasi-inertial frame frozen at each evaluation time.
    Inertial,
}

impl fmt::Display for OrbitRelativeVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rotating => write!(f, "rotating"),
            Self::Inertial => write!(f, "inertial"),
        }
    }
}

/// A local orbital frame: a kind plus rotating/inertial-snapshot variant.
///
/// Represents a frame that rotates with or tracks the orbit, composed of:
/// - A frame construction type (e.g., RTN, LVLH) defining the axes
/// - A variant indicating whether the frame rotates with the orbit or is
///   frozen
///
/// Fields are private: per the SANA registry, `EQW` and `PQW` exist only as
/// inertial-snapshot frames, so construction goes through
/// [`OrbitRelativeFrame::new`] to reject that combination rather than
/// allowing it and erroring later.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::{OrbitRelativeFrame, OrbitRelativeKind, OrbitRelativeVariant};
///
/// let rtn = OrbitRelativeFrame::new(OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating);
/// assert!(rtn.is_ok());
/// assert_eq!(rtn.unwrap().to_string(), "RTN (rotating)");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrbitRelativeFrame {
    kind: OrbitRelativeKind,
    variant: OrbitRelativeVariant,
}

impl OrbitRelativeFrame {
    /// Constructs an orbit-relative frame, validating the kind/variant
    /// combination.
    ///
    /// Per the SANA orbit-relative frame registry, `EQW` and `PQW` are
    /// defined only as inertial-snapshot frames; they have no rotating
    /// variant.
    ///
    /// # Arguments
    /// * `kind` - The frame construction (axes definition)
    /// * `variant` - Rotating (true local orbital frame) or inertial
    ///   snapshot
    ///
    /// # Returns
    /// * `Ok(OrbitRelativeFrame)`: If the combination is valid
    /// * `Err(BraheError)`: If `kind` is `EQW` or `PQW` and `variant` is
    ///   `Rotating`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{OrbitRelativeFrame, OrbitRelativeKind, OrbitRelativeVariant};
    ///
    /// let rtn = OrbitRelativeFrame::new(OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating);
    /// assert!(rtn.is_ok());
    ///
    /// let eqw_rotating = OrbitRelativeFrame::new(OrbitRelativeKind::EQW, OrbitRelativeVariant::Rotating);
    /// assert!(eqw_rotating.is_err());
    /// ```
    pub fn new(kind: OrbitRelativeKind, variant: OrbitRelativeVariant) -> Result<Self, BraheError> {
        if matches!(kind, OrbitRelativeKind::EQW | OrbitRelativeKind::PQW)
            && variant == OrbitRelativeVariant::Rotating
        {
            return Err(BraheError::Error(format!(
                "orbit-relative frame {} exists only as an inertial SANA frame and cannot be \
                 constructed with the rotating variant",
                kind
            )));
        }
        Ok(Self { kind, variant })
    }

    /// Returns the frame construction (axes definition).
    ///
    /// # Returns
    /// `OrbitRelativeKind`: The frame construction
    pub fn kind(&self) -> OrbitRelativeKind {
        self.kind
    }

    /// Returns the rotating/inertial-snapshot variant.
    ///
    /// # Returns
    /// `OrbitRelativeVariant`: Rotating or inertial
    pub fn variant(&self) -> OrbitRelativeVariant {
        self.variant
    }
}

impl fmt::Display for OrbitRelativeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.kind, self.variant)
    }
}

/// An object-local spacecraft body frame with an optional instance
/// designator.
///
/// Variants represent different spacecraft subsystems and sensors. The
/// optional `String` designator (e.g., `SCBody(Some("1"))`) is appended to
/// the frame name in [`Display`](fmt::Display) output (e.g., `SC_BODY_1`).
///
/// # Examples
///
/// ```rust
/// use brahe::frames::BodyFrame;
///
/// assert_eq!(BodyFrame::CSS(Some("1".to_string())).to_string(), "CSS_1");
/// assert_eq!(BodyFrame::SCBody(None).to_string(), "SC_BODY");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyFrame {
    /// Accelerometer frame.
    ACC(Option<String>),
    /// Actuator frame.
    Actuator(Option<String>),
    /// Autonomous star tracker frame.
    AST(Option<String>),
    /// Coarse sun sensor frame.
    CSS(Option<String>),
    /// Digital sun sensor frame.
    DSS(Option<String>),
    /// Earth sensor assembly frame.
    ESA(Option<String>),
    /// Gyroscope frame.
    GyroFrame(Option<String>),
    /// Inertial measurement unit frame.
    IMUFrame(Option<String>),
    /// Instrument frame.
    Instrument(Option<String>),
    /// Magnetic torque assembly frame.
    MTA(Option<String>),
    /// Reaction wheel frame.
    RW(Option<String>),
    /// Solar array frame.
    SA(Option<String>),
    /// Spacecraft body frame.
    SCBody(Option<String>),
    /// Generic sensor frame.
    Sensor(Option<String>),
    /// Star tracker frame.
    StarTracker(Option<String>),
    /// Three-axis magnetometer frame.
    TAM(Option<String>),
}

impl fmt::Display for BodyFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (family, suffix) = match self {
            Self::ACC(s) => ("ACC", s),
            Self::Actuator(s) => ("ACTUATOR", s),
            Self::AST(s) => ("AST", s),
            Self::CSS(s) => ("CSS", s),
            Self::DSS(s) => ("DSS", s),
            Self::ESA(s) => ("ESA", s),
            Self::GyroFrame(s) => ("GYRO_FRAME", s),
            Self::IMUFrame(s) => ("IMU_FRAME", s),
            Self::Instrument(s) => ("INSTRUMENT", s),
            Self::MTA(s) => ("MTA", s),
            Self::RW(s) => ("RW", s),
            Self::SA(s) => ("SA", s),
            Self::SCBody(s) => ("SC_BODY", s),
            Self::Sensor(s) => ("SENSOR", s),
            Self::StarTracker(s) => ("STARTRACKER", s),
            Self::TAM(s) => ("TAM", s),
        };
        match suffix {
            Some(i) => write!(f, "{}_{}", family, i),
            None => write!(f, "{}", family),
        }
    }
}

/// Unified frame identity spanning celestial, orbit-relative, and body
/// frames.
///
/// `Frame` is the top-level frame type used throughout `brahe`. See the
/// [module documentation](self) for the three frame kinds.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::{Frame, CelestialFrame};
///
/// let rtn = Frame::RTN("SC");
/// assert_eq!(rtn.to_string(), "RTN (rotating)@SC");
///
/// let gcrf: Frame = CelestialFrame::GCRF.into();
/// assert_eq!(gcrf.to_string(), "GCRF");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Frame {
    /// Evaluable analytically from an epoch alone (existing frames router).
    Celestial(CelestialFrame),
    /// Local orbital frame of `object`. Evaluable when bound (`object` is
    /// `Some`) and the object is registered.
    OrbitRelative {
        /// Frame construction (axes definition).
        kind: OrbitRelativeKind,
        /// Rotating (true local orbital frame) or inertial snapshot.
        variant: OrbitRelativeVariant,
        /// Bound object, if any. `None` is a pure label.
        object: Option<ObjectId>,
    },
    /// Body/sensor/actuator frame of `object`. Evaluable when bound
    /// (`object` is `Some`) and its orientation chain is registered.
    Body {
        /// The body frame kind and optional instance designator.
        frame: BodyFrame,
        /// Bound object, if any. `None` is a pure label.
        object: Option<ObjectId>,
    },
}

impl Frame {
    /// Constructs a bound Radial/Transverse/Normal orbit-relative frame
    /// (rotating variant). SANA: RSW.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `RTN (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::RTN("SC").to_string(), "RTN (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn RTN(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::RTN,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Local-Vertical Local-Horizontal orbit-relative
    /// frame (rotating variant).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `LVLH (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::LVLH("SC").to_string(), "LVLH (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn LVLH(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::LVLH,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Normal/Tangential/cross-track orbit-relative
    /// frame (rotating variant).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `NTW (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::NTW("SC").to_string(), "NTW (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn NTW(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::NTW,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Tangential/Normal/cross-track orbit-relative
    /// frame (rotating variant).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `TNW (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::TNW("SC").to_string(), "TNW (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn TNW(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::TNW,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound topocentric South/East/Zenith orbit-relative
    /// frame (rotating variant).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `SEZ (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::SEZ("SC").to_string(), "SEZ (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SEZ(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::SEZ,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Velocity/Normal/Co-normal orbit-relative frame
    /// (rotating variant).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `VNC (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::VNC("SC").to_string(), "VNC (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn VNC(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::VNC,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Nadir/Sun/Normal orbit-relative frame (rotating
    /// variant).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `NSW (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::NSW("SC").to_string(), "NSW (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn NSW(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::NSW,
            OrbitRelativeVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Perifocal orbit-relative frame (inertial-snapshot
    /// variant; `PQW` is SANA-registered only as inertial).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `PQW (inertial)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::PQW("SC").to_string(), "PQW (inertial)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn PQW(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::PQW,
            OrbitRelativeVariant::Inertial,
            object,
        )
    }

    /// Constructs a bound Equinoctial orbit-relative frame
    /// (inertial-snapshot variant; `EQW` is SANA-registered only as
    /// inertial).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `EQW (inertial)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::EQW("SC").to_string(), "EQW (inertial)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn EQW(object: impl Into<ObjectId>) -> Frame {
        Frame::orbit_relative_unchecked(
            OrbitRelativeKind::EQW,
            OrbitRelativeVariant::Inertial,
            object,
        )
    }

    /// Constructs a bound orbit-relative frame without kind/variant
    /// validation, for combinations known valid by construction (all
    /// `Frame::<KIND>(object)` associated functions).
    fn orbit_relative_unchecked(
        kind: OrbitRelativeKind,
        variant: OrbitRelativeVariant,
        object: impl Into<ObjectId>,
    ) -> Frame {
        Frame::OrbitRelative {
            kind,
            variant,
            object: Some(object.into()),
        }
    }

    /// Constructs an orbit-relative frame, validating the kind/variant
    /// combination.
    ///
    /// General form of the `Frame::<KIND>(object)` constructors, for
    /// callers that hold a runtime `kind`/`variant` pair (e.g. parsed from
    /// a CCSDS file) and an optional, not-yet-bound object.
    ///
    /// # Arguments
    /// * `kind` - The frame construction (axes definition)
    /// * `variant` - Rotating (true local orbital frame) or inertial
    ///   snapshot
    /// * `object` - The bound object, or `None` for an unbound label
    ///
    /// # Returns
    /// * `Ok(Frame)`: The `OrbitRelative` frame, if the combination is
    ///   valid
    /// * `Err(BraheError)`: If `kind` is `EQW` or `PQW` and `variant` is
    ///   `Rotating`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{Frame, OrbitRelativeKind, OrbitRelativeVariant};
    ///
    /// let bound = Frame::orbit_relative(OrbitRelativeKind::RTN, OrbitRelativeVariant::Inertial, Some("SC".into()));
    /// assert!(bound.is_ok());
    ///
    /// let invalid = Frame::orbit_relative(OrbitRelativeKind::EQW, OrbitRelativeVariant::Rotating, None);
    /// assert!(invalid.is_err());
    /// ```
    pub fn orbit_relative(
        kind: OrbitRelativeKind,
        variant: OrbitRelativeVariant,
        object: Option<ObjectId>,
    ) -> Result<Frame, BraheError> {
        if matches!(kind, OrbitRelativeKind::EQW | OrbitRelativeKind::PQW)
            && variant == OrbitRelativeVariant::Rotating
        {
            return Err(BraheError::Error(format!(
                "orbit-relative frame {} exists only as an inertial SANA frame and cannot be \
                 constructed with the rotating variant",
                kind
            )));
        }
        Ok(Frame::OrbitRelative {
            kind,
            variant,
            object,
        })
    }

    /// Constructs a bound spacecraft body frame (no instance designator).
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `Frame`: The bound `SC_BODY` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::SC_BODY("SC").to_string(), "SC_BODY@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SC_BODY(object: impl Into<ObjectId>) -> Frame {
        Frame::body(object, BodyFrame::SCBody(None))
    }

    /// Constructs a bound coarse sun sensor frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `CSS_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::CSS("SC", "1").to_string(), "CSS_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn CSS(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::CSS(Some(designator.into())))
    }

    /// Constructs a bound accelerometer frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `ACC_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::ACC("SC", "1").to_string(), "ACC_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn ACC(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::ACC(Some(designator.into())))
    }

    /// Constructs a bound autonomous star tracker frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `AST_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::AST("SC", "1").to_string(), "AST_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn AST(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::AST(Some(designator.into())))
    }

    /// Constructs a bound digital sun sensor frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `DSS_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::DSS("SC", "1").to_string(), "DSS_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn DSS(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::DSS(Some(designator.into())))
    }

    /// Constructs a bound Earth sensor assembly frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `ESA_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::ESA("SC", "1").to_string(), "ESA_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn ESA(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::ESA(Some(designator.into())))
    }

    /// Constructs a bound gyroscope frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `GYRO_FRAME_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::GYRO_FRAME("SC", "1").to_string(), "GYRO_FRAME_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn GYRO_FRAME(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::GyroFrame(Some(designator.into())))
    }

    /// Constructs a bound inertial measurement unit frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `IMU_FRAME_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::IMU_FRAME("SC", "1").to_string(), "IMU_FRAME_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn IMU_FRAME(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::IMUFrame(Some(designator.into())))
    }

    /// Constructs a bound instrument frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The instrument instance designator (e.g. `"A"`)
    ///
    /// # Returns
    /// `Frame`: The bound `INSTRUMENT_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::INSTRUMENT("SC", "A").to_string(), "INSTRUMENT_A@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn INSTRUMENT(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::Instrument(Some(designator.into())))
    }

    /// Constructs a bound magnetic torque assembly frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The actuator instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `MTA_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::MTA("SC", "1").to_string(), "MTA_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn MTA(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::MTA(Some(designator.into())))
    }

    /// Constructs a bound reaction wheel frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The actuator instance designator (e.g. `"4"`)
    ///
    /// # Returns
    /// `Frame`: The bound `RW_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::RW("SC", "4").to_string(), "RW_4@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn RW(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::RW(Some(designator.into())))
    }

    /// Constructs a bound solar array frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The array instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `SA_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::SA("SC", "1").to_string(), "SA_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SA(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::SA(Some(designator.into())))
    }

    /// Constructs a bound generic sensor frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"10"`)
    ///
    /// # Returns
    /// `Frame`: The bound `SENSOR_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::SENSOR("SC", "10").to_string(), "SENSOR_10@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SENSOR(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::Sensor(Some(designator.into())))
    }

    /// Constructs a bound star tracker frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"2"`)
    ///
    /// # Returns
    /// `Frame`: The bound `STARTRACKER_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::STARTRACKER("SC", "2").to_string(), "STARTRACKER_2@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn STARTRACKER(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::StarTracker(Some(designator.into())))
    }

    /// Constructs a bound three-axis magnetometer frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `TAM_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::TAM("SC", "1").to_string(), "TAM_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn TAM(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::TAM(Some(designator.into())))
    }

    /// Constructs a bound actuator frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The actuator instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `Frame`: The bound `ACTUATOR_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::ACTUATOR("SC", "1").to_string(), "ACTUATOR_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn ACTUATOR(object: impl Into<ObjectId>, designator: impl Into<String>) -> Frame {
        Frame::body(object, BodyFrame::Actuator(Some(designator.into())))
    }

    /// Constructs a bound body frame, general form.
    ///
    /// Covers designator-less and non-standard [`BodyFrame`] cases beyond
    /// the family-specific constructors (e.g. `Frame::CSS`, `Frame::RW`).
    /// Use [`From<BodyFrame>`](Frame#impl-From<BodyFrame>-for-Frame) to
    /// construct an unbound (`object: None`) body frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `frame` - The body frame kind and optional instance designator
    ///
    /// # Returns
    /// `Frame`: The bound body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{Frame, BodyFrame};
    ///
    /// let frame = Frame::body("SC", BodyFrame::SCBody(None));
    /// assert_eq!(frame.to_string(), "SC_BODY@SC");
    /// ```
    pub fn body(object: impl Into<ObjectId>, frame: BodyFrame) -> Frame {
        Frame::Body {
            frame,
            object: Some(object.into()),
        }
    }

    /// Returns whether the frame is evaluable: a celestial frame, or an
    /// orbit-relative/body frame with a bound object.
    ///
    /// # Returns
    /// `bool`: `true` if the frame is bound (celestial frames are always
    /// bound)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{Frame, BodyFrame};
    ///
    /// assert!(Frame::SC_BODY("SC").is_bound());
    /// let unbound: Frame = BodyFrame::SCBody(None).into();
    /// assert!(!unbound.is_bound());
    /// ```
    pub fn is_bound(&self) -> bool {
        match self {
            Frame::Celestial(_) => true,
            Frame::OrbitRelative { object, .. } => object.is_some(),
            Frame::Body { object, .. } => object.is_some(),
        }
    }

    /// Returns the bound object, if any.
    ///
    /// # Returns
    /// `Option<&ObjectId>`: The bound object, or `None` for a celestial
    /// frame or an unbound orbit-relative/body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::Frame;
    ///
    /// assert_eq!(Frame::RTN("SC").object().unwrap().to_string(), "SC");
    /// ```
    pub fn object(&self) -> Option<&ObjectId> {
        match self {
            Frame::Celestial(_) => None,
            Frame::OrbitRelative { object, .. } => object.as_ref(),
            Frame::Body { object, .. } => object.as_ref(),
        }
    }
}

impl From<CelestialFrame> for Frame {
    fn from(frame: CelestialFrame) -> Self {
        Frame::Celestial(frame)
    }
}

impl From<OrbitRelativeFrame> for Frame {
    fn from(frame: OrbitRelativeFrame) -> Self {
        Frame::OrbitRelative {
            kind: frame.kind,
            variant: frame.variant,
            object: None,
        }
    }
}

impl From<BodyFrame> for Frame {
    fn from(frame: BodyFrame) -> Self {
        Frame::Body {
            frame,
            object: None,
        }
    }
}

impl fmt::Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Frame::Celestial(frame) => write!(f, "{}", frame),
            Frame::OrbitRelative {
                kind,
                variant,
                object,
            } => match object {
                Some(object) => write!(f, "{} ({})@{}", kind, variant, object),
                None => write!(f, "{} ({})", kind, variant),
            },
            Frame::Body { frame, object } => match object {
                Some(object) => write!(f, "{}@{}", frame, object),
                None => write!(f, "{}", frame),
            },
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn test_frame_constructors_and_display() {
        let f = Frame::RTN("SC");
        assert_eq!(f.to_string(), "RTN (rotating)@SC");
        assert!(f.is_bound());
        assert_eq!(f.object().unwrap().to_string(), "SC");
        // EQW/PQW default to Inertial so construction never errors
        assert_eq!(Frame::PQW("SC").to_string(), "PQW (inertial)@SC");
        assert_eq!(Frame::CSS("SC", "1").to_string(), "CSS_1@SC");
        assert_eq!(Frame::SC_BODY("SC").to_string(), "SC_BODY@SC");
        let unbound: Frame = BodyFrame::SCBody(None).into();
        assert!(!unbound.is_bound());
        assert_eq!(unbound.to_string(), "SC_BODY");
        let cel: Frame = CelestialFrame::GCRF.into();
        assert_eq!(cel.to_string(), "GCRF");
        assert!(cel.is_bound());
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_validation() {
        assert!(
            Frame::orbit_relative(OrbitRelativeKind::EQW, OrbitRelativeVariant::Rotating, None)
                .is_err()
        );
        assert!(
            Frame::orbit_relative(
                OrbitRelativeKind::RTN,
                OrbitRelativeVariant::Inertial,
                Some("SC".into())
            )
            .is_ok()
        );
    }

    #[test]
    #[parallel]
    fn test_frame_serde_round_trip() {
        for f in [
            Frame::RTN("SC"),
            Frame::CSS("SC", "1"),
            CelestialFrame::ITRF.into(),
            Frame::from(
                OrbitRelativeFrame::new(OrbitRelativeKind::LVLH, OrbitRelativeVariant::Rotating)
                    .unwrap(),
            ),
        ] {
            let s = serde_json::to_string(&f).unwrap();
            assert_eq!(serde_json::from_str::<Frame>(&s).unwrap(), f);
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_kind_display_all_variants() {
        let cases = [
            (OrbitRelativeKind::LVLH, "LVLH"),
            (OrbitRelativeKind::RTN, "RTN"),
            (OrbitRelativeKind::NTW, "NTW"),
            (OrbitRelativeKind::TNW, "TNW"),
            (OrbitRelativeKind::PQW, "PQW"),
            (OrbitRelativeKind::EQW, "EQW"),
            (OrbitRelativeKind::SEZ, "SEZ"),
            (OrbitRelativeKind::VNC, "VNC"),
            (OrbitRelativeKind::NSW, "NSW"),
        ];
        for (kind, expected) in cases {
            assert_eq!(kind.to_string(), expected);
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_variant_display() {
        assert_eq!(OrbitRelativeVariant::Rotating.to_string(), "rotating");
        assert_eq!(OrbitRelativeVariant::Inertial.to_string(), "inertial");
    }

    #[test]
    #[parallel]
    fn test_body_frame_display_all_variants() {
        let cases = [
            (BodyFrame::ACC(Some("1".to_string())), "ACC_1"),
            (BodyFrame::Actuator(None), "ACTUATOR"),
            (BodyFrame::AST(Some("1".to_string())), "AST_1"),
            (BodyFrame::CSS(Some("2".to_string())), "CSS_2"),
            (BodyFrame::DSS(Some("1".to_string())), "DSS_1"),
            (BodyFrame::ESA(Some("1".to_string())), "ESA_1"),
            (BodyFrame::GyroFrame(Some("1".to_string())), "GYRO_FRAME_1"),
            (BodyFrame::IMUFrame(Some("2".to_string())), "IMU_FRAME_2"),
            (BodyFrame::Instrument(Some("A".to_string())), "INSTRUMENT_A"),
            (BodyFrame::MTA(Some("1".to_string())), "MTA_1"),
            (BodyFrame::RW(Some("4".to_string())), "RW_4"),
            (BodyFrame::SA(Some("1".to_string())), "SA_1"),
            (BodyFrame::SCBody(None), "SC_BODY"),
            (BodyFrame::Sensor(Some("10".to_string())), "SENSOR_10"),
            (
                BodyFrame::StarTracker(Some("2".to_string())),
                "STARTRACKER_2",
            ),
            (BodyFrame::TAM(Some("1".to_string())), "TAM_1"),
        ];
        for (frame, expected) in cases {
            assert_eq!(frame.to_string(), expected);
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_new_rejects_eqw_pqw_rotating() {
        assert!(
            OrbitRelativeFrame::new(OrbitRelativeKind::EQW, OrbitRelativeVariant::Rotating)
                .is_err()
        );
        assert!(
            OrbitRelativeFrame::new(OrbitRelativeKind::PQW, OrbitRelativeVariant::Rotating)
                .is_err()
        );
        assert!(
            OrbitRelativeFrame::new(OrbitRelativeKind::EQW, OrbitRelativeVariant::Inertial).is_ok()
        );
        assert!(
            OrbitRelativeFrame::new(OrbitRelativeKind::PQW, OrbitRelativeVariant::Inertial).is_ok()
        );
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_new_accepts_valid_combos() {
        for kind in [
            OrbitRelativeKind::LVLH,
            OrbitRelativeKind::RTN,
            OrbitRelativeKind::NTW,
            OrbitRelativeKind::TNW,
            OrbitRelativeKind::SEZ,
            OrbitRelativeKind::VNC,
            OrbitRelativeKind::NSW,
        ] {
            assert!(OrbitRelativeFrame::new(kind, OrbitRelativeVariant::Rotating).is_ok());
            assert!(OrbitRelativeFrame::new(kind, OrbitRelativeVariant::Inertial).is_ok());
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_kind_variant_accessors() {
        let frame = OrbitRelativeFrame::new(OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating)
            .unwrap();
        assert_eq!(frame.kind(), OrbitRelativeKind::RTN);
        assert_eq!(frame.variant(), OrbitRelativeVariant::Rotating);
    }
}
