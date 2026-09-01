/*!
 * Unified `ReferenceFrame` type spanning celestial, orbit-relative, and body frames.
 *
 * [`ReferenceFrame`] is the top-level frame identity used throughout `brahe`. It
 * covers three kinds of frame:
 *
 * 1. **Celestial** ([`ReferenceFrame::Celestial`]): a frame the frames router can
 *    evaluate analytically from an epoch alone ([`CelestialFrame`]).
 * 2. **Orbit-relative** ([`ReferenceFrame::OrbitRelative`]): a local orbital frame
 *    (RTN, LVLH, ...) of a specific object, rotating with the orbit or
 *    frozen as an inertial snapshot.
 * 3. **Body** ([`ReferenceFrame::Body`]): an object-local frame (spacecraft body,
 *    sensor, actuator, ...) that has no global transformation.
 *
 * Orbit-relative and body frames carry an `object: Option<`[`ObjectId`]`>`.
 * `None` is a pure label, which is what a data file can express before
 * binding to a specific object; `Some` identifies the object the frame is
 * evaluable against.
 */

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::frames::CelestialFrame;
use crate::utils::errors::BraheError;

pub mod body;
pub mod object_id;
pub mod orbit_relative;

pub use body::*;
pub use object_id::*;
pub use orbit_relative::*;

/// Unified frame identity spanning celestial, orbit-relative, and body
/// frames.
///
/// `ReferenceFrame` is the top-level frame type used throughout `brahe`. See the
/// [module documentation](self) for the three frame kinds.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::{ReferenceFrame, CelestialFrame};
///
/// let rtn = ReferenceFrame::RTN("SC");
/// assert_eq!(rtn.to_string(), "RTN (rotating)@SC");
///
/// let gcrf: ReferenceFrame = CelestialFrame::GCRF.into();
/// assert_eq!(gcrf.to_string(), "GCRF");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReferenceFrame {
    /// Evaluable analytically from an epoch alone (existing frames router).
    Celestial(CelestialFrame),
    /// Local orbital frame of `object`. Evaluable when bound (`object` is
    /// `Some`) and the object is registered.
    OrbitRelative {
        /// Frame construction (axes definition).
        kind: OrbitRelativeFrameKind,
        /// Rotating (true local orbital frame) or inertial snapshot.
        variant: OrbitRelativeFrameVariant,
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

impl ReferenceFrame {
    /// Constructs a bound Radial/Transverse/Normal orbit-relative frame
    /// (rotating variant). SANA: RSW.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `RTN (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::RTN("SC").to_string(), "RTN (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn RTN(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::RTN,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Local-Vertical Local-Horizontal orbit-relative
    /// frame (rotating variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `LVLH (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::LVLH("SC").to_string(), "LVLH (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn LVLH(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::LVLH,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Normal/Tangential/cross-track orbit-relative
    /// frame (rotating variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `NTW (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::NTW("SC").to_string(), "NTW (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn NTW(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::NTW,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Tangential/Normal/cross-track orbit-relative
    /// frame (rotating variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `TNW (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::TNW("SC").to_string(), "TNW (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn TNW(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::TNW,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound topocentric South/East/Zenith orbit-relative
    /// frame (rotating variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `SEZ (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::SEZ("SC").to_string(), "SEZ (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SEZ(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::SEZ,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Velocity/Normal/Co-normal orbit-relative frame
    /// (rotating variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `VNC (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::VNC("SC").to_string(), "VNC (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn VNC(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::VNC,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Nadir/Sun/Normal orbit-relative frame (rotating
    /// variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `NSW (rotating)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::NSW("SC").to_string(), "NSW (rotating)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn NSW(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::NSW,
            OrbitRelativeFrameVariant::Rotating,
            object,
        )
    }

    /// Constructs a bound Perifocal orbit-relative frame (inertial-snapshot
    /// variant; `PQW` is SANA-registered only as inertial).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `PQW (inertial)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::PQW("SC").to_string(), "PQW (inertial)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn PQW(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::PQW,
            OrbitRelativeFrameVariant::Inertial,
            object,
        )
    }

    /// Constructs a bound Equinoctial orbit-relative frame
    /// (inertial-snapshot variant; `EQW` is SANA-registered only as
    /// inertial).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `EQW (inertial)` orbit-relative frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::EQW("SC").to_string(), "EQW (inertial)@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn EQW(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::orbit_relative_unchecked(
            OrbitRelativeFrameKind::EQW,
            OrbitRelativeFrameVariant::Inertial,
            object,
        )
    }

    /// Constructs a bound orbit-relative frame without kind/variant
    /// validation, for combinations known valid by construction (all
    /// `ReferenceFrame::<KIND>(object)` associated functions).
    fn orbit_relative_unchecked(
        kind: OrbitRelativeFrameKind,
        variant: OrbitRelativeFrameVariant,
        object: impl Into<ObjectId>,
    ) -> ReferenceFrame {
        ReferenceFrame::OrbitRelative {
            kind,
            variant,
            object: Some(object.into()),
        }
    }

    /// Constructs an orbit-relative frame, validating the kind/variant
    /// combination.
    ///
    /// General form of the `ReferenceFrame::<KIND>(object)` constructors, for
    /// callers that hold a runtime `kind`/`variant` pair (e.g. parsed from
    /// a CCSDS file) and an optional, not-yet-bound object.
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today; the others construct successfully but every transform through
    /// them errors until issue #452 adds the remaining derivations.
    ///
    /// # Arguments
    /// * `kind` - The frame construction (axes definition)
    /// * `variant` - Rotating (true local orbital frame) or inertial
    ///   snapshot
    /// * `object` - The bound object, or `None` for an unbound label
    ///
    /// # Returns
    /// * `Ok(ReferenceFrame)`: The `OrbitRelative` frame, if the combination is
    ///   valid
    /// * `Err(BraheError)`: If `kind` is `EQW` or `PQW` and `variant` is
    ///   `Rotating`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{ReferenceFrame, OrbitRelativeFrameKind, OrbitRelativeFrameVariant};
    ///
    /// let bound = ReferenceFrame::orbit_relative(OrbitRelativeFrameKind::RTN, OrbitRelativeFrameVariant::Inertial, Some("SC".into()));
    /// assert!(bound.is_ok());
    ///
    /// let invalid = ReferenceFrame::orbit_relative(OrbitRelativeFrameKind::EQW, OrbitRelativeFrameVariant::Rotating, None);
    /// assert!(invalid.is_err());
    /// ```
    pub fn orbit_relative(
        kind: OrbitRelativeFrameKind,
        variant: OrbitRelativeFrameVariant,
        object: Option<ObjectId>,
    ) -> Result<ReferenceFrame, BraheError> {
        if matches!(
            kind,
            OrbitRelativeFrameKind::EQW | OrbitRelativeFrameKind::PQW
        ) && variant == OrbitRelativeFrameVariant::Rotating
        {
            return Err(BraheError::Error(format!(
                "orbit-relative frame {} exists only as an inertial SANA frame and cannot be \
                 constructed with the rotating variant",
                kind
            )));
        }
        Ok(ReferenceFrame::OrbitRelative {
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
    /// `ReferenceFrame`: The bound `SC_BODY` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::SC_BODY("SC").to_string(), "SC_BODY@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SC_BODY(object: impl Into<ObjectId>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::SCBody(None))
    }

    /// Constructs a bound coarse sun sensor frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `CSS_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::CSS("SC", "1").to_string(), "CSS_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn CSS(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::CSS(Some(designator.into())))
    }

    /// Constructs a bound accelerometer frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `ACC_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::ACC("SC", "1").to_string(), "ACC_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn ACC(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::ACC(Some(designator.into())))
    }

    /// Constructs a bound autonomous star tracker frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `AST_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::AST("SC", "1").to_string(), "AST_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn AST(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::AST(Some(designator.into())))
    }

    /// Constructs a bound digital sun sensor frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `DSS_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::DSS("SC", "1").to_string(), "DSS_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn DSS(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::DSS(Some(designator.into())))
    }

    /// Constructs a bound Earth sensor assembly frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `ESA_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::ESA("SC", "1").to_string(), "ESA_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn ESA(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::ESA(Some(designator.into())))
    }

    /// Constructs a bound gyroscope frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `GYRO_FRAME_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::GYRO_FRAME("SC", "1").to_string(), "GYRO_FRAME_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn GYRO_FRAME(
        object: impl Into<ObjectId>,
        designator: impl Into<String>,
    ) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::GyroFrame(Some(designator.into())))
    }

    /// Constructs a bound inertial measurement unit frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `IMU_FRAME_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::IMU_FRAME("SC", "1").to_string(), "IMU_FRAME_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn IMU_FRAME(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::IMUFrame(Some(designator.into())))
    }

    /// Constructs a bound instrument frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The instrument instance designator (e.g. `"A"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `INSTRUMENT_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::INSTRUMENT("SC", "A").to_string(), "INSTRUMENT_A@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn INSTRUMENT(
        object: impl Into<ObjectId>,
        designator: impl Into<String>,
    ) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::Instrument(Some(designator.into())))
    }

    /// Constructs a bound magnetic torque assembly frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The actuator instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `MTA_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::MTA("SC", "1").to_string(), "MTA_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn MTA(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::MTA(Some(designator.into())))
    }

    /// Constructs a bound reaction wheel frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The actuator instance designator (e.g. `"4"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `RW_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::RW("SC", "4").to_string(), "RW_4@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn RW(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::RW(Some(designator.into())))
    }

    /// Constructs a bound solar array frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The array instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `SA_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::SA("SC", "1").to_string(), "SA_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SA(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::SA(Some(designator.into())))
    }

    /// Constructs a bound generic sensor frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"10"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `SENSOR_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::SENSOR("SC", "10").to_string(), "SENSOR_10@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn SENSOR(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::Sensor(Some(designator.into())))
    }

    /// Constructs a bound star tracker frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"2"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `STARTRACKER_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::STARTRACKER("SC", "2").to_string(), "STARTRACKER_2@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn STARTRACKER(
        object: impl Into<ObjectId>,
        designator: impl Into<String>,
    ) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::StarTracker(Some(designator.into())))
    }

    /// Constructs a bound three-axis magnetometer frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The sensor instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `TAM_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::TAM("SC", "1").to_string(), "TAM_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn TAM(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::TAM(Some(designator.into())))
    }

    /// Constructs a bound actuator frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `designator` - The actuator instance designator (e.g. `"1"`)
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound `ACTUATOR_<designator>` body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::ACTUATOR("SC", "1").to_string(), "ACTUATOR_1@SC");
    /// ```
    #[allow(non_snake_case)]
    pub fn ACTUATOR(object: impl Into<ObjectId>, designator: impl Into<String>) -> ReferenceFrame {
        ReferenceFrame::body(object, BodyFrame::Actuator(Some(designator.into())))
    }

    /// Constructs a bound body frame, general form.
    ///
    /// Covers designator-less and non-standard [`BodyFrame`] cases beyond
    /// the family-specific constructors (e.g. `ReferenceFrame::CSS`, `ReferenceFrame::RW`).
    /// Use [`From<BodyFrame>`](ReferenceFrame#impl-From<BodyFrame>-for-ReferenceFrame) to
    /// construct an unbound (`object: None`) body frame.
    ///
    /// # Arguments
    /// * `object` - The object the frame is defined relative to
    /// * `frame` - The body frame kind and optional instance designator
    ///
    /// # Returns
    /// `ReferenceFrame`: The bound body frame
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{ReferenceFrame, BodyFrame};
    ///
    /// let frame = ReferenceFrame::body("SC", BodyFrame::SCBody(None));
    /// assert_eq!(frame.to_string(), "SC_BODY@SC");
    /// ```
    pub fn body(object: impl Into<ObjectId>, frame: BodyFrame) -> ReferenceFrame {
        ReferenceFrame::Body {
            frame,
            object: Some(object.into()),
        }
    }

    /// Returns whether the frame carries the object identity resolution
    /// requires: a celestial frame (always), or an orbit-relative/body
    /// frame with a bound object. `true` is necessary but not sufficient
    /// for the frame to actually resolve. An orbit-relative frame also needs
    /// an axes derivation for its `kind` (currently only `RTN`), and a body
    /// frame also needs its orientation chain registered
    /// (`register_frame`).
    ///
    /// # Returns
    /// `bool`: `true` if the frame is bound (celestial frames are always
    /// bound)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{ReferenceFrame, BodyFrame};
    ///
    /// assert!(ReferenceFrame::SC_BODY("SC").is_bound());
    /// let unbound: ReferenceFrame = BodyFrame::SCBody(None).into();
    /// assert!(!unbound.is_bound());
    /// ```
    pub fn is_bound(&self) -> bool {
        match self {
            ReferenceFrame::Celestial(_) => true,
            ReferenceFrame::OrbitRelative { object, .. } => object.is_some(),
            ReferenceFrame::Body { object, .. } => object.is_some(),
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
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::RTN("SC").object().unwrap().to_string(), "SC");
    /// ```
    pub fn object(&self) -> Option<&ObjectId> {
        match self {
            ReferenceFrame::Celestial(_) => None,
            ReferenceFrame::OrbitRelative { object, .. } => object.as_ref(),
            ReferenceFrame::Body { object, .. } => object.as_ref(),
        }
    }
}

impl From<CelestialFrame> for ReferenceFrame {
    fn from(frame: CelestialFrame) -> Self {
        ReferenceFrame::Celestial(frame)
    }
}

impl From<OrbitRelativeFrame> for ReferenceFrame {
    fn from(frame: OrbitRelativeFrame) -> Self {
        ReferenceFrame::OrbitRelative {
            kind: frame.kind(),
            variant: frame.variant(),
            object: None,
        }
    }
}

impl From<BodyFrame> for ReferenceFrame {
    fn from(frame: BodyFrame) -> Self {
        ReferenceFrame::Body {
            frame,
            object: None,
        }
    }
}

impl fmt::Display for ReferenceFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceFrame::Celestial(frame) => write!(f, "{}", frame),
            ReferenceFrame::OrbitRelative {
                kind,
                variant,
                object,
            } => match object {
                Some(object) => write!(f, "{} ({})@{}", kind, variant, object),
                None => write!(f, "{} ({})", kind, variant),
            },
            ReferenceFrame::Body { frame, object } => match object {
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
        let f = ReferenceFrame::RTN("SC");
        assert_eq!(f.to_string(), "RTN (rotating)@SC");
        assert!(f.is_bound());
        assert_eq!(f.object().unwrap().to_string(), "SC");
        // EQW/PQW default to Inertial so construction never errors
        assert_eq!(ReferenceFrame::PQW("SC").to_string(), "PQW (inertial)@SC");
        assert_eq!(ReferenceFrame::CSS("SC", "1").to_string(), "CSS_1@SC");
        assert_eq!(ReferenceFrame::SC_BODY("SC").to_string(), "SC_BODY@SC");
        let unbound: ReferenceFrame = BodyFrame::SCBody(None).into();
        assert!(!unbound.is_bound());
        assert_eq!(unbound.to_string(), "SC_BODY");
        let cel: ReferenceFrame = CelestialFrame::GCRF.into();
        assert_eq!(cel.to_string(), "GCRF");
        assert!(cel.is_bound());
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_validation() {
        assert!(
            ReferenceFrame::orbit_relative(
                OrbitRelativeFrameKind::EQW,
                OrbitRelativeFrameVariant::Rotating,
                None
            )
            .is_err()
        );
        assert!(
            ReferenceFrame::orbit_relative(
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Inertial,
                Some("SC".into())
            )
            .is_ok()
        );
    }

    #[test]
    #[parallel]
    fn test_frame_serde_round_trip() {
        for f in [
            ReferenceFrame::RTN("SC"),
            ReferenceFrame::CSS("SC", "1"),
            CelestialFrame::ITRF.into(),
            ReferenceFrame::from(
                OrbitRelativeFrame::new(
                    OrbitRelativeFrameKind::LVLH,
                    OrbitRelativeFrameVariant::Rotating,
                )
                .unwrap(),
            ),
        ] {
            let s = serde_json::to_string(&f).unwrap();
            assert_eq!(serde_json::from_str::<ReferenceFrame>(&s).unwrap(), f);
        }
    }

    #[test]
    #[parallel]
    fn test_frame_object_accessor_all_kinds() {
        let cel: ReferenceFrame = CelestialFrame::GCRF.into();
        assert!(cel.object().is_none());
        assert_eq!(
            ReferenceFrame::RTN("SC").object().unwrap().to_string(),
            "SC"
        );
        assert_eq!(
            ReferenceFrame::SC_BODY("SC").object().unwrap().to_string(),
            "SC"
        );
        let unbound: ReferenceFrame = BodyFrame::SCBody(None).into();
        assert!(unbound.object().is_none());
    }

    #[test]
    #[parallel]
    fn test_frame_orbit_relative_convenience_constructors() {
        let cases = [
            (ReferenceFrame::NTW("SC"), "NTW (rotating)@SC"),
            (ReferenceFrame::TNW("SC"), "TNW (rotating)@SC"),
            (ReferenceFrame::SEZ("SC"), "SEZ (rotating)@SC"),
            (ReferenceFrame::VNC("SC"), "VNC (rotating)@SC"),
            (ReferenceFrame::NSW("SC"), "NSW (rotating)@SC"),
            (ReferenceFrame::EQW("SC"), "EQW (inertial)@SC"),
        ];
        for (frame, expected) in cases {
            assert_eq!(frame.to_string(), expected);
            assert!(frame.is_bound());
        }
    }

    #[test]
    #[parallel]
    fn test_frame_body_family_convenience_constructors() {
        let cases = [
            (ReferenceFrame::ACC("SC", "1"), "ACC_1@SC"),
            (ReferenceFrame::AST("SC", "1"), "AST_1@SC"),
            (ReferenceFrame::DSS("SC", "1"), "DSS_1@SC"),
            (ReferenceFrame::ESA("SC", "1"), "ESA_1@SC"),
            (ReferenceFrame::GYRO_FRAME("SC", "1"), "GYRO_FRAME_1@SC"),
            (ReferenceFrame::IMU_FRAME("SC", "2"), "IMU_FRAME_2@SC"),
            (ReferenceFrame::INSTRUMENT("SC", "A"), "INSTRUMENT_A@SC"),
            (ReferenceFrame::MTA("SC", "1"), "MTA_1@SC"),
            (ReferenceFrame::RW("SC", "4"), "RW_4@SC"),
            (ReferenceFrame::SA("SC", "1"), "SA_1@SC"),
            (ReferenceFrame::SENSOR("SC", "10"), "SENSOR_10@SC"),
            (ReferenceFrame::STARTRACKER("SC", "2"), "STARTRACKER_2@SC"),
            (ReferenceFrame::TAM("SC", "1"), "TAM_1@SC"),
            (ReferenceFrame::ACTUATOR("SC", "1"), "ACTUATOR_1@SC"),
        ];
        for (frame, expected) in cases {
            assert_eq!(frame.to_string(), expected);
            assert!(frame.is_bound());
        }
    }
}
