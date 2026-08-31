/*!
Attitude frame definitions for CCSDS ADM/APM/TDM support.

An attitude (quaternion, rotation matrix, ...) relates two frames. This module
defines the three frame endpoint types:

1. **Reference frames** ([`AttitudeFrame::ReferenceFrame`]) — frames that the
   brahe frames router knows about. Compose directly with `rotation_frame_to_frame`.
2. **Orbit-relative frames** ([`AttitudeFrame::OrbitRelative`]) — local frames
   defined given an orbit state. Support rotating (true local orbital frame) or
   inertial-snapshot variants.
3. **Spacecraft body frames** ([`AttitudeFrame::SpacecraftBody`]) — object-local
   frames (spacecraft body, sensor, actuator, ...) that the attitude data
   itself defines; they have no global transformation and are not composable.

The `OrbitRelativeFrame` composition story (rotations beyond RTN tracking the
current orbit state) is deferred to issue #452. Spacecraft body frames are
defined purely by the attitude data and object-local convention.
*/

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::frames::ReferenceFrame;
use crate::utils::errors::BraheError;

/// One endpoint of an attitude transformation.
///
/// An attitude (quaternion, rotation matrix, ...) relates two frames. Frame
/// endpoints are one of three kinds: a frame the brahe frames router can
/// transform ([`AttitudeFrame::ReferenceFrame`]), a local orbital frame
/// defined given an orbit state ([`AttitudeFrame::OrbitRelative`]), or an
/// object-local frame that the attitude data itself defines
/// ([`AttitudeFrame::SpacecraftBody`]).
///
/// # Examples
///
/// ```rust
/// use brahe::attitude::AttitudeFrame;
/// use brahe::frames::ReferenceFrame;
///
/// // Reference frame endpoint
/// let frame = AttitudeFrame::ReferenceFrame(ReferenceFrame::GCRF);
/// assert_eq!(frame.to_string(), "GCRF");
///
/// // Display shows the frame name or designation
/// let ref_frame = AttitudeFrame::ReferenceFrame(ReferenceFrame::EME2000);
/// println!("Frame: {}", ref_frame);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttitudeFrame {
    /// A frame known to the brahe frames router; composes directly with
    /// `rotation_frame_to_frame`.
    ReferenceFrame(ReferenceFrame),
    /// A local orbital frame, defined given an orbit state.
    OrbitRelative(OrbitRelativeFrame),
    /// An object-local frame (spacecraft body, sensor, actuator, ...); has no
    /// global transformation — it is the frame attitude data defines.
    SpacecraftBody(SpacecraftBodyFrame),
}

/// A local orbital frame: a kind plus rotating/inertial-snapshot variant.
///
/// Represents a frame that rotates with or tracks the orbit, composed of:
/// - A frame construction type (e.g., RTN, LVLH) defining the axes
/// - A variant indicating whether the frame rotates with the orbit or is frozen
///
/// Fields are private: per the SANA registry, `EQW` and `PQW` exist only as
/// inertial-snapshot frames, so construction goes through [`OrbitRelativeFrame::new`]
/// to reject that combination rather than allowing it and erroring later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrbitRelativeFrame {
    kind: OrbitRelativeKind,
    variant: OrbitRelativeVariant,
}

impl OrbitRelativeFrame {
    /// Constructs an orbit-relative frame, validating the kind/variant combination.
    ///
    /// Per the SANA orbit-relative frame registry, `EQW` and `PQW` are defined
    /// only as inertial-snapshot frames; they have no rotating variant.
    ///
    /// # Arguments
    /// * `kind` - The frame construction (axes definition)
    /// * `variant` - Rotating (true local orbital frame) or inertial snapshot
    ///
    /// # Returns
    /// * `Ok(OrbitRelativeFrame)` - If the combination is valid
    /// * `Err(BraheError)` - If `kind` is `EQW` or `PQW` and `variant` is `Rotating`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::attitude::{OrbitRelativeFrame, OrbitRelativeKind, OrbitRelativeVariant};
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
    /// `OrbitRelativeKind` - The frame construction
    pub fn kind(&self) -> OrbitRelativeKind {
        self.kind
    }

    /// Returns the rotating/inertial-snapshot variant.
    ///
    /// # Returns
    /// `OrbitRelativeVariant` - Rotating or inertial
    pub fn variant(&self) -> OrbitRelativeVariant {
        self.variant
    }
}

/// Local orbital frame axes definitions.
///
/// `RTN` is the frame the SANA registries call `RSW`; brahe uses its existing
/// RTN vocabulary (`state_eci_to_rtn`, `covariance_rtn`).
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

/// Rotating vs. quasi-inertial snapshot variant of a local orbital frame.
///
/// - **Rotating**: True local orbital frame, rotating with the orbit.
/// - **Inertial**: Quasi-inertial frame frozen at each evaluation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRelativeVariant {
    /// True local orbital frame, rotating with the orbit.
    Rotating,
    /// Quasi-inertial frame frozen at each evaluation time.
    Inertial,
}

/// An object-local spacecraft body frame with an optional instance designator.
///
/// Variants represent different spacecraft subsystems and sensors. The
/// optional `String` designator (e.g., `SCBody(Some("1"))`) is appended
/// to the frame name in Display output (e.g., `SC_BODY_1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpacecraftBodyFrame {
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

impl fmt::Display for AttitudeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReferenceFrame(frame) => write!(f, "{}", frame),
            Self::OrbitRelative(frame) => write!(f, "{}", frame),
            Self::SpacecraftBody(frame) => write!(f, "{}", frame),
        }
    }
}

impl fmt::Display for OrbitRelativeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.kind, self.variant)
    }
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

impl fmt::Display for OrbitRelativeVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rotating => write!(f, "rotating"),
            Self::Inertial => write!(f, "inertial"),
        }
    }
}

impl fmt::Display for SpacecraftBodyFrame {
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn test_attitude_frame_display() {
        assert_eq!(
            AttitudeFrame::ReferenceFrame(ReferenceFrame::GCRF).to_string(),
            "GCRF"
        );
        assert_eq!(
            AttitudeFrame::OrbitRelative(
                OrbitRelativeFrame::new(OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating)
                    .unwrap()
            )
            .to_string(),
            "RTN (rotating)"
        );
        assert_eq!(
            AttitudeFrame::SpacecraftBody(SpacecraftBodyFrame::SCBody(Some("1".to_string())))
                .to_string(),
            "SC_BODY_1"
        );
    }

    #[test]
    #[parallel]
    fn test_attitude_frame_equality() {
        let a = AttitudeFrame::ReferenceFrame(ReferenceFrame::EME2000);
        let b = AttitudeFrame::ReferenceFrame(ReferenceFrame::EME2000);
        assert_eq!(a, b);
        assert_ne!(a, AttitudeFrame::ReferenceFrame(ReferenceFrame::GCRF));
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
    fn test_spacecraft_body_frame_display_all_variants() {
        let cases = [
            (SpacecraftBodyFrame::ACC(Some("1".to_string())), "ACC_1"),
            (SpacecraftBodyFrame::Actuator(None), "ACTUATOR"),
            (SpacecraftBodyFrame::AST(Some("1".to_string())), "AST_1"),
            (SpacecraftBodyFrame::CSS(Some("2".to_string())), "CSS_2"),
            (SpacecraftBodyFrame::DSS(Some("1".to_string())), "DSS_1"),
            (SpacecraftBodyFrame::ESA(Some("1".to_string())), "ESA_1"),
            (
                SpacecraftBodyFrame::GyroFrame(Some("1".to_string())),
                "GYRO_FRAME_1",
            ),
            (
                SpacecraftBodyFrame::IMUFrame(Some("2".to_string())),
                "IMU_FRAME_2",
            ),
            (
                SpacecraftBodyFrame::Instrument(Some("A".to_string())),
                "INSTRUMENT_A",
            ),
            (SpacecraftBodyFrame::MTA(Some("1".to_string())), "MTA_1"),
            (SpacecraftBodyFrame::RW(Some("4".to_string())), "RW_4"),
            (SpacecraftBodyFrame::SA(Some("1".to_string())), "SA_1"),
            (SpacecraftBodyFrame::SCBody(None), "SC_BODY"),
            (
                SpacecraftBodyFrame::Sensor(Some("10".to_string())),
                "SENSOR_10",
            ),
            (
                SpacecraftBodyFrame::StarTracker(Some("2".to_string())),
                "STARTRACKER_2",
            ),
            (SpacecraftBodyFrame::TAM(Some("1".to_string())), "TAM_1"),
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
