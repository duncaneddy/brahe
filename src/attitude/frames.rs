/*!
Attitude frame definitions for CCSDS ADM/APM/TDM support.

An attitude (quaternion, rotation matrix, ...) relates two frames. This module
defines the three frame endpoint types:

1. **Reference frames** ([`AttitudeFrame::Reference`]) — frames that the brahe
   frames router knows about. Compose directly with `rotation_frame_to_frame`.
2. **Orbit-relative frames** ([`AttitudeFrame::OrbitRelative`]) — local frames
   defined given an orbit state. Support rotating (true local orbital frame) or
   inertial-snapshot variants.
3. **Spacecraft frames** ([`AttitudeFrame::Spacecraft`]) — object-local frames
   (spacecraft body, sensor, actuator, ...) that the attitude data itself
   defines; they have no global transformation and are not composable.

The `OrbitRelativeFrame` composition story (rotations beyond RTN tracking the
current orbit state) is deferred to issue #452. Spacecraft frames are defined
purely by the attitude data and object-local convention.
*/

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::frames::ReferenceFrame;

/// One endpoint of an attitude transformation.
///
/// An attitude (quaternion, rotation matrix, ...) relates two frames. Frame
/// endpoints are one of three kinds: a frame the brahe frames router can
/// transform ([`AttitudeFrame::Reference`]), a local orbital frame defined
/// given an orbit state ([`AttitudeFrame::OrbitRelative`]), or an
/// object-local frame that the attitude data itself defines
/// ([`AttitudeFrame::Spacecraft`]).
///
/// # Examples
///
/// ```rust
/// use brahe::attitude::AttitudeFrame;
/// use brahe::frames::ReferenceFrame;
///
/// // Reference frame endpoint
/// let frame = AttitudeFrame::Reference(ReferenceFrame::GCRF);
/// assert_eq!(frame.to_string(), "GCRF");
///
/// // Display shows the frame name or designation
/// let ref_frame = AttitudeFrame::Reference(ReferenceFrame::EME2000);
/// println!("Frame: {}", ref_frame);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttitudeFrame {
    /// A frame known to the brahe frames router; composes directly with
    /// `rotation_frame_to_frame`.
    Reference(ReferenceFrame),
    /// A local orbital frame, defined given an orbit state.
    OrbitRelative(OrbitRelativeFrame),
    /// An object-local frame (spacecraft body, sensor, actuator, ...); has no
    /// global transformation — it is the frame attitude data defines.
    Spacecraft(SpacecraftFrame),
}

/// A local orbital frame: a kind plus rotating/inertial-snapshot variant.
///
/// Represents a frame that rotates with or tracks the orbit, composed of:
/// - A frame construction type (e.g., RTN, LVLH) defining the axes
/// - A variant indicating whether the frame rotates with the orbit or is frozen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrbitRelativeFrame {
    /// The frame construction (axes definition).
    pub kind: OrbitRelativeKind,
    /// Rotating (true local orbital frame) or inertial snapshot.
    pub variant: OrbitRelativeVariant,
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
    /// Perifocal.
    PQW,
    /// Equinoctial.
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

/// An object-local spacecraft frame with an optional instance designator.
///
/// Variants represent different spacecraft subsystems and sensors. The
/// optional `String` designator (e.g., `SCBody(Some("1"))`) is appended
/// to the frame name in Display output (e.g., `SC_BODY_1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpacecraftFrame {
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
            Self::Reference(frame) => write!(f, "{}", frame),
            Self::OrbitRelative(frame) => write!(f, "{}", frame),
            Self::Spacecraft(frame) => write!(f, "{}", frame),
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

impl fmt::Display for SpacecraftFrame {
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
            AttitudeFrame::Reference(ReferenceFrame::GCRF).to_string(),
            "GCRF"
        );
        assert_eq!(
            AttitudeFrame::OrbitRelative(OrbitRelativeFrame {
                kind: OrbitRelativeKind::RTN,
                variant: OrbitRelativeVariant::Rotating,
            })
            .to_string(),
            "RTN (rotating)"
        );
        assert_eq!(
            AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(Some("1".to_string()))).to_string(),
            "SC_BODY_1"
        );
    }

    #[test]
    #[parallel]
    fn test_attitude_frame_equality() {
        let a = AttitudeFrame::Reference(ReferenceFrame::EME2000);
        let b = AttitudeFrame::Reference(ReferenceFrame::EME2000);
        assert_eq!(a, b);
        assert_ne!(a, AttitudeFrame::Reference(ReferenceFrame::GCRF));
    }
}
