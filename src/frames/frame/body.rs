/*!
 * Object-local spacecraft body, sensor, and actuator frames.
 */

use std::fmt;

use serde::{Deserialize, Serialize};

/// An object-local spacecraft body frame with an optional instance
/// designator.
///
/// The variants are the values of the SANA spacecraft body reference frame
/// registry (<https://sanaregistry.org/r/spacecraft_body_reference_frames/>),
/// covering spacecraft subsystems, sensors, and actuators. The optional
/// `String` designator (e.g., `SCBody(Some("1"))`) is appended to the frame
/// name in [`Display`](fmt::Display) output (e.g., `SC_BODY_1`).
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

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
}
