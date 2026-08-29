/*!
 * CCSDS Attitude Ephemeris Message (AEM) data structures.
 *
 * AEM messages contain time-series of attitude data (quaternions, Euler
 * angles, or spin parameters), optionally with derivatives or angular
 * velocity. A message has a header and one or more segments; each segment
 * carries its own metadata (object, frames, time span, attitude
 * representation) and a time-ordered sequence of attitude states. This
 * module provides the message types using brahe-native (radian, SI)
 * internal units; conversion to/from the wire units defined by the standard
 * happens in the KVN/XML/JSON read and write support.
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), §4
 */

use std::fmt;
use std::path::Path;

use nalgebra::{Vector3, Vector4};

use crate::attitude::attitude_types::{EulerAngle, EulerAngleOrder, Quaternion};
use crate::ccsds::common::{CCSDSFormat, CCSDSJsonKeyCase, CCSDSTimeSystem};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::frames::ADMReferenceFrame;
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Resolves a raw `ANGVEL_FRAME` token against a segment's `REF_FRAME_A` and
/// `REF_FRAME_B`.
///
/// CCSDS 504.0-B-2 Annex G-13 uses the literal tokens `REF_FRAME_A` and
/// `REF_FRAME_B` as `ANGVEL_FRAME` values meaning "same as `REF_FRAME_A`" and
/// "same as `REF_FRAME_B`" respectively, rather than the corresponding SANA
/// registry token. Matching is case-insensitive. Any other token is parsed
/// normally via [`ADMReferenceFrame::parse`], so a message that legitimately
/// names a frame that happens to be spelled `REF_FRAME_A`/`REF_FRAME_B` in a
/// non-Annex-G-13 sense is not expressible; the annex does not provide for
/// that case.
///
/// # Arguments
/// - `raw`: the raw `ANGVEL_FRAME` token as it appeared on the wire.
/// - `ref_frame_a`: the segment's `REF_FRAME_A`.
/// - `ref_frame_b`: the segment's `REF_FRAME_B`.
///
/// # Returns
/// ADMReferenceFrame: `ref_frame_a` or `ref_frame_b` when `raw` is the
/// corresponding literal alias, otherwise the ordinary parse of `raw`.
pub(crate) fn resolve_angvel_frame_token(
    raw: &str,
    ref_frame_a: &ADMReferenceFrame,
    ref_frame_b: &ADMReferenceFrame,
) -> ADMReferenceFrame {
    match raw.trim().to_uppercase().as_str() {
        "REF_FRAME_A" => ref_frame_a.clone(),
        "REF_FRAME_B" => ref_frame_b.clone(),
        _ => ADMReferenceFrame::parse(raw),
    }
}

/// CCSDS AEM message header (504.0-B-2 table 4-2).
#[derive(Debug, Clone)]
pub struct AEMHeader {
    /// Format version; 2.0 for messages conforming to 504.0-B-2.
    pub format_version: f64,
    /// Optional classification marking.
    pub classification: Option<String>,
    /// Message creation date (UTC).
    pub creation_date: Epoch,
    /// Creating agency or operator (SANA organizations registry abbreviation).
    pub originator: String,
    /// Optional message identifier, unique within the originator's context.
    pub message_id: Option<String>,
    /// Header comment lines.
    pub comments: Vec<String>,
}

impl AEMHeader {
    /// Creates a header with format version 2.0 and the current UTC time as
    /// the creation date.
    ///
    /// # Arguments
    /// - `originator`: creating agency or operator identifier.
    ///
    /// # Returns
    /// AEMHeader: A header with defaulted optional fields.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEMHeader;
    /// let header = AEMHeader::new("BRAHE");
    /// assert_eq!(header.format_version, 2.0);
    /// ```
    pub fn new(originator: &str) -> Self {
        Self {
            format_version: 2.0,
            classification: None,
            creation_date: Epoch::now(),
            originator: originator.to_string(),
            message_id: None,
            comments: Vec::new(),
        }
    }

    /// Sets the classification marking.
    ///
    /// # Arguments
    /// - `classification`: classification marking string (e.g. `"UNCLASSIFIED"`).
    ///
    /// # Returns
    /// AEMHeader: The header with the classification marking set.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEMHeader;
    /// let header = AEMHeader::new("BRAHE").with_classification("UNCLASSIFIED");
    /// assert_eq!(header.classification.as_deref(), Some("UNCLASSIFIED"));
    /// ```
    pub fn with_classification(mut self, classification: &str) -> Self {
        self.classification = Some(classification.to_string());
        self
    }

    /// Sets the message identifier.
    ///
    /// # Arguments
    /// - `message_id`: message identifier, unique within the originator's context.
    ///
    /// # Returns
    /// AEMHeader: The header with the message identifier set.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEMHeader;
    /// let header = AEMHeader::new("BRAHE").with_message_id("MSG-001");
    /// assert_eq!(header.message_id.as_deref(), Some("MSG-001"));
    /// ```
    pub fn with_message_id(mut self, message_id: &str) -> Self {
        self.message_id = Some(message_id.to_string());
        self
    }

    /// Sets the header comment lines.
    ///
    /// # Arguments
    /// - `comments`: comment lines to associate with the header.
    ///
    /// # Returns
    /// AEMHeader: The header with the comment lines set.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEMHeader;
    /// let header = AEMHeader::new("BRAHE").with_comments(vec!["generated by brahe".to_string()]);
    /// assert_eq!(header.comments.len(), 1);
    /// ```
    pub fn with_comments(mut self, comments: Vec<String>) -> Self {
        self.comments = comments;
        self
    }
}

/// AEM `ATTITUDE_TYPE` value (504.0-B-2 table 4-3), identifying both the
/// attitude representation and which derivative/rate quantities accompany it
/// in the data section (table 4-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AEMAttitudeType {
    /// Attitude quaternion only.
    Quaternion,
    /// Attitude quaternion plus its time derivative.
    QuaternionDerivative,
    /// Attitude quaternion plus angular velocity.
    QuaternionAngVel,
    /// Euler angles only.
    EulerAngle,
    /// Euler angles plus their time derivatives.
    EulerAngleDerivative,
    /// Euler angles plus angular velocity.
    EulerAngleAngVel,
    /// Spin axis and phase angle.
    Spin,
    /// Spin parameters plus nutation angle triple.
    SpinNutation,
    /// Spin parameters plus angular momentum vector triple.
    SpinNutationMom,
}

impl fmt::Display for AEMAttitudeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quaternion => write!(f, "QUATERNION"),
            Self::QuaternionDerivative => write!(f, "QUATERNION/DERIVATIVE"),
            Self::QuaternionAngVel => write!(f, "QUATERNION/ANGVEL"),
            Self::EulerAngle => write!(f, "EULER_ANGLE"),
            Self::EulerAngleDerivative => write!(f, "EULER_ANGLE/DERIVATIVE"),
            Self::EulerAngleAngVel => write!(f, "EULER_ANGLE/ANGVEL"),
            Self::Spin => write!(f, "SPIN"),
            Self::SpinNutation => write!(f, "SPIN/NUTATION"),
            Self::SpinNutationMom => write!(f, "SPIN/NUTATION_MOM"),
        }
    }
}

impl AEMAttitudeType {
    /// Parses an AEM `ATTITUDE_TYPE` token.
    ///
    /// The nine 504.0-B-2 (v2) tokens are matched exactly. The 504.0-B-1
    /// (v1) tokens `QUATERNION/RATE` and `EULER_ANGLE/RATE` are recognized
    /// specifically to produce an error naming them as v1 forms replaced by
    /// `QUATERNION/ANGVEL` and `EULER_ANGLE/ANGVEL` respectively; any other
    /// unrecognized token produces an error naming the offending value.
    ///
    /// # Arguments
    /// - `s`: the `ATTITUDE_TYPE` token to parse.
    ///
    /// # Returns
    /// Result<AEMAttitudeType, BraheError>: The parsed attitude type, or a
    /// `ParseError` naming the offending token.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEMAttitudeType;
    /// assert_eq!(
    ///     AEMAttitudeType::parse("QUATERNION/ANGVEL").unwrap(),
    ///     AEMAttitudeType::QuaternionAngVel
    /// );
    /// assert!(AEMAttitudeType::parse("QUATERNION/RATE").is_err());
    /// ```
    pub fn parse(s: &str) -> Result<Self, BraheError> {
        match s.trim() {
            "QUATERNION" => Ok(Self::Quaternion),
            "QUATERNION/DERIVATIVE" => Ok(Self::QuaternionDerivative),
            "QUATERNION/ANGVEL" => Ok(Self::QuaternionAngVel),
            "EULER_ANGLE" => Ok(Self::EulerAngle),
            "EULER_ANGLE/DERIVATIVE" => Ok(Self::EulerAngleDerivative),
            "EULER_ANGLE/ANGVEL" => Ok(Self::EulerAngleAngVel),
            "SPIN" => Ok(Self::Spin),
            "SPIN/NUTATION" => Ok(Self::SpinNutation),
            "SPIN/NUTATION_MOM" => Ok(Self::SpinNutationMom),
            "QUATERNION/RATE" => Err(ccsds_parse_error(
                "AEM",
                "invalid ATTITUDE_TYPE value 'QUATERNION/RATE'; this is a 504.0-B-1 (v1) form, \
                 replaced by 'QUATERNION/ANGVEL' in 504.0-B-2",
            )),
            "EULER_ANGLE/RATE" => Err(ccsds_parse_error(
                "AEM",
                "invalid ATTITUDE_TYPE value 'EULER_ANGLE/RATE'; this is a 504.0-B-1 (v1) form, \
                 replaced by 'EULER_ANGLE/ANGVEL' in 504.0-B-2",
            )),
            other => Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "invalid ATTITUDE_TYPE value '{}'; expected one of QUATERNION, \
                     QUATERNION/DERIVATIVE, QUATERNION/ANGVEL, EULER_ANGLE, \
                     EULER_ANGLE/DERIVATIVE, EULER_ANGLE/ANGVEL, SPIN, SPIN/NUTATION, \
                     SPIN/NUTATION_MOM",
                    other
                ),
            )),
        }
    }
}

/// AEM `INTERPOLATION_METHOD` value (504.0-B-2 table 4-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AEMInterpolationMethod {
    /// Linear interpolation.
    Linear,
    /// Hermite interpolation.
    Hermite,
    /// Lagrange interpolation.
    Lagrange,
}

impl fmt::Display for AEMInterpolationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "LINEAR"),
            Self::Hermite => write!(f, "HERMITE"),
            Self::Lagrange => write!(f, "LAGRANGE"),
        }
    }
}

impl AEMInterpolationMethod {
    /// Parses an AEM `INTERPOLATION_METHOD` token. Matching is
    /// case-insensitive per 504.0-B-2 §6.8.6 (all-upper or all-lower on the
    /// wire); [`fmt::Display`] always writes the all-upper form.
    ///
    /// # Arguments
    /// - `s`: the `INTERPOLATION_METHOD` token to parse.
    ///
    /// # Returns
    /// Result<AEMInterpolationMethod, BraheError>: The parsed method, or a
    /// `ParseError` naming the offending token.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEMInterpolationMethod;
    /// assert_eq!(
    ///     AEMInterpolationMethod::parse("linear").unwrap(),
    ///     AEMInterpolationMethod::Linear
    /// );
    /// ```
    pub fn parse(s: &str) -> Result<Self, BraheError> {
        match s.trim().to_uppercase().as_str() {
            "LINEAR" => Ok(Self::Linear),
            "HERMITE" => Ok(Self::Hermite),
            "LAGRANGE" => Ok(Self::Lagrange),
            other => Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "invalid INTERPOLATION_METHOD value '{}'; expected one of LINEAR, HERMITE, \
                     LAGRANGE (case-insensitive)",
                    other
                ),
            )),
        }
    }
}

/// Attitude data carried by a single AEM ephemeris line (504.0-B-2 table
/// 4-4). The variant determines the corresponding [`AEMAttitudeType`]; see
/// [`AEMAttitudeData::attitude_type`].
///
/// Unlike [`crate::ccsds::apm::APMSpin`], the spin variants here take no
/// `AngleFormat` conversion: construct them directly with radian values.
#[derive(Debug, Clone)]
pub enum AEMAttitudeData {
    /// Attitude quaternion from `REF_FRAME_A` to `REF_FRAME_B`.
    Quaternion {
        /// Attitude quaternion.
        quaternion: Quaternion,
    },
    /// Attitude quaternion plus its time derivative.
    QuaternionDerivative {
        /// Attitude quaternion.
        quaternion: Quaternion,
        /// Quaternion time derivative, scalar-first. Units: 1/s.
        derivative: Vector4<f64>,
    },
    /// Attitude quaternion plus angular velocity.
    QuaternionAngVel {
        /// Attitude quaternion.
        quaternion: Quaternion,
        /// Angular velocity vector, expressed in the segment's
        /// `ANGVEL_FRAME`. Units: rad/s.
        angular_velocity: Vector3<f64>,
    },
    /// Euler angles from `REF_FRAME_A` to `REF_FRAME_B`; the rotation
    /// sequence is carried by `angles.order` and must match the segment's
    /// `EULER_ROT_SEQ`.
    EulerAngle {
        /// Euler angles. Units: radians.
        angles: EulerAngle,
    },
    /// Euler angles plus their time derivatives.
    EulerAngleDerivative {
        /// Euler angles. Units: radians.
        angles: EulerAngle,
        /// Angle rates, in the same sequence order as `angles.order`.
        /// Units: rad/s.
        rates: Vector3<f64>,
    },
    /// Euler angles plus angular velocity.
    EulerAngleAngVel {
        /// Euler angles. Units: radians.
        angles: EulerAngle,
        /// Angular velocity vector, expressed in the segment's
        /// `ANGVEL_FRAME`. Units: rad/s.
        angular_velocity: Vector3<f64>,
    },
    /// Simple (non-nutating) spin.
    Spin {
        /// Right ascension of the spin axis in `REF_FRAME_A`. Units: radians.
        spin_alpha: f64,
        /// Declination of the spin axis in `REF_FRAME_A`. Units: radians.
        spin_delta: f64,
        /// Phase angle about the spin axis. Units: radians.
        spin_angle: f64,
        /// Angular velocity about the spin axis. Units: rad/s.
        spin_angle_vel: f64,
    },
    /// Spin with the `NUTATION` / `NUTATION_PER` / `NUTATION_PHASE` triple.
    SpinNutation {
        /// Right ascension of the spin axis in `REF_FRAME_A`. Units: radians.
        spin_alpha: f64,
        /// Declination of the spin axis in `REF_FRAME_A`. Units: radians.
        spin_delta: f64,
        /// Phase angle about the spin axis. Units: radians.
        spin_angle: f64,
        /// Angular velocity about the spin axis. Units: rad/s.
        spin_angle_vel: f64,
        /// Nutation angle. Units: radians.
        nutation: f64,
        /// Nutation period. Units: seconds.
        nutation_period: f64,
        /// Inertial nutation phase. Units: radians.
        nutation_phase: f64,
    },
    /// Spin with the `MOMENTUM_ALPHA` / `MOMENTUM_DELTA` / `NUTATION_VEL`
    /// triple.
    SpinNutationMom {
        /// Right ascension of the spin axis in `REF_FRAME_A`. Units: radians.
        spin_alpha: f64,
        /// Declination of the spin axis in `REF_FRAME_A`. Units: radians.
        spin_delta: f64,
        /// Phase angle about the spin axis. Units: radians.
        spin_angle: f64,
        /// Angular velocity about the spin axis. Units: rad/s.
        spin_angle_vel: f64,
        /// Right ascension of the angular momentum vector. Units: radians.
        momentum_alpha: f64,
        /// Declination of the angular momentum vector. Units: radians.
        momentum_delta: f64,
        /// Angular velocity of the spin axis around the momentum vector.
        /// Units: rad/s.
        nutation_vel: f64,
    },
}

impl AEMAttitudeData {
    /// Returns the [`AEMAttitudeType`] corresponding to this variant.
    ///
    /// # Returns
    /// AEMAttitudeType: The attitude type matching this data's shape.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::{AEMAttitudeData, AEMAttitudeType};
    /// use brahe::attitude::Quaternion;
    /// let data = AEMAttitudeData::Quaternion {
    ///     quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
    /// };
    /// assert_eq!(data.attitude_type(), AEMAttitudeType::Quaternion);
    /// ```
    pub fn attitude_type(&self) -> AEMAttitudeType {
        match self {
            Self::Quaternion { .. } => AEMAttitudeType::Quaternion,
            Self::QuaternionDerivative { .. } => AEMAttitudeType::QuaternionDerivative,
            Self::QuaternionAngVel { .. } => AEMAttitudeType::QuaternionAngVel,
            Self::EulerAngle { .. } => AEMAttitudeType::EulerAngle,
            Self::EulerAngleDerivative { .. } => AEMAttitudeType::EulerAngleDerivative,
            Self::EulerAngleAngVel { .. } => AEMAttitudeType::EulerAngleAngVel,
            Self::Spin { .. } => AEMAttitudeType::Spin,
            Self::SpinNutation { .. } => AEMAttitudeType::SpinNutation,
            Self::SpinNutationMom { .. } => AEMAttitudeType::SpinNutationMom,
        }
    }
}

/// A single AEM ephemeris line: a time tag plus the attitude data at that
/// epoch (504.0-B-2 §4.2.4.2).
#[derive(Debug, Clone)]
pub struct AEMAttitudeState {
    /// Epoch of this attitude state.
    pub epoch: Epoch,
    /// Attitude data at `epoch`.
    pub data: AEMAttitudeData,
}

/// AEM segment metadata (504.0-B-2 table 4-3).
#[derive(Debug, Clone)]
pub struct AEMMetadata {
    /// Spacecraft name.
    pub object_name: String,
    /// International designator, recommended form `YYYY-NNNP{PP}`.
    pub object_id: String,
    /// Optional celestial body the object is centered on (e.g. `EARTH`).
    pub center_name: Option<String>,
    /// Frame defining the transformation start point.
    pub ref_frame_a: ADMReferenceFrame,
    /// Frame defining the transformation end point.
    pub ref_frame_b: ADMReferenceFrame,
    /// Time system for all epochs in this segment.
    pub time_system: CCSDSTimeSystem,
    /// Start of the total time span covered by the data block.
    pub start_time: Epoch,
    /// End of the total time span covered by the data block.
    pub stop_time: Epoch,
    /// Optional start of the useable (interpolation-safe) span.
    pub useable_start_time: Option<Epoch>,
    /// Optional end of the useable (interpolation-safe) span.
    pub useable_stop_time: Option<Epoch>,
    /// Attitude representation and accompanying derivative/rate data.
    pub attitude_type: AEMAttitudeType,
    /// Euler rotation sequence; required when `attitude_type` is one of the
    /// Euler angle types, and not applicable otherwise.
    pub euler_rot_seq: Option<EulerAngleOrder>,
    /// Frame in which angular velocity components are expressed; required
    /// when `attitude_type` is one of the `/ANGVEL` types (and must equal
    /// `ref_frame_a` or `ref_frame_b`), and not applicable otherwise.
    pub angvel_frame: Option<ADMReferenceFrame>,
    /// Recommended interpolation method for this data block.
    pub interpolation_method: Option<AEMInterpolationMethod>,
    /// Interpolation polynomial degree; required when `interpolation_method`
    /// is present (a degree given without a method is permitted).
    pub interpolation_degree: Option<u32>,
    /// Metadata comment lines.
    pub comments: Vec<String>,
}

impl AEMMetadata {
    /// Creates metadata with the mandatory fields; all optional fields
    /// default to unset.
    ///
    /// # Arguments
    /// - `object_name`: spacecraft name.
    /// - `object_id`: international designator.
    /// - `ref_frame_a`: frame defining the transformation start point.
    /// - `ref_frame_b`: frame defining the transformation end point.
    /// - `time_system`: time system for the segment's epochs.
    /// - `start_time`: start of the total time span.
    /// - `stop_time`: end of the total time span.
    /// - `attitude_type`: attitude representation for this segment's data.
    ///
    /// # Returns
    /// AEMMetadata: Metadata with defaulted optional fields.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::{AEMAttitudeType, AEMMetadata};
    /// use brahe::ccsds::{ADMReferenceFrame, CCSDSTimeSystem};
    /// use brahe::time::Epoch;
    /// let metadata = AEMMetadata::new(
    ///     "SAT1", "2024-001A",
    ///     ADMReferenceFrame::parse("ICRF"),
    ///     ADMReferenceFrame::parse("SC_BODY_1"),
    ///     CCSDSTimeSystem::UTC,
    ///     Epoch::now(),
    ///     Epoch::now(),
    ///     AEMAttitudeType::Quaternion,
    /// );
    /// assert!(metadata.euler_rot_seq.is_none());
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        object_name: &str,
        object_id: &str,
        ref_frame_a: ADMReferenceFrame,
        ref_frame_b: ADMReferenceFrame,
        time_system: CCSDSTimeSystem,
        start_time: Epoch,
        stop_time: Epoch,
        attitude_type: AEMAttitudeType,
    ) -> Self {
        Self {
            object_name: object_name.to_string(),
            object_id: object_id.to_string(),
            center_name: None,
            ref_frame_a,
            ref_frame_b,
            time_system,
            start_time,
            stop_time,
            useable_start_time: None,
            useable_stop_time: None,
            attitude_type,
            euler_rot_seq: None,
            angvel_frame: None,
            interpolation_method: None,
            interpolation_degree: None,
            comments: Vec::new(),
        }
    }

    /// Sets the center name.
    pub fn with_center_name(mut self, center_name: &str) -> Self {
        self.center_name = Some(center_name.to_string());
        self
    }

    /// Sets the useable (interpolation-safe) time span.
    pub fn with_useable_times(
        mut self,
        useable_start_time: Epoch,
        useable_stop_time: Epoch,
    ) -> Self {
        self.useable_start_time = Some(useable_start_time);
        self.useable_stop_time = Some(useable_stop_time);
        self
    }

    /// Sets the Euler rotation sequence (only applicable to Euler angle
    /// attitude types; see [`AEMMetadata::validate`]).
    pub fn with_euler_rot_seq(mut self, euler_rot_seq: EulerAngleOrder) -> Self {
        self.euler_rot_seq = Some(euler_rot_seq);
        self
    }

    /// Sets the angular velocity frame (only applicable to `/ANGVEL`
    /// attitude types; see [`AEMMetadata::validate`]).
    pub fn with_angvel_frame(mut self, angvel_frame: ADMReferenceFrame) -> Self {
        self.angvel_frame = Some(angvel_frame);
        self
    }

    /// Sets the interpolation method and degree.
    ///
    /// # Arguments
    /// - `method`: recommended interpolation method.
    /// - `degree`: interpolation polynomial degree; required by
    ///   [`AEMMetadata::validate`] whenever `method` is set.
    pub fn with_interpolation(
        mut self,
        method: AEMInterpolationMethod,
        degree: Option<u32>,
    ) -> Self {
        self.interpolation_method = Some(method);
        self.interpolation_degree = degree;
        self
    }

    /// Validates the conditional metadata rules of CCSDS 504.0-B-2 table
    /// 4-3.
    ///
    /// Checks:
    /// - `start_time` does not fall after `stop_time`.
    /// - `EULER_ROT_SEQ` is present iff `attitude_type` is one of the Euler
    ///   angle types (`EulerAngle`, `EulerAngleDerivative`,
    ///   `EulerAngleAngVel`); it is an error both for it to be missing on a
    ///   Euler type and for it to be present on a non-Euler type.
    /// - `ANGVEL_FRAME` is present iff `attitude_type` is one of the
    ///   `/ANGVEL` types (`QuaternionAngVel`, `EulerAngleAngVel`), and when
    ///   present must equal `ref_frame_a` or `ref_frame_b`.
    /// - `interpolation_degree` is present whenever `interpolation_method`
    ///   is present (a degree given without a method is permitted).
    /// - `useable_start_time` and `useable_stop_time`, when present, fall
    ///   within `[start_time, stop_time]`, and `useable_start_time` does not
    ///   fall after `useable_stop_time`.
    ///
    /// # Returns
    /// Result<(), BraheError>: `Ok(())` if all conditional rules are
    /// satisfied, otherwise a `ParseError` describing the first violation
    /// found.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::{AEMAttitudeType, AEMMetadata};
    /// use brahe::ccsds::{ADMReferenceFrame, CCSDSTimeSystem};
    /// use brahe::time::Epoch;
    /// let metadata = AEMMetadata::new(
    ///     "SAT1", "2024-001A",
    ///     ADMReferenceFrame::parse("ICRF"),
    ///     ADMReferenceFrame::parse("SC_BODY_1"),
    ///     CCSDSTimeSystem::UTC,
    ///     Epoch::now(),
    ///     Epoch::now(),
    ///     AEMAttitudeType::Quaternion,
    /// );
    /// assert!(metadata.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), BraheError> {
        if self.start_time > self.stop_time {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "START_TIME {} must not fall after STOP_TIME {}",
                    self.start_time, self.stop_time
                ),
            ));
        }

        let is_euler_type = matches!(
            self.attitude_type,
            AEMAttitudeType::EulerAngle
                | AEMAttitudeType::EulerAngleDerivative
                | AEMAttitudeType::EulerAngleAngVel
        );
        match (is_euler_type, self.euler_rot_seq.is_some()) {
            (true, false) => {
                return Err(ccsds_parse_error(
                    "AEM",
                    &format!(
                        "EULER_ROT_SEQ is required when ATTITUDE_TYPE is '{}'",
                        self.attitude_type
                    ),
                ));
            }
            (false, true) => {
                return Err(ccsds_parse_error(
                    "AEM",
                    &format!(
                        "EULER_ROT_SEQ is only applicable to Euler angle ATTITUDE_TYPE values, \
                         not '{}'",
                        self.attitude_type
                    ),
                ));
            }
            _ => {}
        }

        let is_angvel_type = matches!(
            self.attitude_type,
            AEMAttitudeType::QuaternionAngVel | AEMAttitudeType::EulerAngleAngVel
        );
        match (is_angvel_type, &self.angvel_frame) {
            (true, None) => {
                return Err(ccsds_parse_error(
                    "AEM",
                    &format!(
                        "ANGVEL_FRAME is required when ATTITUDE_TYPE is '{}'",
                        self.attitude_type
                    ),
                ));
            }
            (false, Some(_)) => {
                return Err(ccsds_parse_error(
                    "AEM",
                    &format!(
                        "ANGVEL_FRAME is only applicable to '/ANGVEL' ATTITUDE_TYPE values, not \
                         '{}'",
                        self.attitude_type
                    ),
                ));
            }
            (true, Some(angvel_frame)) => {
                if angvel_frame != &self.ref_frame_a && angvel_frame != &self.ref_frame_b {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!(
                            "ANGVEL_FRAME '{}' must equal REF_FRAME_A '{}' or REF_FRAME_B '{}'",
                            angvel_frame, self.ref_frame_a, self.ref_frame_b
                        ),
                    ));
                }
            }
            (false, None) => {}
        }

        if self.interpolation_method.is_some() && self.interpolation_degree.is_none() {
            return Err(ccsds_parse_error(
                "AEM",
                "INTERPOLATION_DEGREE is required when INTERPOLATION_METHOD is present",
            ));
        }

        if let Some(useable_start_time) = self.useable_start_time
            && (useable_start_time < self.start_time || useable_start_time > self.stop_time)
        {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "USEABLE_START_TIME {} must fall within [START_TIME {}, STOP_TIME {}]",
                    useable_start_time, self.start_time, self.stop_time
                ),
            ));
        }
        if let Some(useable_stop_time) = self.useable_stop_time
            && (useable_stop_time < self.start_time || useable_stop_time > self.stop_time)
        {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "USEABLE_STOP_TIME {} must fall within [START_TIME {}, STOP_TIME {}]",
                    useable_stop_time, self.start_time, self.stop_time
                ),
            ));
        }
        if let (Some(useable_start_time), Some(useable_stop_time)) =
            (self.useable_start_time, self.useable_stop_time)
            && useable_start_time > useable_stop_time
        {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "USEABLE_START_TIME {} must not fall after USEABLE_STOP_TIME {}",
                    useable_start_time, useable_stop_time
                ),
            ));
        }

        Ok(())
    }
}

/// A single segment within an AEM message: metadata plus a time-ordered
/// sequence of attitude states (504.0-B-2 §4.2.1).
#[derive(Debug, Clone)]
pub struct AEMSegment {
    /// Segment metadata.
    pub metadata: AEMMetadata,
    /// Comments associated with the data block (before the first state).
    pub comments: Vec<String>,
    /// Time-ordered attitude states.
    pub states: Vec<AEMAttitudeState>,
}

impl AEMSegment {
    /// Creates a new empty segment with the given metadata.
    ///
    /// # Arguments
    /// - `metadata`: segment metadata.
    ///
    /// # Returns
    /// AEMSegment: A segment with no attitude states.
    pub fn new(metadata: AEMMetadata) -> Self {
        Self {
            metadata,
            comments: Vec::new(),
            states: Vec::new(),
        }
    }

    /// Appends an attitude state to this segment.
    ///
    /// # Arguments
    /// - `state`: attitude state to append.
    ///
    /// # Returns
    /// Result<(), BraheError>: `Ok(())` on success, or a `ParseError` if
    /// `state.data`'s attitude type does not match `metadata.attitude_type`,
    /// if `state.data` is one of the Euler angle variants whose
    /// `angles.order` does not match `metadata.euler_rot_seq` (when the
    /// latter is set), or if `state.epoch` is not strictly after the last
    /// state's epoch (504.0-B-2 §4.2.4.8.1: increasing time, no repeated
    /// time tags).
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::{
    ///     AEMAttitudeData, AEMAttitudeState, AEMAttitudeType, AEMMetadata, AEMSegment,
    /// };
    /// use brahe::ccsds::{ADMReferenceFrame, CCSDSTimeSystem};
    /// use brahe::attitude::Quaternion;
    /// use brahe::time::{Epoch, TimeSystem};
    /// let metadata = AEMMetadata::new(
    ///     "SAT1", "2024-001A",
    ///     ADMReferenceFrame::parse("ICRF"),
    ///     ADMReferenceFrame::parse("SC_BODY_1"),
    ///     CCSDSTimeSystem::UTC,
    ///     Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC),
    ///     Epoch::from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, TimeSystem::UTC),
    ///     AEMAttitudeType::Quaternion,
    /// );
    /// let mut segment = AEMSegment::new(metadata);
    /// let state = AEMAttitudeState {
    ///     epoch: Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC),
    ///     data: AEMAttitudeData::Quaternion {
    ///         quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
    ///     },
    /// };
    /// assert!(segment.push_state(state).is_ok());
    /// assert_eq!(segment.states.len(), 1);
    /// ```
    pub fn push_state(&mut self, state: AEMAttitudeState) -> Result<(), BraheError> {
        let state_type = state.data.attitude_type();
        if state_type != self.metadata.attitude_type {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "attitude state type '{}' does not match segment ATTITUDE_TYPE '{}'",
                    state_type, self.metadata.attitude_type
                ),
            ));
        }
        if let Some(angles) = match &state.data {
            AEMAttitudeData::EulerAngle { angles }
            | AEMAttitudeData::EulerAngleDerivative { angles, .. }
            | AEMAttitudeData::EulerAngleAngVel { angles, .. } => Some(angles),
            _ => None,
        } && let Some(expected_order) = self.metadata.euler_rot_seq
            && angles.order != expected_order
        {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "attitude state Euler angle order '{:?}' does not match segment \
                     EULER_ROT_SEQ '{:?}'",
                    angles.order, expected_order
                ),
            ));
        }
        if let Some(last_state) = self.states.last()
            && state.epoch <= last_state.epoch
        {
            return Err(ccsds_parse_error(
                "AEM",
                &format!(
                    "epoch {} is not strictly increasing after previous epoch {}",
                    state.epoch, last_state.epoch
                ),
            ));
        }
        self.states.push(state);
        Ok(())
    }
}

/// A complete CCSDS Attitude Ephemeris Message.
#[derive(Debug, Clone)]
pub struct AEM {
    /// Message header.
    pub header: AEMHeader,
    /// One or more attitude ephemeris segments.
    pub segments: Vec<AEMSegment>,
}

impl AEM {
    /// Creates a new AEM message with no segments.
    ///
    /// # Arguments
    /// - `originator`: creating agency or operator identifier.
    ///
    /// # Returns
    /// AEM: A message with an empty segment list.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEM;
    /// let aem = AEM::new("BRAHE");
    /// assert!(aem.segments.is_empty());
    /// ```
    pub fn new(originator: &str) -> Self {
        Self {
            header: AEMHeader::new(originator),
            segments: Vec::new(),
        }
    }

    /// Appends a segment to the AEM.
    ///
    /// # Arguments
    /// - `segment`: segment to append.
    pub fn push_segment(&mut self, segment: AEMSegment) {
        self.segments.push(segment);
    }

    /// Validates the message for writing (504.0-B-2 §4.2).
    ///
    /// Every writer ([`crate::ccsds::kvn::write_aem`],
    /// [`crate::ccsds::xml::write_aem_xml`],
    /// [`crate::ccsds::json::write_aem_json`]) calls this before
    /// serializing. Checks:
    /// - at least one segment is present
    /// - every segment has at least one attitude state
    /// - every segment's metadata passes [`AEMMetadata::validate`]
    /// - every segment's `TIME_SYSTEM` can represent epochs (its
    ///   [`CCSDSTimeSystem::to_time_system`] is not `None`); `SCLK`, `MET`,
    ///   `MRT`, `GMST`, and `TDR` have no fixed relationship to a physical
    ///   time system brahe can convert epochs into or out of, so a message
    ///   using one of them could not be read back by brahe's own parsers
    /// - every state in every segment matches the segment's
    ///   `metadata.attitude_type` and carries a strictly increasing epoch
    ///   (the same invariants [`AEMSegment::push_state`] enforces, checked
    ///   again here because `states` is a public field and can be mutated
    ///   directly, bypassing `push_state`)
    /// - for adjacent segments that both carry useable times, the later
    ///   segment's `USEABLE_START_TIME` is not before the earlier segment's
    ///   `USEABLE_STOP_TIME` (504.0-B-2 table 4-3); segments where either
    ///   side omits useable times are not compared
    ///
    /// # Returns
    /// Result<(), BraheError>: Ok if the message is well-formed for
    /// writing, or an error describing the first violation found.
    ///
    /// # Examples
    /// ```
    /// use brahe::ccsds::aem::AEM;
    ///
    /// let aem = AEM::new("BRAHE");
    /// assert!(aem.validate_for_write().is_err());
    /// ```
    pub fn validate_for_write(&self) -> Result<(), BraheError> {
        if self.segments.is_empty() {
            return Err(ccsds_missing_field("AEM", "at least one segment"));
        }
        for (idx, segment) in self.segments.iter().enumerate() {
            if segment.states.is_empty() {
                return Err(ccsds_missing_field(
                    "AEM",
                    &format!(
                        "at least one attitude state in segment '{}'",
                        segment.metadata.object_name
                    ),
                ));
            }
            segment.metadata.validate()?;

            if segment.metadata.time_system.to_time_system().is_none() {
                return Err(ccsds_parse_error(
                    "AEM",
                    &format!(
                        "TIME_SYSTEM '{}' cannot be written: brahe has no native \
                         representation for its epochs (SCLK, MET, MRT, GMST, and TDR are \
                         spacecraft- or mission-specific clocks with no fixed relationship to \
                         brahe's physical time systems), so a message using it could not be \
                         read back by brahe's own parsers",
                        segment.metadata.time_system
                    ),
                ));
            }

            let mut previous_epoch: Option<Epoch> = None;
            for state in &segment.states {
                let state_type = state.data.attitude_type();
                if state_type != segment.metadata.attitude_type {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!(
                            "attitude state type '{}' does not match segment ATTITUDE_TYPE '{}'",
                            state_type, segment.metadata.attitude_type
                        ),
                    ));
                }
                if let Some(previous) = previous_epoch
                    && state.epoch <= previous
                {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!(
                            "epoch {} is not strictly increasing after previous epoch {}",
                            state.epoch, previous
                        ),
                    ));
                }
                previous_epoch = Some(state.epoch);
            }

            if idx > 0 {
                let previous_segment = &self.segments[idx - 1];
                if let (Some(previous_useable_stop), Some(current_useable_start)) = (
                    previous_segment.metadata.useable_stop_time,
                    segment.metadata.useable_start_time,
                ) && current_useable_start < previous_useable_stop
                {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!(
                            "segment {} USEABLE_START_TIME {} must not fall before segment {} \
                             USEABLE_STOP_TIME {}",
                            idx,
                            current_useable_start,
                            idx - 1,
                            previous_useable_stop
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Parses an AEM message from a string, auto-detecting the format.
    ///
    /// # Arguments
    /// - `content`: string content of the AEM message (KVN, XML, or JSON).
    ///
    /// # Returns
    /// Result<AEM, BraheError>: The parsed message, or an error if the
    /// content is malformed.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let format = crate::ccsds::common::detect_format(content);
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::parse_aem(content),
            CCSDSFormat::XML => crate::ccsds::xml::parse_aem_xml(content),
            CCSDSFormat::JSON => crate::ccsds::json::parse_aem_json(content),
        }
    }

    /// Parses an AEM message from a file, auto-detecting the format.
    ///
    /// # Arguments
    /// - `path`: path to the AEM file.
    ///
    /// # Returns
    /// Result<AEM, BraheError>: The parsed message, or an error if the file
    /// cannot be read or its content is malformed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| BraheError::IoError(format!("Failed to read AEM file: {}", e)))?;
        Self::from_str(&content)
    }

    /// Writes the AEM message to a string in the specified format.
    ///
    /// # Arguments
    /// - `format`: output encoding format (KVN, XML, or JSON).
    ///
    /// # Returns
    /// Result<String, BraheError>: The serialized message, or an error if
    /// serialization fails.
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::write_aem(self),
            CCSDSFormat::XML => crate::ccsds::xml::write_aem_xml(self),
            CCSDSFormat::JSON => crate::ccsds::json::write_aem_json(self, CCSDSJsonKeyCase::Lower),
        }
    }

    /// Writes the AEM message to JSON with explicit key case control.
    ///
    /// # Arguments
    /// - `key_case`: whether CCSDS keywords should be lowercase or
    ///   uppercase.
    ///
    /// # Returns
    /// Result<String, BraheError>: The serialized JSON string, or an error if
    /// serialization fails.
    pub fn to_json_string(&self, key_case: CCSDSJsonKeyCase) -> Result<String, BraheError> {
        crate::ccsds::json::write_aem_json(self, key_case)
    }

    /// Writes the AEM message to a file in the specified format.
    ///
    /// # Arguments
    /// - `path`: output file path.
    /// - `format`: output encoding format (KVN, XML, or JSON).
    ///
    /// # Returns
    /// Result<(), BraheError>: Success, or an error if serialization or the
    /// file write fails.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write AEM file: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_aem_header_new_defaults() {
        let header = AEMHeader::new("BRAHE");
        assert_eq!(header.format_version, 2.0);
        assert_eq!(header.originator, "BRAHE");
        assert!(header.classification.is_none());
        assert!(header.message_id.is_none());
        assert!(header.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_aem_header_builders() {
        let header = AEMHeader::new("BRAHE")
            .with_classification("UNCLASSIFIED")
            .with_message_id("MSG-001")
            .with_comments(vec!["generated by brahe".to_string()]);
        assert_eq!(header.classification.as_deref(), Some("UNCLASSIFIED"));
        assert_eq!(header.message_id.as_deref(), Some("MSG-001"));
        assert_eq!(header.comments.len(), 1);
    }

    // ------------------------------------------------------------------
    // AEMAttitudeType
    // ------------------------------------------------------------------

    #[test]
    #[parallel]
    fn test_aem_attitude_type_display_and_parse_round_trip() {
        let cases = [
            (AEMAttitudeType::Quaternion, "QUATERNION"),
            (
                AEMAttitudeType::QuaternionDerivative,
                "QUATERNION/DERIVATIVE",
            ),
            (AEMAttitudeType::QuaternionAngVel, "QUATERNION/ANGVEL"),
            (AEMAttitudeType::EulerAngle, "EULER_ANGLE"),
            (
                AEMAttitudeType::EulerAngleDerivative,
                "EULER_ANGLE/DERIVATIVE",
            ),
            (AEMAttitudeType::EulerAngleAngVel, "EULER_ANGLE/ANGVEL"),
            (AEMAttitudeType::Spin, "SPIN"),
            (AEMAttitudeType::SpinNutation, "SPIN/NUTATION"),
            (AEMAttitudeType::SpinNutationMom, "SPIN/NUTATION_MOM"),
        ];
        for (variant, token) in cases {
            assert_eq!(variant.to_string(), token);
            assert_eq!(AEMAttitudeType::parse(token).unwrap(), variant);
        }
    }

    #[test]
    #[parallel]
    fn test_aem_attitude_type_parse_v1_quaternion_rate_error_names_v1() {
        let err = AEMAttitudeType::parse("QUATERNION/RATE").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("QUATERNION/RATE"));
        assert!(msg.contains("504.0-B-1"));
        assert!(msg.contains("QUATERNION/ANGVEL"));
    }

    #[test]
    #[parallel]
    fn test_aem_attitude_type_parse_v1_euler_angle_rate_error_names_v1() {
        let err = AEMAttitudeType::parse("EULER_ANGLE/RATE").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("EULER_ANGLE/RATE"));
        assert!(msg.contains("504.0-B-1"));
        assert!(msg.contains("EULER_ANGLE/ANGVEL"));
    }

    #[test]
    #[parallel]
    fn test_aem_attitude_type_parse_unknown_token_names_token() {
        let err = AEMAttitudeType::parse("BOGUS_TYPE").unwrap_err();
        assert!(err.to_string().contains("BOGUS_TYPE"));
    }

    // ------------------------------------------------------------------
    // AEMInterpolationMethod
    // ------------------------------------------------------------------

    #[test]
    #[parallel]
    fn test_aem_interpolation_method_display_upper() {
        assert_eq!(AEMInterpolationMethod::Linear.to_string(), "LINEAR");
        assert_eq!(AEMInterpolationMethod::Hermite.to_string(), "HERMITE");
        assert_eq!(AEMInterpolationMethod::Lagrange.to_string(), "LAGRANGE");
    }

    #[test]
    #[parallel]
    fn test_aem_interpolation_method_parse_case_insensitive() {
        assert_eq!(
            AEMInterpolationMethod::parse("linear").unwrap(),
            AEMInterpolationMethod::Linear
        );
        assert_eq!(
            AEMInterpolationMethod::parse("Hermite").unwrap(),
            AEMInterpolationMethod::Hermite
        );
        assert_eq!(
            AEMInterpolationMethod::parse("LAGRANGE").unwrap(),
            AEMInterpolationMethod::Lagrange
        );
    }

    #[test]
    #[parallel]
    fn test_aem_interpolation_method_parse_invalid_names_token() {
        let err = AEMInterpolationMethod::parse("CUBIC_SPLINE").unwrap_err();
        assert!(err.to_string().contains("CUBIC_SPLINE"));
    }

    // ------------------------------------------------------------------
    // AEMAttitudeData
    // ------------------------------------------------------------------

    fn unit_quaternion() -> Quaternion {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }

    fn zero_euler_angle() -> EulerAngle {
        EulerAngle::new(
            EulerAngleOrder::ZXZ,
            0.0,
            0.0,
            0.0,
            crate::constants::AngleFormat::Radians,
        )
    }

    #[test]
    #[parallel]
    fn test_aem_attitude_data_attitude_type_all_variants() {
        assert_eq!(
            AEMAttitudeData::Quaternion {
                quaternion: unit_quaternion()
            }
            .attitude_type(),
            AEMAttitudeType::Quaternion
        );
        assert_eq!(
            AEMAttitudeData::QuaternionDerivative {
                quaternion: unit_quaternion(),
                derivative: Vector4::new(0.0, 0.0, 0.0, 0.0),
            }
            .attitude_type(),
            AEMAttitudeType::QuaternionDerivative
        );
        assert_eq!(
            AEMAttitudeData::QuaternionAngVel {
                quaternion: unit_quaternion(),
                angular_velocity: Vector3::new(0.0, 0.0, 0.0),
            }
            .attitude_type(),
            AEMAttitudeType::QuaternionAngVel
        );
        assert_eq!(
            AEMAttitudeData::EulerAngle {
                angles: zero_euler_angle()
            }
            .attitude_type(),
            AEMAttitudeType::EulerAngle
        );
        assert_eq!(
            AEMAttitudeData::EulerAngleDerivative {
                angles: zero_euler_angle(),
                rates: Vector3::new(0.0, 0.0, 0.0),
            }
            .attitude_type(),
            AEMAttitudeType::EulerAngleDerivative
        );
        assert_eq!(
            AEMAttitudeData::EulerAngleAngVel {
                angles: zero_euler_angle(),
                angular_velocity: Vector3::new(0.0, 0.0, 0.0),
            }
            .attitude_type(),
            AEMAttitudeType::EulerAngleAngVel
        );
        assert_eq!(
            AEMAttitudeData::Spin {
                spin_alpha: 0.0,
                spin_delta: 0.0,
                spin_angle: 0.0,
                spin_angle_vel: 0.0,
            }
            .attitude_type(),
            AEMAttitudeType::Spin
        );
        assert_eq!(
            AEMAttitudeData::SpinNutation {
                spin_alpha: 0.0,
                spin_delta: 0.0,
                spin_angle: 0.0,
                spin_angle_vel: 0.0,
                nutation: 0.0,
                nutation_period: 0.0,
                nutation_phase: 0.0,
            }
            .attitude_type(),
            AEMAttitudeType::SpinNutation
        );
        assert_eq!(
            AEMAttitudeData::SpinNutationMom {
                spin_alpha: 0.0,
                spin_delta: 0.0,
                spin_angle: 0.0,
                spin_angle_vel: 0.0,
                momentum_alpha: 0.0,
                momentum_delta: 0.0,
                nutation_vel: 0.0,
            }
            .attitude_type(),
            AEMAttitudeType::SpinNutationMom
        );
    }

    // ------------------------------------------------------------------
    // AEMMetadata
    // ------------------------------------------------------------------

    fn icrf() -> ADMReferenceFrame {
        ADMReferenceFrame::parse("ICRF")
    }

    fn sc_body_1() -> ADMReferenceFrame {
        ADMReferenceFrame::parse("SC_BODY_1")
    }

    fn t0() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    fn t1() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    fn base_metadata(attitude_type: AEMAttitudeType) -> AEMMetadata {
        AEMMetadata::new(
            "SAT1",
            "2024-001A",
            icrf(),
            sc_body_1(),
            CCSDSTimeSystem::UTC,
            t0(),
            t1(),
            attitude_type,
        )
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_new_defaults() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion);
        assert_eq!(metadata.object_name, "SAT1");
        assert_eq!(metadata.object_id, "2024-001A");
        assert!(metadata.center_name.is_none());
        assert_eq!(metadata.ref_frame_a, icrf());
        assert_eq!(metadata.ref_frame_b, sc_body_1());
        assert_eq!(metadata.time_system, CCSDSTimeSystem::UTC);
        assert!(metadata.useable_start_time.is_none());
        assert!(metadata.useable_stop_time.is_none());
        assert_eq!(metadata.attitude_type, AEMAttitudeType::Quaternion);
        assert!(metadata.euler_rot_seq.is_none());
        assert!(metadata.angvel_frame.is_none());
        assert!(metadata.interpolation_method.is_none());
        assert!(metadata.interpolation_degree.is_none());
        assert!(metadata.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_builders() {
        let metadata = base_metadata(AEMAttitudeType::EulerAngleAngVel)
            .with_center_name("EARTH")
            .with_useable_times(t0(), t1())
            .with_euler_rot_seq(EulerAngleOrder::ZXZ)
            .with_angvel_frame(sc_body_1())
            .with_interpolation(AEMInterpolationMethod::Hermite, Some(5));
        assert_eq!(metadata.center_name.as_deref(), Some("EARTH"));
        assert_eq!(metadata.useable_start_time, Some(t0()));
        assert_eq!(metadata.useable_stop_time, Some(t1()));
        assert_eq!(metadata.euler_rot_seq, Some(EulerAngleOrder::ZXZ));
        assert_eq!(metadata.angvel_frame, Some(sc_body_1()));
        assert_eq!(
            metadata.interpolation_method,
            Some(AEMInterpolationMethod::Hermite)
        );
        assert_eq!(metadata.interpolation_degree, Some(5));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_quaternion_ok() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion);
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_euler_missing_rot_seq_errors() {
        let metadata = base_metadata(AEMAttitudeType::EulerAngle);
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("EULER_ROT_SEQ"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_euler_with_rot_seq_ok() {
        let metadata =
            base_metadata(AEMAttitudeType::EulerAngle).with_euler_rot_seq(EulerAngleOrder::ZXZ);
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_non_euler_with_rot_seq_errors() {
        let metadata =
            base_metadata(AEMAttitudeType::Quaternion).with_euler_rot_seq(EulerAngleOrder::ZXZ);
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("EULER_ROT_SEQ"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_angvel_missing_frame_errors() {
        let metadata = base_metadata(AEMAttitudeType::QuaternionAngVel);
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("ANGVEL_FRAME"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_angvel_frame_matches_ref_frame_a_ok() {
        let metadata = base_metadata(AEMAttitudeType::QuaternionAngVel).with_angvel_frame(icrf());
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_angvel_frame_matches_ref_frame_b_ok() {
        let metadata =
            base_metadata(AEMAttitudeType::QuaternionAngVel).with_angvel_frame(sc_body_1());
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_angvel_frame_mismatch_errors() {
        let metadata = base_metadata(AEMAttitudeType::QuaternionAngVel)
            .with_angvel_frame(ADMReferenceFrame::parse("INSTRUMENT_A"));
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("must equal REF_FRAME_A"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_non_angvel_with_frame_errors() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion).with_angvel_frame(icrf());
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("ANGVEL_FRAME"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_interpolation_method_without_degree_errors() {
        let mut metadata = base_metadata(AEMAttitudeType::Quaternion);
        metadata.interpolation_method = Some(AEMInterpolationMethod::Linear);
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("INTERPOLATION_DEGREE"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_interpolation_degree_without_method_ok() {
        let mut metadata = base_metadata(AEMAttitudeType::Quaternion);
        metadata.interpolation_degree = Some(3);
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_interpolation_both_set_ok() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion)
            .with_interpolation(AEMInterpolationMethod::Linear, Some(1));
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_useable_start_before_start_time_errors() {
        let mut metadata = base_metadata(AEMAttitudeType::Quaternion);
        metadata.useable_start_time = Some(t0() - 10.0);
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("USEABLE_START_TIME"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_useable_stop_after_stop_time_errors() {
        let mut metadata = base_metadata(AEMAttitudeType::Quaternion);
        metadata.useable_stop_time = Some(t1() + 10.0);
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("USEABLE_STOP_TIME"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_useable_start_after_useable_stop_errors() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion).with_useable_times(t1(), t0());
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("must not fall after"));
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_useable_times_within_bounds_ok() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion).with_useable_times(t0(), t1());
        assert!(metadata.validate().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_metadata_validate_start_after_stop_errors() {
        let mut metadata = base_metadata(AEMAttitudeType::Quaternion);
        metadata.start_time = t1();
        metadata.stop_time = t0();
        let err = metadata.validate().unwrap_err();
        assert!(err.to_string().contains("START_TIME"));
    }

    // ------------------------------------------------------------------
    // AEMSegment
    // ------------------------------------------------------------------

    fn quaternion_state(epoch: Epoch) -> AEMAttitudeState {
        AEMAttitudeState {
            epoch,
            data: AEMAttitudeData::Quaternion {
                quaternion: unit_quaternion(),
            },
        }
    }

    #[test]
    #[parallel]
    fn test_aem_segment_new_defaults() {
        let segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        assert!(segment.comments.is_empty());
        assert!(segment.states.is_empty());
    }

    #[test]
    #[parallel]
    fn test_aem_segment_push_state_increasing_epochs_ok() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        assert!(segment.push_state(quaternion_state(t0())).is_ok());
        assert!(segment.push_state(quaternion_state(t1())).is_ok());
        assert_eq!(segment.states.len(), 2);
    }

    #[test]
    #[parallel]
    fn test_aem_segment_push_state_wrong_type_errors_naming_both_types() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        let state = AEMAttitudeState {
            epoch: t0(),
            data: AEMAttitudeData::Spin {
                spin_alpha: 0.0,
                spin_delta: 0.0,
                spin_angle: 0.0,
                spin_angle_vel: 0.0,
            },
        };
        let err = segment.push_state(state).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("SPIN"));
        assert!(msg.contains("QUATERNION"));
        assert!(segment.states.is_empty());
    }

    #[test]
    #[parallel]
    fn test_aem_segment_push_state_equal_epoch_errors_naming_both_epochs() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t0())).unwrap();
        let err = segment.push_state(quaternion_state(t0())).unwrap_err();
        let msg = err.to_string();
        assert_eq!(msg.matches(&t0().to_string()).count(), 2);
        assert_eq!(segment.states.len(), 1);
    }

    #[test]
    #[parallel]
    fn test_aem_segment_push_state_decreasing_epoch_errors() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t1())).unwrap();
        let err = segment.push_state(quaternion_state(t0())).unwrap_err();
        assert!(err.to_string().contains("not strictly increasing"));
        assert_eq!(segment.states.len(), 1);
    }

    #[test]
    #[parallel]
    fn test_aem_segment_push_state_euler_order_mismatch_errors_naming_both_orders() {
        let metadata =
            base_metadata(AEMAttitudeType::EulerAngle).with_euler_rot_seq(EulerAngleOrder::ZXZ);
        let mut segment = AEMSegment::new(metadata);
        let state = AEMAttitudeState {
            epoch: t0(),
            data: AEMAttitudeData::EulerAngle {
                angles: EulerAngle::new(
                    EulerAngleOrder::XYZ,
                    0.0,
                    0.0,
                    0.0,
                    crate::constants::AngleFormat::Radians,
                ),
            },
        };
        let err = segment.push_state(state).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("XYZ"));
        assert!(msg.contains("ZXZ"));
        assert!(segment.states.is_empty());
    }

    #[test]
    #[parallel]
    fn test_aem_segment_push_state_euler_order_match_ok() {
        let metadata =
            base_metadata(AEMAttitudeType::EulerAngle).with_euler_rot_seq(EulerAngleOrder::ZXZ);
        let mut segment = AEMSegment::new(metadata);
        let state = AEMAttitudeState {
            epoch: t0(),
            data: AEMAttitudeData::EulerAngle {
                angles: zero_euler_angle(),
            },
        };
        assert!(segment.push_state(state).is_ok());
    }

    // ------------------------------------------------------------------
    // AEM
    // ------------------------------------------------------------------

    #[test]
    #[parallel]
    fn test_aem_new_defaults() {
        let aem = AEM::new("BRAHE");
        assert_eq!(aem.header.originator, "BRAHE");
        assert!(aem.segments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_aem_push_segment() {
        let mut aem = AEM::new("BRAHE");
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t0())).unwrap();
        aem.push_segment(segment);
        assert_eq!(aem.segments.len(), 1);
        assert_eq!(aem.segments[0].states.len(), 1);
    }

    #[test]
    #[parallel]
    fn test_aem_to_file_from_file_round_trip() {
        use crate::ccsds::common::CCSDSFormat;

        let mut aem = AEM::new("BRAHE");
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t0())).unwrap();
        segment.push_state(quaternion_state(t1())).unwrap();
        aem.push_segment(segment);

        let dir = std::env::temp_dir();
        let path = dir.join("brahe_test_aem_round_trip.txt");
        aem.to_file(&path, CCSDSFormat::KVN).unwrap();
        let aem2 = AEM::from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(aem2.segments.len(), 1);
        assert_eq!(aem2.segments[0].states.len(), 2);
        assert_eq!(
            aem2.segments[0].metadata.object_name,
            aem.segments[0].metadata.object_name
        );
    }

    #[test]
    #[parallel]
    fn test_aem_from_file_nonexistent() {
        let result = AEM::from_file("nonexistent_aem_file.txt");
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // XML + JSON + 5-method wiring (Task 3)
    // ------------------------------------------------------------------

    use crate::ccsds::common::CCSDSFormat;

    fn assert_quaternion_close(a: &Quaternion, b: &Quaternion) {
        let va = a.to_vector(false);
        let vb = b.to_vector(false);
        for i in 0..4 {
            assert!(
                (va[i] - vb[i]).abs() < 1e-9,
                "quaternion component {} mismatch: {} vs {}",
                i,
                va[i],
                vb[i]
            );
        }
    }

    fn assert_euler_angle_close(a: &EulerAngle, b: &EulerAngle) {
        assert_eq!(a.order, b.order);
        assert!((a.phi - b.phi).abs() < 1e-9);
        assert!((a.theta - b.theta).abs() < 1e-9);
        assert!((a.psi - b.psi).abs() < 1e-9);
    }

    /// Compares every field of two [`AEMAttitudeData`] values, including all
    /// nine variants (the complete comparator; the Task 2 KVN round-trip
    /// helper only covered Quaternion/Spin).
    fn assert_aem_attitude_data_match(a: &AEMAttitudeData, b: &AEMAttitudeData) {
        match (a, b) {
            (
                AEMAttitudeData::Quaternion { quaternion: qa },
                AEMAttitudeData::Quaternion { quaternion: qb },
            ) => {
                assert_quaternion_close(qa, qb);
            }
            (
                AEMAttitudeData::QuaternionDerivative {
                    quaternion: qa,
                    derivative: da,
                },
                AEMAttitudeData::QuaternionDerivative {
                    quaternion: qb,
                    derivative: db,
                },
            ) => {
                assert_quaternion_close(qa, qb);
                for i in 0..4 {
                    assert!((da[i] - db[i]).abs() < 1e-9);
                }
            }
            (
                AEMAttitudeData::QuaternionAngVel {
                    quaternion: qa,
                    angular_velocity: wa,
                },
                AEMAttitudeData::QuaternionAngVel {
                    quaternion: qb,
                    angular_velocity: wb,
                },
            ) => {
                assert_quaternion_close(qa, qb);
                for i in 0..3 {
                    assert!((wa[i] - wb[i]).abs() < 1e-9);
                }
            }
            (
                AEMAttitudeData::EulerAngle { angles: ea },
                AEMAttitudeData::EulerAngle { angles: eb },
            ) => {
                assert_euler_angle_close(ea, eb);
            }
            (
                AEMAttitudeData::EulerAngleDerivative {
                    angles: ea,
                    rates: ra,
                },
                AEMAttitudeData::EulerAngleDerivative {
                    angles: eb,
                    rates: rb,
                },
            ) => {
                assert_euler_angle_close(ea, eb);
                for i in 0..3 {
                    assert!((ra[i] - rb[i]).abs() < 1e-9);
                }
            }
            (
                AEMAttitudeData::EulerAngleAngVel {
                    angles: ea,
                    angular_velocity: wa,
                },
                AEMAttitudeData::EulerAngleAngVel {
                    angles: eb,
                    angular_velocity: wb,
                },
            ) => {
                assert_euler_angle_close(ea, eb);
                for i in 0..3 {
                    assert!((wa[i] - wb[i]).abs() < 1e-9);
                }
            }
            (
                AEMAttitudeData::Spin {
                    spin_alpha: aa,
                    spin_delta: da,
                    spin_angle: ga,
                    spin_angle_vel: va,
                },
                AEMAttitudeData::Spin {
                    spin_alpha: ab,
                    spin_delta: db,
                    spin_angle: gb,
                    spin_angle_vel: vb,
                },
            ) => {
                assert!((aa - ab).abs() < 1e-9);
                assert!((da - db).abs() < 1e-9);
                assert!((ga - gb).abs() < 1e-9);
                assert!((va - vb).abs() < 1e-9);
            }
            (
                AEMAttitudeData::SpinNutation {
                    spin_alpha: aa,
                    spin_delta: da,
                    spin_angle: ga,
                    spin_angle_vel: va,
                    nutation: na,
                    nutation_period: pa,
                    nutation_phase: ha,
                },
                AEMAttitudeData::SpinNutation {
                    spin_alpha: ab,
                    spin_delta: db,
                    spin_angle: gb,
                    spin_angle_vel: vb,
                    nutation: nb,
                    nutation_period: pb,
                    nutation_phase: hb,
                },
            ) => {
                assert!((aa - ab).abs() < 1e-9);
                assert!((da - db).abs() < 1e-9);
                assert!((ga - gb).abs() < 1e-9);
                assert!((va - vb).abs() < 1e-9);
                assert!((na - nb).abs() < 1e-9);
                assert!((pa - pb).abs() < 1e-6);
                assert!((ha - hb).abs() < 1e-9);
            }
            (
                AEMAttitudeData::SpinNutationMom {
                    spin_alpha: aa,
                    spin_delta: da,
                    spin_angle: ga,
                    spin_angle_vel: va,
                    momentum_alpha: ma,
                    momentum_delta: mda,
                    nutation_vel: nva,
                },
                AEMAttitudeData::SpinNutationMom {
                    spin_alpha: ab,
                    spin_delta: db,
                    spin_angle: gb,
                    spin_angle_vel: vb,
                    momentum_alpha: mb,
                    momentum_delta: mdb,
                    nutation_vel: nvb,
                },
            ) => {
                assert!((aa - ab).abs() < 1e-9);
                assert!((da - db).abs() < 1e-9);
                assert!((ga - gb).abs() < 1e-9);
                assert!((va - vb).abs() < 1e-9);
                assert!((ma - mb).abs() < 1e-9);
                assert!((mda - mdb).abs() < 1e-9);
                assert!((nva - nvb).abs() < 1e-9);
            }
            (a, b) => panic!("attitude data variant mismatch: {:?} vs {:?}", a, b),
        }
    }

    /// Compares every field of two [`AEM`] messages, including all header,
    /// metadata, and per-segment/per-state comment vectors.
    fn assert_aem_fields_match(a: &AEM, b: &AEM) {
        // Header
        assert!((a.header.format_version - b.header.format_version).abs() < 1e-9);
        assert_eq!(a.header.classification, b.header.classification);
        assert_eq!(a.header.originator, b.header.originator);
        assert_eq!(a.header.message_id, b.header.message_id);
        assert_eq!(a.header.comments, b.header.comments);

        assert_eq!(a.segments.len(), b.segments.len());
        for (sa, sb) in a.segments.iter().zip(b.segments.iter()) {
            assert_eq!(sa.metadata.object_name, sb.metadata.object_name);
            assert_eq!(sa.metadata.object_id, sb.metadata.object_id);
            assert_eq!(sa.metadata.center_name, sb.metadata.center_name);
            assert_eq!(sa.metadata.ref_frame_a, sb.metadata.ref_frame_a);
            assert_eq!(sa.metadata.ref_frame_b, sb.metadata.ref_frame_b);
            assert_eq!(sa.metadata.time_system, sb.metadata.time_system);
            assert!((sa.metadata.start_time - sb.metadata.start_time).abs() < 1e-6);
            assert!((sa.metadata.stop_time - sb.metadata.stop_time).abs() < 1e-6);
            assert_eq!(
                sa.metadata.useable_start_time.is_some(),
                sb.metadata.useable_start_time.is_some()
            );
            if let (Some(ta), Some(tb)) = (
                sa.metadata.useable_start_time,
                sb.metadata.useable_start_time,
            ) {
                assert!((ta - tb).abs() < 1e-6);
            }
            assert_eq!(
                sa.metadata.useable_stop_time.is_some(),
                sb.metadata.useable_stop_time.is_some()
            );
            if let (Some(ta), Some(tb)) =
                (sa.metadata.useable_stop_time, sb.metadata.useable_stop_time)
            {
                assert!((ta - tb).abs() < 1e-6);
            }
            assert_eq!(sa.metadata.attitude_type, sb.metadata.attitude_type);
            assert_eq!(sa.metadata.euler_rot_seq, sb.metadata.euler_rot_seq);
            assert_eq!(sa.metadata.angvel_frame, sb.metadata.angvel_frame);
            assert_eq!(
                sa.metadata.interpolation_method,
                sb.metadata.interpolation_method
            );
            assert_eq!(
                sa.metadata.interpolation_degree,
                sb.metadata.interpolation_degree
            );
            assert_eq!(sa.metadata.comments, sb.metadata.comments);

            assert_eq!(sa.comments, sb.comments);

            assert_eq!(sa.states.len(), sb.states.len());
            for (state_a, state_b) in sa.states.iter().zip(sb.states.iter()) {
                assert!((state_a.epoch - state_b.epoch).abs() < 1e-6);
                assert_aem_attitude_data_match(&state_a.data, &state_b.data);
            }
        }
    }

    fn aem_g4() -> AEM {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        AEM::from_str(&content).unwrap()
    }

    fn aem_g5() -> AEM {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG5.txt").unwrap();
        AEM::from_str(&content).unwrap()
    }

    fn aem_g11() -> AEM {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG11.xml").unwrap();
        AEM::from_str(&content).unwrap()
    }

    #[test]
    #[parallel]
    fn test_aem_g11_xml_parse_fields() {
        let aem = aem_g11();

        assert!((aem.header.format_version - 2.0).abs() < 1e-9);
        assert_eq!(aem.header.originator, "GSFC/FDF");
        assert_eq!(aem.header.message_id.as_deref(), Some("7077456"));

        assert_eq!(aem.segments.len(), 1);
        let seg = &aem.segments[0];
        assert_eq!(seg.metadata.object_name, "ST5-224");
        assert_eq!(seg.metadata.object_id, "2006-224A");
        assert_eq!(seg.metadata.center_name.as_deref(), Some("EARTH"));
        assert_eq!(seg.metadata.ref_frame_a, ADMReferenceFrame::parse("J2000"));
        assert_eq!(
            seg.metadata.ref_frame_b,
            ADMReferenceFrame::parse("SC_BODY_1")
        );
        assert_eq!(seg.metadata.attitude_type, AEMAttitudeType::Spin);
        assert_eq!(seg.comments, vec!["Spin KF ground solution, SPINKF rates"]);

        assert_eq!(seg.states.len(), 8);

        let first = &seg.states[0].data;
        match first {
            AEMAttitudeData::Spin {
                spin_alpha,
                spin_delta,
                spin_angle,
                spin_angle_vel,
            } => {
                assert!((spin_alpha - 2.6862511e2_f64.to_radians()).abs() < 1e-6);
                assert!((spin_delta - 6.8448486e1_f64.to_radians()).abs() < 1e-6);
                assert!((spin_angle - 1.5969509e2_f64.to_radians()).abs() < 1e-6);
                assert!((spin_angle_vel - (-1.0996528e2_f64).to_radians()).abs() < 1e-6);
            }
            other => panic!("expected Spin, got {:?}", other),
        }

        let last = &seg.states[7].data;
        match last {
            AEMAttitudeData::Spin {
                spin_alpha,
                spin_delta,
                spin_angle,
                spin_angle_vel,
            } => {
                assert!((spin_alpha - 2.6843571e2_f64.to_radians()).abs() < 1e-6);
                assert!((spin_delta - 6.8332398e1_f64.to_radians()).abs() < 1e-6);
                assert!((spin_angle - 6.3662262e1_f64.to_radians()).abs() < 1e-6);
                assert!((spin_angle_vel - (-1.0996304e2_f64).to_radians()).abs() < 1e-6);
            }
            other => panic!("expected Spin, got {:?}", other),
        }
    }

    #[test]
    #[parallel]
    fn test_aem_g11_xml_round_trip() {
        let aem1 = aem_g11();
        let xml = aem1.to_string(CCSDSFormat::XML).unwrap();
        let aem2 = AEM::from_str(&xml).unwrap();
        assert_aem_fields_match(&aem1, &aem2);
    }

    #[test]
    #[parallel]
    fn test_aem_g11_json_round_trip_lower() {
        let aem1 = aem_g11();
        let json = aem1.to_string(CCSDSFormat::JSON).unwrap();
        let aem2 = AEM::from_str(&json).unwrap();
        assert_aem_fields_match(&aem1, &aem2);
    }

    #[test]
    #[parallel]
    fn test_aem_g11_json_round_trip_upper() {
        let aem1 = aem_g11();
        let json = aem1.to_json_string(CCSDSJsonKeyCase::Upper).unwrap();
        let aem2 = AEM::from_str(&json).unwrap();
        assert_aem_fields_match(&aem1, &aem2);
    }

    #[test]
    #[parallel]
    fn test_aem_header_comments_and_classification_xml_json_round_trip() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t0())).unwrap();

        let mut aem1 = AEM::new("BRAHE");
        aem1.header = aem1
            .header
            .with_classification("UNCLASSIFIED")
            .with_comments(vec![
                "first header comment".to_string(),
                "second header comment".to_string(),
            ]);
        aem1.push_segment(segment);

        let xml = aem1.to_string(CCSDSFormat::XML).unwrap();
        assert!(xml.contains("<CLASSIFICATION>UNCLASSIFIED</CLASSIFICATION>"));
        assert!(xml.contains("<COMMENT>first header comment</COMMENT>"));
        let aem_xml = AEM::from_str(&xml).unwrap();
        assert_aem_fields_match(&aem1, &aem_xml);

        let json = aem1.to_string(CCSDSFormat::JSON).unwrap();
        assert!(json.contains("UNCLASSIFIED"));
        assert!(json.contains("first header comment"));
        let aem_json = AEM::from_str(&json).unwrap();
        assert_aem_fields_match(&aem1, &aem_json);
    }

    #[test]
    #[parallel]
    fn test_aem_g4_three_way_round_trip() {
        let aem1 = aem_g4();

        let kvn = aem1.to_string(CCSDSFormat::KVN).unwrap();
        let aem_kvn = AEM::from_str(&kvn).unwrap();
        assert_aem_fields_match(&aem1, &aem_kvn);

        let xml = aem1.to_string(CCSDSFormat::XML).unwrap();
        let aem_xml = AEM::from_str(&xml).unwrap();
        assert_aem_fields_match(&aem1, &aem_xml);

        let json = aem1.to_string(CCSDSFormat::JSON).unwrap();
        let aem_json = AEM::from_str(&json).unwrap();
        assert_aem_fields_match(&aem1, &aem_json);
    }

    #[test]
    #[parallel]
    fn test_aem_g5_three_way_round_trip() {
        let aem1 = aem_g5();

        let kvn = aem1.to_string(CCSDSFormat::KVN).unwrap();
        let aem_kvn = AEM::from_str(&kvn).unwrap();
        assert_aem_fields_match(&aem1, &aem_kvn);

        let xml = aem1.to_string(CCSDSFormat::XML).unwrap();
        let aem_xml = AEM::from_str(&xml).unwrap();
        assert_aem_fields_match(&aem1, &aem_xml);

        let json = aem1.to_string(CCSDSFormat::JSON).unwrap();
        let aem_json = AEM::from_str(&json).unwrap();
        assert_aem_fields_match(&aem1, &aem_json);
    }

    #[test]
    #[parallel]
    fn test_aem_format_detection_round_trip_all_formats() {
        let aem1 = aem_g4();

        let kvn = aem1.to_string(CCSDSFormat::KVN).unwrap();
        let xml = aem1.to_string(CCSDSFormat::XML).unwrap();
        let json = aem1.to_string(CCSDSFormat::JSON).unwrap();

        assert_eq!(crate::ccsds::common::detect_format(&kvn), CCSDSFormat::KVN);
        assert_eq!(crate::ccsds::common::detect_format(&xml), CCSDSFormat::XML);
        assert_eq!(
            crate::ccsds::common::detect_format(&json),
            CCSDSFormat::JSON
        );

        assert_aem_fields_match(&aem1, &AEM::from_str(&kvn).unwrap());
        assert_aem_fields_match(&aem1, &AEM::from_str(&xml).unwrap());
        assert_aem_fields_match(&aem1, &AEM::from_str(&json).unwrap());
    }

    /// Builds an AEM with a single EULER_ANGLE segment whose metadata is
    /// missing `EULER_ROT_SEQ` (a conditional-validation violation per
    /// 504.0-B-2 table 4-3). `push_state` only checks the attitude-type
    /// match, not `validate()`, so this builds successfully; every writer
    /// calls `AEM::validate_for_write` (which calls `AEMMetadata::validate`
    /// on each segment) before serializing, so all three formats should
    /// fail identically on write, before ever reaching a parser.
    fn build_missing_euler_rot_seq_aem() -> AEM {
        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            icrf(),
            sc_body_1(),
            CCSDSTimeSystem::UTC,
            t0(),
            t1(),
            AEMAttitudeType::EulerAngle,
        );
        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0(),
                data: AEMAttitudeData::EulerAngle {
                    angles: zero_euler_angle(),
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);
        aem
    }

    #[test]
    #[parallel]
    fn test_aem_conditional_validation_error_consistent_across_formats() {
        let aem = build_missing_euler_rot_seq_aem();

        let kvn_err = aem.to_string(CCSDSFormat::KVN).unwrap_err().to_string();
        let xml_err = aem.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        let json_err = aem.to_string(CCSDSFormat::JSON).unwrap_err().to_string();

        assert!(kvn_err.contains("EULER_ROT_SEQ"), "{}", kvn_err);
        assert!(xml_err.contains("EULER_ROT_SEQ"), "{}", xml_err);
        assert!(json_err.contains("EULER_ROT_SEQ"), "{}", json_err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_empty_segments_errors() {
        let aem = AEM::new("BRAHE");
        let err = aem.validate_for_write().unwrap_err().to_string();
        assert!(err.contains("segment"), "{}", err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_empty_states_errors() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion);
        let segment = AEMSegment::new(metadata);
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let err = aem.validate_for_write().unwrap_err().to_string();
        assert!(err.contains("attitude state"), "{}", err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_ok_for_valid_message() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t0())).unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        assert!(aem.validate_for_write().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_rejects_state_type_mismatch_via_direct_mutation() {
        // `states` is a public field, so a caller can bypass `push_state`'s
        // type check by mutating it directly; `validate_for_write` must
        // catch this before any writer serializes it.
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t0())).unwrap();
        segment.states.push(AEMAttitudeState {
            epoch: t1(),
            data: AEMAttitudeData::Spin {
                spin_alpha: 0.0,
                spin_delta: 0.0,
                spin_angle: 0.0,
                spin_angle_vel: 0.0,
            },
        });
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let err = aem.validate_for_write().unwrap_err().to_string();
        assert!(err.contains("SPIN"), "{}", err);
        assert!(err.contains("QUATERNION"), "{}", err);

        let kvn_err = aem.to_string(CCSDSFormat::KVN).unwrap_err().to_string();
        let xml_err = aem.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        let json_err = aem.to_string(CCSDSFormat::JSON).unwrap_err().to_string();
        assert!(kvn_err.contains("SPIN"), "{}", kvn_err);
        assert!(xml_err.contains("SPIN"), "{}", xml_err);
        assert!(json_err.contains("SPIN"), "{}", json_err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_rejects_decreasing_epoch_via_direct_mutation() {
        let mut segment = AEMSegment::new(base_metadata(AEMAttitudeType::Quaternion));
        segment.push_state(quaternion_state(t1())).unwrap();
        segment.states.push(quaternion_state(t0()));
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let err = aem.validate_for_write().unwrap_err().to_string();
        assert!(err.contains("not strictly increasing"), "{}", err);

        let kvn_err = aem.to_string(CCSDSFormat::KVN).unwrap_err().to_string();
        let xml_err = aem.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        let json_err = aem.to_string(CCSDSFormat::JSON).unwrap_err().to_string();
        assert!(kvn_err.contains("not strictly increasing"), "{}", kvn_err);
        assert!(xml_err.contains("not strictly increasing"), "{}", xml_err);
        assert!(json_err.contains("not strictly increasing"), "{}", json_err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_rejects_unmappable_time_system() {
        let mut metadata = base_metadata(AEMAttitudeType::Quaternion);
        metadata.time_system = CCSDSTimeSystem::MET;
        let mut segment = AEMSegment::new(metadata);
        segment.push_state(quaternion_state(t0())).unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let err = aem.validate_for_write().unwrap_err().to_string();
        assert!(err.contains("TIME_SYSTEM"), "{}", err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_rejects_out_of_order_useable_times_between_segments() {
        let mut segment0 = AEMSegment::new(
            base_metadata(AEMAttitudeType::Quaternion).with_useable_times(t0(), t1()),
        );
        segment0.push_state(quaternion_state(t0())).unwrap();

        let mut segment1 = AEMSegment::new(
            base_metadata(AEMAttitudeType::Quaternion).with_useable_times(t0(), t0()),
        );
        segment1.push_state(quaternion_state(t0())).unwrap();

        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment0);
        aem.push_segment(segment1);

        let err = aem.validate_for_write().unwrap_err().to_string();
        assert!(err.contains("USEABLE_START_TIME"), "{}", err);
        assert!(err.contains("USEABLE_STOP_TIME"), "{}", err);
    }

    #[test]
    #[parallel]
    fn test_aem_validate_for_write_ok_for_in_order_useable_times_between_segments() {
        let mut segment0 = AEMSegment::new(
            base_metadata(AEMAttitudeType::Quaternion).with_useable_times(t0(), t1()),
        );
        segment0.push_state(quaternion_state(t0())).unwrap();

        let t2 = t1() + 60.0;
        let metadata1 = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            icrf(),
            sc_body_1(),
            CCSDSTimeSystem::UTC,
            t1(),
            t2,
            AEMAttitudeType::Quaternion,
        )
        .with_useable_times(t1(), t2);
        let mut segment1 = AEMSegment::new(metadata1);
        segment1.push_state(quaternion_state(t1())).unwrap();

        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment0);
        aem.push_segment(segment1);

        assert!(aem.validate_for_write().is_ok());
    }

    #[test]
    #[parallel]
    fn test_aem_write_all_formats_reject_empty_segments() {
        let aem = AEM::new("BRAHE");

        let kvn_err = aem.to_string(CCSDSFormat::KVN).unwrap_err().to_string();
        let xml_err = aem.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        let json_err = aem.to_string(CCSDSFormat::JSON).unwrap_err().to_string();

        assert!(kvn_err.contains("segment"), "{}", kvn_err);
        assert!(xml_err.contains("segment"), "{}", xml_err);
        assert!(json_err.contains("segment"), "{}", json_err);
    }

    #[test]
    #[parallel]
    fn test_aem_write_all_formats_reject_empty_states() {
        let metadata = base_metadata(AEMAttitudeType::Quaternion);
        let segment = AEMSegment::new(metadata);
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let kvn_err = aem.to_string(CCSDSFormat::KVN).unwrap_err().to_string();
        let xml_err = aem.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        let json_err = aem.to_string(CCSDSFormat::JSON).unwrap_err().to_string();

        assert!(kvn_err.contains("attitude state"), "{}", kvn_err);
        assert!(xml_err.contains("attitude state"), "{}", xml_err);
        assert!(json_err.contains("attitude state"), "{}", json_err);
    }

    /// Builds a synthetic AEM with nine single-state segments, one per
    /// [`AEMAttitudeType`] variant, closing the variant-coverage gap left by
    /// the G-4 (QUATERNION only) and G-5 (SPIN only) fixtures.
    fn build_all_types_aem() -> AEM {
        let quaternion = Quaternion::new(0.5, 0.5, 0.5, 0.5);
        let euler_angles = EulerAngle::new(
            EulerAngleOrder::ZXZ,
            30.0_f64.to_radians(),
            45.0_f64.to_radians(),
            60.0_f64.to_radians(),
            crate::constants::AngleFormat::Radians,
        );

        let mut aem = AEM::new("BRAHE");

        let mut push = |attitude_type: AEMAttitudeType, data: AEMAttitudeData| {
            let mut metadata = AEMMetadata::new(
                &format!("SAT-{:?}", attitude_type),
                "2024-001A",
                icrf(),
                sc_body_1(),
                CCSDSTimeSystem::UTC,
                t0(),
                t1(),
                attitude_type,
            )
            .with_center_name("EARTH");
            if matches!(
                attitude_type,
                AEMAttitudeType::EulerAngle
                    | AEMAttitudeType::EulerAngleDerivative
                    | AEMAttitudeType::EulerAngleAngVel
            ) {
                metadata = metadata.with_euler_rot_seq(EulerAngleOrder::ZXZ);
            }
            if matches!(
                attitude_type,
                AEMAttitudeType::QuaternionAngVel | AEMAttitudeType::EulerAngleAngVel
            ) {
                metadata = metadata.with_angvel_frame(sc_body_1());
            }
            metadata.validate().unwrap();

            let mut segment = AEMSegment::new(metadata);
            segment.comments = vec![format!("{:?} segment data comment", attitude_type)];
            segment
                .push_state(AEMAttitudeState { epoch: t0(), data })
                .unwrap();
            aem.push_segment(segment);
        };

        push(
            AEMAttitudeType::Quaternion,
            AEMAttitudeData::Quaternion { quaternion },
        );
        push(
            AEMAttitudeType::QuaternionDerivative,
            AEMAttitudeData::QuaternionDerivative {
                quaternion,
                derivative: Vector4::new(0.01, 0.02, 0.03, 0.04),
            },
        );
        push(
            AEMAttitudeType::QuaternionAngVel,
            AEMAttitudeData::QuaternionAngVel {
                quaternion,
                angular_velocity: Vector3::new(0.001, 0.002, 0.003),
            },
        );
        push(
            AEMAttitudeType::EulerAngle,
            AEMAttitudeData::EulerAngle {
                angles: euler_angles,
            },
        );
        push(
            AEMAttitudeType::EulerAngleDerivative,
            AEMAttitudeData::EulerAngleDerivative {
                angles: euler_angles,
                rates: Vector3::new(0.001, 0.002, 0.003),
            },
        );
        push(
            AEMAttitudeType::EulerAngleAngVel,
            AEMAttitudeData::EulerAngleAngVel {
                angles: euler_angles,
                angular_velocity: Vector3::new(0.001, 0.002, 0.003),
            },
        );
        push(
            AEMAttitudeType::Spin,
            AEMAttitudeData::Spin {
                spin_alpha: 0.1,
                spin_delta: 0.2,
                spin_angle: 0.3,
                spin_angle_vel: 0.4,
            },
        );
        push(
            AEMAttitudeType::SpinNutation,
            AEMAttitudeData::SpinNutation {
                spin_alpha: 0.1,
                spin_delta: 0.2,
                spin_angle: 0.3,
                spin_angle_vel: 0.4,
                nutation: 0.05,
                nutation_period: 120.0,
                nutation_phase: 0.06,
            },
        );
        push(
            AEMAttitudeType::SpinNutationMom,
            AEMAttitudeData::SpinNutationMom {
                spin_alpha: 0.1,
                spin_delta: 0.2,
                spin_angle: 0.3,
                spin_angle_vel: 0.4,
                momentum_alpha: 0.07,
                momentum_delta: 0.08,
                nutation_vel: 0.09,
            },
        );

        aem
    }

    #[test]
    #[parallel]
    fn test_aem_all_types_synthetic_three_way_round_trip() {
        let aem1 = build_all_types_aem();
        assert_eq!(aem1.segments.len(), 9);

        let kvn = aem1.to_string(CCSDSFormat::KVN).unwrap();
        let aem_kvn = AEM::from_str(&kvn).unwrap();
        assert_aem_fields_match(&aem1, &aem_kvn);

        let xml = aem1.to_string(CCSDSFormat::XML).unwrap();
        let aem_xml = AEM::from_str(&xml).unwrap();
        assert_aem_fields_match(&aem1, &aem_xml);

        let json = aem1.to_string(CCSDSFormat::JSON).unwrap();
        let aem_json = AEM::from_str(&json).unwrap();
        assert_aem_fields_match(&aem1, &aem_json);
    }
}
