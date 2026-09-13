/*!
 * Modified ITC message types.
 */

use std::fmt;
use std::str::FromStr;

use nalgebra::SMatrix;

use crate::clients::spacetrack::SpaceTrackEphemerisFileName;
use crate::frames::CelestialFrame;
use crate::math::is_symmetric;
use crate::math::linalg::SVector6;
use crate::time::Epoch;
use crate::utils::BraheError;

/// Frame in which a Modified ITC file expresses its covariance.
///
/// The fourth header line names this frame. `UVW` is Space-Track's name for
/// the radial, in-track, cross-track frame and is written for `RTN`.
///
/// # Examples
///
/// ```
/// use brahe::itc::ITCCovarianceFrame;
///
/// assert_eq!(ITCCovarianceFrame::parse("UVW").unwrap(), ITCCovarianceFrame::RTN);
/// assert_eq!(ITCCovarianceFrame::RTN.to_string(), "UVW");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ITCCovarianceFrame {
    /// Radial, in-track (transverse), cross-track; tokens `UVW`, `RTN`, `RSW`, `RIC`.
    RTN,
    /// Mean equator and equinox of J2000.0; tokens `EME2000`, `J2000`.
    EME2000,
    /// International Terrestrial Reference Frame; token `ITRF`.
    ITRF,
}

impl ITCCovarianceFrame {
    /// Parses the covariance-frame header token.
    ///
    /// # Arguments
    /// * `token` - Fourth header line of a Modified ITC file, case-insensitive
    ///
    /// # Returns
    /// * `Ok(ITCCovarianceFrame)`: The frame
    /// * `Err(BraheError)`: If the token is not one the handbook lists
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCCovarianceFrame;
    ///
    /// assert_eq!(ITCCovarianceFrame::parse("J2000").unwrap(), ITCCovarianceFrame::EME2000);
    /// assert!(ITCCovarianceFrame::parse("TEME").is_err());
    /// ```
    pub fn parse(token: &str) -> Result<Self, BraheError> {
        match token.trim().to_ascii_uppercase().as_str() {
            "UVW" | "RTN" | "RSW" | "RIC" => Ok(Self::RTN),
            "EME2000" | "J2000" => Ok(Self::EME2000),
            "ITRF" => Ok(Self::ITRF),
            other => Err(BraheError::ParseError(format!(
                "unknown Modified ITC covariance frame '{}'; expected UVW, RTN, RSW, RIC, EME2000, J2000 or ITRF",
                other
            ))),
        }
    }

    /// The token written on the fourth header line.
    ///
    /// # Returns
    /// * `&'static str`: `UVW`, `EME2000` or `ITRF`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCCovarianceFrame;
    ///
    /// assert_eq!(ITCCovarianceFrame::ITRF.token(), "ITRF");
    /// ```
    pub fn token(&self) -> &'static str {
        match self {
            Self::RTN => "UVW",
            Self::EME2000 => "EME2000",
            Self::ITRF => "ITRF",
        }
    }
}

impl fmt::Display for ITCCovarianceFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl FromStr for ITCCovarianceFrame {
    type Err = BraheError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Header of a Modified ITC file.
///
/// The first three header lines are descriptive and Space-Track ignores
/// them; Starlink writes them as `created:`, `ephemeris_start: ...
/// ephemeris_stop: ... step_size:` and `ephemeris_source:`. The typed
/// fields hold those values when present. `state_frame` is not written in
/// the file body; it comes from the file name's DataType (default
/// `EME2000`, the handbook's MEME J2000.0).
///
/// # Examples
///
/// ```
/// use brahe::itc::{ITCCovarianceFrame, ITCHeader};
/// use brahe::frames::CelestialFrame;
///
/// let header = ITCHeader::new().with_ephemeris_source("blend");
/// assert_eq!(header.state_frame, CelestialFrame::EME2000);
/// assert_eq!(header.covariance_frame, ITCCovarianceFrame::RTN);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ITCHeader {
    /// File creation time.
    pub created: Option<Epoch>,
    /// Declared first epoch.
    pub ephemeris_start: Option<Epoch>,
    /// Declared last epoch.
    pub ephemeris_stop: Option<Epoch>,
    /// Declared step between records, seconds.
    pub step_size: Option<f64>,
    /// Free-text source label, for example `blend`.
    pub ephemeris_source: Option<String>,
    /// Frame of the state vectors.
    pub state_frame: CelestialFrame,
    /// Frame of the covariance matrices.
    pub covariance_frame: ITCCovarianceFrame,
}

impl Default for ITCHeader {
    fn default() -> Self {
        Self {
            created: None,
            ephemeris_start: None,
            ephemeris_stop: None,
            step_size: None,
            ephemeris_source: None,
            state_frame: CelestialFrame::EME2000,
            covariance_frame: ITCCovarianceFrame::RTN,
        }
    }
}

impl ITCHeader {
    /// Creates a header with `EME2000` states, `RTN` covariance and no
    /// descriptive fields.
    ///
    /// # Returns
    /// * `ITCHeader`: The default header
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCHeader;
    ///
    /// let header = ITCHeader::new();
    /// assert!(header.created.is_none());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the creation time.
    ///
    /// # Arguments
    /// * `created` - File creation epoch
    ///
    /// # Returns
    /// * `ITCHeader`: The header with `created` set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCHeader;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let created = Epoch::from_datetime(2026, 9, 11, 1, 55, 52.0, 0.0, TimeSystem::UTC);
    /// let header = ITCHeader::new().with_created(created);
    /// assert_eq!(header.created, Some(created));
    /// ```
    pub fn with_created(mut self, created: Epoch) -> Self {
        self.created = Some(created);
        self
    }

    /// Sets the ephemeris source label.
    ///
    /// # Arguments
    /// * `source` - Free text, for example `blend`
    ///
    /// # Returns
    /// * `ITCHeader`: The header with the source set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCHeader;
    ///
    /// let header = ITCHeader::new().with_ephemeris_source("blend");
    /// assert_eq!(header.ephemeris_source.as_deref(), Some("blend"));
    /// ```
    pub fn with_ephemeris_source(mut self, source: &str) -> Self {
        self.ephemeris_source = Some(source.to_string());
        self
    }

    /// Sets the frame of the state vectors.
    ///
    /// # Arguments
    /// * `frame` - State frame; `EME2000`, `TEME` and `ITRF` have file-name data types
    ///
    /// # Returns
    /// * `ITCHeader`: The header with the state frame set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCHeader;
    /// use brahe::frames::CelestialFrame;
    ///
    /// let header = ITCHeader::new().with_state_frame(CelestialFrame::TEME);
    /// assert_eq!(header.state_frame, CelestialFrame::TEME);
    /// ```
    pub fn with_state_frame(mut self, frame: CelestialFrame) -> Self {
        self.state_frame = frame;
        self
    }

    /// Sets the frame of the covariance matrices.
    ///
    /// # Arguments
    /// * `frame` - Covariance frame
    ///
    /// # Returns
    /// * `ITCHeader`: The header with the covariance frame set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITCCovarianceFrame, ITCHeader};
    ///
    /// let header = ITCHeader::new().with_covariance_frame(ITCCovarianceFrame::EME2000);
    /// assert_eq!(header.covariance_frame, ITCCovarianceFrame::EME2000);
    /// ```
    pub fn with_covariance_frame(mut self, frame: ITCCovarianceFrame) -> Self {
        self.covariance_frame = frame;
        self
    }
}

/// One ephemeris record: epoch, position and velocity.
///
/// # Examples
///
/// ```
/// use brahe::itc::ITCStateVector;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epoch = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
/// let state = ITCStateVector::new(epoch, [4.244e6, 1.264e6, 5.044e6], [3595.2, 5258.8, -4335.0]);
/// assert_eq!(state.position[0], 4.244e6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ITCStateVector {
    /// Record epoch, UTC.
    pub epoch: Epoch,
    /// Position in the header's state frame, meters.
    pub position: [f64; 3],
    /// Velocity in the header's state frame, meters per second.
    pub velocity: [f64; 3],
}

impl ITCStateVector {
    /// Creates a record.
    ///
    /// # Arguments
    /// * `epoch` - Record epoch
    /// * `position` - Position, meters
    /// * `velocity` - Velocity, meters per second
    ///
    /// # Returns
    /// * `ITCStateVector`: The record
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCStateVector;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
    /// let state = ITCStateVector::new(epoch, [7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0]);
    /// assert_eq!(state.velocity[1], 7.5e3);
    /// ```
    pub fn new(epoch: Epoch, position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            epoch,
            position,
            velocity,
        }
    }

    /// The record's position and velocity as a single Cartesian state.
    ///
    /// # Returns
    /// * `SVector6`: `[x, y, z, vx, vy, vz]` in the header's state frame, meters and meters per second
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITCStateVector;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
    /// let state = ITCStateVector::new(epoch, [7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0]);
    /// assert_eq!(state.to_vector()[0], 7.0e6);
    /// assert_eq!(state.to_vector()[4], 7.5e3);
    /// ```
    pub fn to_vector(&self) -> SVector6 {
        SVector6::new(
            self.position[0],
            self.position[1],
            self.position[2],
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
        )
    }
}

impl From<&ITCStateVector> for SVector6 {
    fn from(state: &ITCStateVector) -> Self {
        state.to_vector()
    }
}

/// A Modified ITC ephemeris message.
///
/// `covariances` is either empty or has one 6x6 matrix per state, in SI
/// units (m², m²/s, m²/s²) in `header.covariance_frame`. `source_name` is
/// set by [`ITC::from_file`] when the file name follows the Space-Track
/// convention.
///
/// # Examples
///
/// ```
/// use brahe::itc::ITC;
///
/// let itc = ITC::from_file("test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt").unwrap();
/// assert_eq!(itc.len(), 50);
/// assert!(itc.has_covariance());
/// assert_eq!(itc.source_name.as_ref().unwrap().object_name, "STARLINK-37711");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ITC {
    /// Header fields.
    pub header: ITCHeader,
    /// Parsed file name, when loaded from a compliant file name.
    pub source_name: Option<SpaceTrackEphemerisFileName>,
    /// Ephemeris records in increasing epoch order.
    pub states: Vec<ITCStateVector>,
    /// Covariance per record, or empty when the file carries none.
    pub covariances: Vec<SMatrix<f64, 6, 6>>,
}

impl ITC {
    /// Creates an empty message with the given header.
    ///
    /// # Arguments
    /// * `header` - Header fields
    ///
    /// # Returns
    /// * `ITC`: A message with no records
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// let itc = ITC::new(ITCHeader::new());
    /// assert!(itc.is_empty());
    /// ```
    pub fn new(header: ITCHeader) -> Self {
        Self {
            header,
            source_name: None,
            states: Vec::new(),
            covariances: Vec::new(),
        }
    }

    fn check_epoch_order(&self, epoch: Epoch) -> Result<(), BraheError> {
        if let Some(last) = self.states.last()
            && epoch <= last.epoch
        {
            return Err(BraheError::Error(format!(
                "Modified ITC records must have strictly increasing epochs; {} does not follow {}",
                epoch, last.epoch
            )));
        }
        Ok(())
    }

    fn check_finite(
        state: &ITCStateVector,
        covariance: Option<&SMatrix<f64, 6, 6>>,
    ) -> Result<(), BraheError> {
        let finite = state.position.iter().all(|v| v.is_finite())
            && state.velocity.iter().all(|v| v.is_finite())
            && covariance.is_none_or(|c| c.iter().all(|v| v.is_finite()));
        if !finite {
            return Err(BraheError::Error(format!(
                "Modified ITC record at {} has a non-finite position, velocity or covariance element",
                state.epoch
            )));
        }
        Ok(())
    }

    /// Appends a record without covariance.
    ///
    /// # Arguments
    /// * `state` - Record with an epoch later than the last one
    ///
    /// # Returns
    /// * `Ok(())`: Record appended
    /// * `Err(BraheError)`: If the epoch is not increasing, or the message already carries covariance
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader, ITCStateVector};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
    /// let mut itc = ITC::new(ITCHeader::new());
    /// itc.push_state(ITCStateVector::new(epoch, [7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0])).unwrap();
    /// assert_eq!(itc.len(), 1);
    /// ```
    pub fn push_state(&mut self, state: ITCStateVector) -> Result<(), BraheError> {
        if self.has_covariance() {
            return Err(BraheError::Error(
                "Modified ITC covariance is all-or-none; this message already carries covariance"
                    .to_string(),
            ));
        }
        Self::check_finite(&state, None)?;
        self.check_epoch_order(state.epoch)?;
        self.states.push(state);
        Ok(())
    }

    /// Appends a record with its covariance.
    ///
    /// # Arguments
    /// * `state` - Record with an epoch later than the last one
    /// * `covariance` - 6x6 covariance in `header.covariance_frame`, SI units
    ///
    /// # Returns
    /// * `Ok(())`: Record appended
    /// * `Err(BraheError)`: If the epoch is not increasing, or earlier records have no covariance
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader, ITCStateVector};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::SMatrix;
    ///
    /// let epoch = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
    /// let mut itc = ITC::new(ITCHeader::new());
    /// let state = ITCStateVector::new(epoch, [7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0]);
    /// itc.push_state_with_covariance(state, SMatrix::<f64, 6, 6>::identity()).unwrap();
    /// assert!(itc.has_covariance());
    /// ```
    pub fn push_state_with_covariance(
        &mut self,
        state: ITCStateVector,
        covariance: SMatrix<f64, 6, 6>,
    ) -> Result<(), BraheError> {
        if !self.states.is_empty() && !self.has_covariance() {
            return Err(BraheError::Error(
                "Modified ITC covariance is all-or-none; earlier records carry no covariance"
                    .to_string(),
            ));
        }
        Self::check_finite(&state, Some(&covariance))?;
        if !is_symmetric(&covariance, 1.0e-9) {
            return Err(BraheError::Error(format!(
                "Modified ITC covariance at {} is not symmetric",
                state.epoch
            )));
        }
        self.check_epoch_order(state.epoch)?;
        self.states.push(state);
        self.covariances.push(covariance);
        Ok(())
    }

    /// Whether the message carries covariance.
    ///
    /// # Returns
    /// * `bool`: `true` when every record has a covariance matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// assert!(!ITC::new(ITCHeader::new()).has_covariance());
    /// ```
    pub fn has_covariance(&self) -> bool {
        !self.covariances.is_empty()
    }

    /// Number of records.
    ///
    /// # Returns
    /// * `usize`: Record count
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// assert_eq!(ITC::new(ITCHeader::new()).len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the message has no records.
    ///
    /// # Returns
    /// * `bool`: `true` when there are no records
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// assert!(ITC::new(ITCHeader::new()).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Epoch of the first record.
    ///
    /// # Returns
    /// * `Option<Epoch>`: First epoch, or `None` when empty
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// assert!(ITC::new(ITCHeader::new()).start_epoch().is_none());
    /// ```
    pub fn start_epoch(&self) -> Option<Epoch> {
        self.states.first().map(|s| s.epoch)
    }

    /// Epoch of the last record.
    ///
    /// # Returns
    /// * `Option<Epoch>`: Last epoch, or `None` when empty
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::{ITC, ITCHeader};
    ///
    /// assert!(ITC::new(ITCHeader::new()).end_epoch().is_none());
    /// ```
    pub fn end_epoch(&self) -> Option<Epoch> {
        self.states.last().map(|s| s.epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use nalgebra::SMatrix;
    use serial_test::parallel;

    fn epoch(sec: f64) -> Epoch {
        Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC) + sec
    }

    fn state(sec: f64) -> ITCStateVector {
        ITCStateVector::new(epoch(sec), [7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0])
    }

    #[test]
    #[parallel]
    fn test_itc_state_vector_to_vector() {
        let sv = ITCStateVector::new(epoch(0.0), [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let x = sv.to_vector();
        for i in 0..3 {
            assert_eq!(x[i], sv.position[i]);
            assert_eq!(x[3 + i], sv.velocity[i]);
        }
        assert_eq!(SVector6::from(&sv), x);
    }

    #[test]
    #[parallel]
    fn test_itc_covariance_frame_parse_and_token() {
        for token in ["UVW", "uvw", "RTN", "RSW", "RIC"] {
            assert_eq!(
                ITCCovarianceFrame::parse(token).unwrap(),
                ITCCovarianceFrame::RTN
            );
        }
        for token in ["EME2000", "J2000", "j2000"] {
            assert_eq!(
                ITCCovarianceFrame::parse(token).unwrap(),
                ITCCovarianceFrame::EME2000
            );
        }
        assert_eq!(
            ITCCovarianceFrame::parse("ITRF").unwrap(),
            ITCCovarianceFrame::ITRF
        );
        assert!(ITCCovarianceFrame::parse("TEME").is_err());
        assert_eq!(ITCCovarianceFrame::RTN.to_string(), "UVW");
        assert_eq!(ITCCovarianceFrame::EME2000.to_string(), "EME2000");
        assert_eq!(ITCCovarianceFrame::ITRF.to_string(), "ITRF");
        assert_eq!(
            "rsw".parse::<ITCCovarianceFrame>().unwrap(),
            ITCCovarianceFrame::RTN
        );
    }

    #[test]
    #[parallel]
    fn test_itc_header_defaults_and_builders() {
        let header = ITCHeader::new();
        assert_eq!(header.state_frame, CelestialFrame::EME2000);
        assert_eq!(header.covariance_frame, ITCCovarianceFrame::RTN);
        assert!(header.created.is_none());
        assert!(header.ephemeris_start.is_none());
        assert!(header.ephemeris_stop.is_none());
        assert!(header.step_size.is_none());
        assert!(header.ephemeris_source.is_none());

        let header = ITCHeader::new()
            .with_created(epoch(0.0))
            .with_ephemeris_source("blend")
            .with_state_frame(CelestialFrame::TEME)
            .with_covariance_frame(ITCCovarianceFrame::EME2000);
        assert_eq!(header.created, Some(epoch(0.0)));
        assert_eq!(header.ephemeris_source.as_deref(), Some("blend"));
        assert_eq!(header.state_frame, CelestialFrame::TEME);
        assert_eq!(header.covariance_frame, ITCCovarianceFrame::EME2000);
    }

    #[test]
    #[parallel]
    fn test_itc_push_state_and_accessors() {
        let mut itc = ITC::new(ITCHeader::new());
        assert!(itc.is_empty());
        assert!(itc.start_epoch().is_none());
        itc.push_state(state(0.0)).unwrap();
        itc.push_state(state(60.0)).unwrap();
        assert_eq!(itc.len(), 2);
        assert!(!itc.has_covariance());
        assert_eq!(itc.start_epoch(), Some(epoch(0.0)));
        assert_eq!(itc.end_epoch(), Some(epoch(60.0)));
    }

    #[test]
    #[parallel]
    fn test_itc_push_state_rejects_non_increasing_epochs() {
        let mut itc = ITC::new(ITCHeader::new());
        itc.push_state(state(60.0)).unwrap();
        assert!(itc.push_state(state(60.0)).is_err());
        assert!(itc.push_state(state(0.0)).is_err());
        assert_eq!(itc.len(), 1);
    }

    #[test]
    #[parallel]
    fn test_itc_push_state_rejects_non_finite_values() {
        let mut itc = ITC::new(ITCHeader::new());
        let nan_state = ITCStateVector::new(epoch(0.0), [f64::NAN, 0.0, 0.0], [0.0, 7.5e3, 0.0]);
        assert!(itc.push_state(nan_state).is_err());
        assert!(itc.is_empty());
    }

    #[test]
    #[parallel]
    fn test_itc_covariance_all_or_none() {
        let cov = SMatrix::<f64, 6, 6>::identity();
        let mut with_cov = ITC::new(ITCHeader::new());
        with_cov
            .push_state_with_covariance(state(0.0), cov)
            .unwrap();
        assert!(with_cov.has_covariance());
        assert!(with_cov.push_state(state(60.0)).is_err());
        with_cov
            .push_state_with_covariance(state(60.0), cov)
            .unwrap();
        assert_eq!(with_cov.covariances.len(), 2);

        let mut without = ITC::new(ITCHeader::new());
        without.push_state(state(0.0)).unwrap();
        assert!(
            without
                .push_state_with_covariance(state(60.0), cov)
                .is_err()
        );
    }

    #[test]
    #[parallel]
    fn test_itc_push_state_with_covariance_rejects_asymmetric() {
        let mut asymmetric = SMatrix::<f64, 6, 6>::identity();
        asymmetric[(0, 1)] = 1.0;
        asymmetric[(1, 0)] = 2.0;
        let mut itc = ITC::new(ITCHeader::new());
        assert!(
            itc.push_state_with_covariance(state(0.0), asymmetric)
                .is_err()
        );

        let mut noisy = SMatrix::<f64, 6, 6>::identity();
        noisy[(0, 1)] = 1.0 + 1.0e-15;
        noisy[(1, 0)] = 1.0;
        itc.push_state_with_covariance(state(0.0), noisy).unwrap();
        assert_eq!(itc.len(), 1);
    }
}
