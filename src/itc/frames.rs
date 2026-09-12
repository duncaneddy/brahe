/*!
 * Mapping between Space-Track file-name DataType tokens and state frames.
 */

use super::types::ITC;
use crate::clients::spacetrack::{SpaceTrackEphemerisFileCategory, SpaceTrackEphemerisFileName};
use crate::frames::CelestialFrame;
use crate::time::Epoch;
use crate::utils::BraheError;

/// State frame implied by a file name's DataType field.
///
/// `MEME` (the handbook's normative value), `EME2000` and `J2000` all name
/// the mean equator and equinox of J2000.0; `TEME` and `ITRF` name those
/// frames.
///
/// # Arguments
/// * `data_type` - DataType field of a Space-Track ephemeris file name, case-insensitive
///
/// # Returns
/// * `Ok(CelestialFrame)`: `EME2000`, `TEME` or `ITRF`
/// * `Err(BraheError)`: If the token is not one of the supported values
///
/// # Examples
///
/// ```
/// use brahe::itc::state_frame_for_data_type;
/// use brahe::frames::CelestialFrame;
///
/// assert_eq!(state_frame_for_data_type("MEME").unwrap(), CelestialFrame::EME2000);
/// assert!(state_frame_for_data_type("GCRF").is_err());
/// ```
pub fn state_frame_for_data_type(data_type: &str) -> Result<CelestialFrame, BraheError> {
    match data_type.trim().to_ascii_uppercase().as_str() {
        "MEME" | "EME2000" | "J2000" => Ok(CelestialFrame::EME2000),
        "TEME" => Ok(CelestialFrame::TEME),
        "ITRF" => Ok(CelestialFrame::ITRF),
        other => Err(BraheError::ParseError(format!(
            "unsupported ephemeris file DataType '{}'; expected MEME, EME2000, J2000, TEME or ITRF",
            other
        ))),
    }
}

/// DataType token to write for a state frame.
///
/// # Arguments
/// * `frame` - State frame of the ephemeris
///
/// # Returns
/// * `Ok(&'static str)`: `MEME` for `EME2000`, `TEME` for `TEME`, `ITRF` for `ITRF`
/// * `Err(BraheError)`: For any other frame
///
/// # Examples
///
/// ```
/// use brahe::itc::data_type_for_state_frame;
/// use brahe::frames::CelestialFrame;
///
/// assert_eq!(data_type_for_state_frame(&CelestialFrame::EME2000).unwrap(), "MEME");
/// assert!(data_type_for_state_frame(&CelestialFrame::GCRF).is_err());
/// ```
pub fn data_type_for_state_frame(frame: &CelestialFrame) -> Result<&'static str, BraheError> {
    match frame {
        CelestialFrame::EME2000 => Ok("MEME"),
        CelestialFrame::TEME => Ok("TEME"),
        CelestialFrame::ITRF => Ok("ITRF"),
        other => Err(BraheError::Error(format!(
            "no Space-Track ephemeris DataType for frame {}; supported frames are EME2000, TEME and ITRF",
            other
        ))),
    }
}

impl ITC {
    /// Builds a Space-Track compliant file name for this message.
    ///
    /// The day-time group comes from `header.ephemeris_start`, or the first
    /// record when the header does not set it; the DataType comes from
    /// `header.state_frame`.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number or analyst number
    /// * `object_name` - Common name of the object
    /// * `category` - Operational or Special
    /// * `metadata` - Operator-defined metadata, may be empty
    ///
    /// # Returns
    /// * `Ok(SpaceTrackEphemerisFileName)`: The name; call `to_string()` for the text
    /// * `Err(BraheError)`: If the message has no start epoch or its frame has no DataType
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITC;
    /// use brahe::spacetrack::SpaceTrackEphemerisFileCategory;
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let name = itc.file_name(100002, "STARLINK-37711", SpaceTrackEphemerisFileCategory::Operational, "").unwrap();
    /// assert_eq!(name.to_string(), "MEME_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt");
    /// ```
    pub fn file_name(
        &self,
        norad_cat_id: u32,
        object_name: &str,
        category: SpaceTrackEphemerisFileCategory,
        metadata: &str,
    ) -> Result<SpaceTrackEphemerisFileName, BraheError> {
        let start: Epoch = self
            .header
            .ephemeris_start
            .or_else(|| self.start_epoch())
            .ok_or_else(|| {
                BraheError::Error(
                    "cannot build a file name: the message has no start epoch".to_string(),
                )
            })?;
        let data_type = data_type_for_state_frame(&self.header.state_frame)?;
        SpaceTrackEphemerisFileName::new(norad_cat_id, object_name, start, category, metadata)?
            .with_data_type(data_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itc::{ITCHeader, ITCStateVector};
    use crate::time::TimeSystem;
    use serial_test::parallel;

    const TRUNCATED: &str = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";

    #[test]
    #[parallel]
    fn test_state_frame_for_data_type() {
        for token in ["MEME", "meme", "EME2000", "J2000"] {
            assert_eq!(
                state_frame_for_data_type(token).unwrap(),
                CelestialFrame::EME2000
            );
        }
        assert_eq!(
            state_frame_for_data_type("TEME").unwrap(),
            CelestialFrame::TEME
        );
        assert_eq!(
            state_frame_for_data_type("ITRF").unwrap(),
            CelestialFrame::ITRF
        );
        assert!(state_frame_for_data_type("GCRF").is_err());
        assert!(state_frame_for_data_type("").is_err());
    }

    #[test]
    #[parallel]
    fn test_data_type_for_state_frame() {
        assert_eq!(
            data_type_for_state_frame(&CelestialFrame::EME2000).unwrap(),
            "MEME"
        );
        assert_eq!(
            data_type_for_state_frame(&CelestialFrame::TEME).unwrap(),
            "TEME"
        );
        assert_eq!(
            data_type_for_state_frame(&CelestialFrame::ITRF).unwrap(),
            "ITRF"
        );
        assert!(data_type_for_state_frame(&CelestialFrame::GCRF).is_err());
        assert!(data_type_for_state_frame(&CelestialFrame::MOD).is_err());
    }

    #[test]
    #[parallel]
    fn test_from_file_infers_frame_from_data_type() {
        let dir = tempfile::tempdir().unwrap();
        let text = std::fs::read_to_string(TRUNCATED).unwrap();

        let teme = dir
            .path()
            .join("TEME_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt");
        std::fs::write(&teme, &text).unwrap();
        let itc = ITC::from_file(&teme).unwrap();
        assert_eq!(itc.header.state_frame, CelestialFrame::TEME);
        assert_eq!(itc.source_name.as_ref().unwrap().data_type, "TEME");

        let itrf = dir
            .path()
            .join("ITRF_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt");
        std::fs::write(&itrf, &text).unwrap();
        assert_eq!(
            ITC::from_file(&itrf).unwrap().header.state_frame,
            CelestialFrame::ITRF
        );

        let unknown = dir
            .path()
            .join("GCRF_100002_STARLINK-37711_2540149_Operational__UNCLASSIFIED.txt");
        std::fs::write(&unknown, &text).unwrap();
        assert!(ITC::from_file(&unknown).is_err());

        let plain = dir.path().join("ephemeris.txt");
        std::fs::write(&plain, &text).unwrap();
        let itc = ITC::from_file(&plain).unwrap();
        assert_eq!(itc.header.state_frame, CelestialFrame::EME2000);
        assert!(itc.source_name.is_none());
    }

    #[test]
    #[parallel]
    fn test_itc_file_name() {
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let name = itc
            .file_name(
                100002,
                "STARLINK-37711",
                SpaceTrackEphemerisFileCategory::Operational,
                "1473385800",
            )
            .unwrap();
        assert_eq!(
            name.to_string(),
            "MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt"
        );

        let mut teme = ITC::new(ITCHeader::new().with_state_frame(CelestialFrame::TEME));
        teme.push_state(ITCStateVector::new(
            Epoch::from_datetime(2020, 10, 26, 12, 24, 0.0, 0.0, TimeSystem::UTC),
            [7.0e6, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ))
        .unwrap();
        let name = teme
            .file_name(25544, "ISS", SpaceTrackEphemerisFileCategory::Special, "")
            .unwrap();
        assert_eq!(
            name.to_string(),
            "TEME_25544_ISS_3001224_Special__UNCLASSIFIED.txt"
        );

        assert!(
            ITC::new(ITCHeader::new())
                .file_name(1, "A", SpaceTrackEphemerisFileCategory::Operational, "")
                .is_err()
        );
        let mut gcrf = ITC::new(ITCHeader::new().with_state_frame(CelestialFrame::GCRF));
        gcrf.push_state(ITCStateVector::new(
            Epoch::from_datetime(2020, 10, 26, 12, 24, 0.0, 0.0, TimeSystem::UTC),
            [7.0e6, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ))
        .unwrap();
        assert!(
            gcrf.file_name(1, "A", SpaceTrackEphemerisFileCategory::Operational, "")
                .is_err()
        );
    }
}
