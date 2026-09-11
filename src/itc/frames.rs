/*!
 * Mapping between Space-Track file-name DataType tokens and state frames.
 */

use crate::frames::CelestialFrame;
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
