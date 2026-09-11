/*!
 * Modified ITC ephemeris format.
 *
 * The Modified ITC format is one of the ephemeris formats accepted by
 * Space-Track for conjunction screening (Spaceflight Safety Handbook for
 * Operators, "Modified ITC Ephemeris Format"). Starlink publishes its public
 * ephemerides in it. A file holds four header lines, the fourth naming the
 * covariance frame, followed by one record per epoch: a line with the epoch
 * (`YYYYDDDHHMMSS.sss`, UTC), position (km) and velocity (km/s), then three
 * lines carrying the 21 lower-triangular elements of the 6x6 position and
 * velocity covariance.
 *
 * [`ITC`] holds the parsed message in SI units. [`ITC::from_file`] also
 * decodes the Space-Track file name to recover the state frame and object
 * identity, and [`ITC::file_name`] produces a compliant name for writing.
 */

mod frames;
mod parse;
mod types;
mod write;

pub use frames::{data_type_for_state_frame, state_frame_for_data_type};
pub use types::{ITC, ITCCovarianceFrame, ITCHeader, ITCStateVector};
