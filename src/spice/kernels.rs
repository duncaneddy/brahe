/*!
 * This is where kernel file enums live. Currently DE only, will scale up to other kernel files.
 */

use crate::propagators::force_model_config::EphemerisSource;
use crate::utils::BraheError;

/// Selects which JPL DE SPK kernel file to load.
///
/// This is different from `EphemerisSource` in `crate::propagators::force_model_config`,
/// which is a force-model concept. `SpkKernel` is purely "which kernel file to open".
pub enum SpkKernel {
    /// JPL DE440s - covers 1849-2150 CE, 32MB download.
    DE440s,
    /// JPL DE440 - covers 1550-2650 CE, 115MB download.
    DE440,
}

impl TryFrom<EphemerisSource> for SpkKernel {
    type Error = BraheError;

    fn try_from(source: EphemerisSource) -> Result<Self, Self::Error> {
        match source {
            EphemerisSource::DE440s => Ok(SpkKernel::DE440s),
            EphemerisSource::DE440 => Ok(SpkKernel::DE440),
            EphemerisSource::LowPrecision => Err(BraheError::Error(
                "LowPrecision is not a valid DE kernel - use SpkKernel::DE440s or DE440"
                    .to_string(),
            )),
        }
    }
}
