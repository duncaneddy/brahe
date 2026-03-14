/*!
 * This is where kernel file enums live. Currently DE only, will scale up to other kernel files.
 */

use crate::propagators::force_model_config::EphemerisSource;
use crate::utils::BraheError;

/// Selects which JPL DE SPK kernel file to load.
///
/// This is different from `EphemerisSource` in `crate::propagators::force_model_config`,
/// which is a force-model concept. `SPKKernel` is purely "which kernel file to open".
#[allow(non_camel_case_types)]
pub enum SPKKernel {
    /// JPL DE440s - covers 1849-2150 CE, 32MB download.
    DE440s,
    /// JPL DE440 - covers 1550-2650 CE, 115MB download.
    DE440,
}

impl TryFrom<EphemerisSource> for SPKKernel {
    type Error = BraheError;

    fn try_from(source: EphemerisSource) -> Result<Self, Self::Error> {
        match source {
            EphemerisSource::DE440s => Ok(SPKKernel::DE440s),
            EphemerisSource::DE440 => Ok(SPKKernel::DE440),
            EphemerisSource::LowPrecision => Err(BraheError::Error(
                "LowPrecision is not a valid DE kernel - use SPKKernel::DE440s or DE440"
                    .to_string(),
            )),
        }
    }
}
