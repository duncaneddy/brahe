/*!
 * This is where kernel file enums live. Currently DE only, will scale up to other kernel files.
 */

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

impl From<crate::propagators::force_model_config::EphemerisSource> for SpkKernel {
    fn from(source: crate::propagators::force_model_config::EphemerisSource) -> Self {
        use crate::propagators::force_model_config::EphemerisSource;
        match source {
            EphemerisSource::DE440s => SpkKernel::DE440s,
            EphemerisSource::DE440 => SpkKernel::DE440,
            EphemerisSource::LowPrecision => {
                panic!("LowPrecision is not a valid DE kernel — use SpkKernel::DE440s or DE440")
            }
        }
    }
}
