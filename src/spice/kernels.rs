/*!
 * This is where kernel file enums live. Currently DE only, will scale up to other kernel files.
 */

/// Selects which JPL DE SPK kernel file to load.
///
/// This is different from `EphemerisSource`, which is a force-model concept.
/// `SPKKernel` is purely "which kernel file to open".
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub enum SPKKernel {
    /// JPL DE440s - covers 1849-2150 CE, 32MB download.
    DE440s,
    /// JPL DE440 - covers 1550-2650 CE, 115MB download.
    DE440,
}
