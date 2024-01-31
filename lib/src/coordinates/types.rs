/*!
 * Defines types for coordinate systems.
 */

/// Defines the type of ellipsoidal conversion utilized in various coordinate transformations.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EllipsoidalConversionType {
    Geocentric,
    Geodetic,
}
