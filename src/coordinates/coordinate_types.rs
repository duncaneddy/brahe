/*!
 * Defines types for coordinate systems.
 */

/// Defines the type of ellipsoidal conversion utilized in various coordinate transformations.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EllipsoidalConversionType {
    /// Geocentric coordinates: position defined by radius from Earth's center, latitude,
    /// and longitude. Uses spherical geometry, simpler but less accurate for surface positions.
    Geocentric,
    /// Geodetic coordinates: position defined by latitude/longitude on reference ellipsoid
    /// (WGS84) and height above ellipsoid. Standard for GPS and most Earth surface applications.
    Geodetic,
}
