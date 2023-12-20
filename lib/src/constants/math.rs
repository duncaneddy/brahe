/*!
 Math constants.
*/

#![allow(dead_code)]

/// Constant to convert degrees to radians. Units: [rad/deg]
pub const DEG2RAD:f64 = std::f64::consts::PI/180.0;

/// Constant to convert radians to degrees. Units: [deg/rad]
pub const RAD2DEG:f64 = 180.0/std::f64::consts::PI;

/// Constant to convert arc seconds to radians. Units: [rad/as]
pub const AS2RAD:f64 = DEG2RAD / 3600.0;

/// Constant to convert radians to arc seconds. Units: [ad/rad]
pub const RAD2AS:f64 = RAD2DEG * 3600.0;