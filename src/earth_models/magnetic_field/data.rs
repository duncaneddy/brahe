/// IGRF-14 spherical harmonic coefficient file (SHC format)
///
/// Contains Gauss coefficients for degrees 1-13 at 27 five-year epochs from 1900.0 to 2030.0.
/// International Geomagnetic Reference Field, 14th generation.
/// See <https://doi.org/10.5281/zenodo.14012302>
pub(crate) static IGRF14_SHC: &str = include_str!("../../../data/magnetic_field/IGRF14.shc");

/// WMMHR-2025 spherical harmonic coefficient file (COF format)
///
/// Contains Gauss coefficients for degrees 1-133 at epoch 2025.0
/// with secular variation rates for the core field (degrees 1-15).
/// World Magnetic Model High Resolution, 2025 edition.
pub(crate) static WMMHR_COF: &str = include_str!("../../../data/magnetic_field/WMMHR.COF");
