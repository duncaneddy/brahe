/*!
Magnetic field models for computing Earth's geomagnetic field.

This module provides implementations of two geomagnetic field models:

- **IGRF-14** (International Geomagnetic Reference Field, 14th generation):
  Spherical harmonic expansion to degree 13, covering 1900-2030 with
  5-year epoch intervals.

- **WMMHR-2025** (World Magnetic Model High Resolution):
  Spherical harmonic expansion to degree 133, at epoch 2025.0 with
  secular variation. Provides much higher spatial resolution than IGRF
  by including crustal field contributions.

## Output Frames

Each model provides three output frame options:

- `*_geodetic_enz`: Field in the geodetic ENZ frame (East, North, Zenith where
  Zenith is perpendicular to the WGS84 ellipsoid)
- `*_geocentric_enz`: Field in the geocentric ENZ frame (East, North, Zenith where
  Zenith is along the geocentric radial direction)
- `*_ecef`: Field in the ECEF (Earth-Centered Earth-Fixed) frame

## Input

All functions accept geodetic coordinates as `(longitude, latitude, altitude)`
with `AngleFormat` controlling whether angles are in degrees or radians.
Altitude is always in meters (SI).

## Units

All magnetic field outputs are in **nanoTesla (nT)**.
*/

pub use igrf::*;
pub use wmmhr::*;

pub(crate) mod data;
mod igrf;
pub(crate) mod spherical_harmonics;
pub(crate) mod transforms;
mod wmmhr;
