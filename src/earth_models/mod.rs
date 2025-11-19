/*!
Module providing Earth environment models for orbital dynamics calculations.

This module contains implementations of various Earth environmental models used in
orbital dynamics computations, including atmospheric density models and (future)
magnetic field models.

## Atmospheric Density Models

- [`harris_priester`]: Modified Harris-Priester atmospheric density model
- `nrlmsise00`: NRLMSISE-00 atmospheric model (planned)

## Usage

```rust
use brahe::earth_models::density_harris_priester;
use nalgebra::Vector3;

let r_sat = Vector3::new(6778137.0, 0.0, 0.0);  // ~400 km altitude
let r_sun = Vector3::new(1.5e11, 0.0, 0.0);     // Approximate sun position
let density = density_harris_priester(r_sat, r_sun);
```
 */

pub use harris_priester::*;
pub use nrlmsise00::*;

pub mod harris_priester;
#[allow(clippy::assign_op_pattern)]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::type_complexity)]
pub mod nrlmsise00;
pub mod nrlmsise00_data;
