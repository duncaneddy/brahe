/*!
The Space Weather module provides functions for loading and accessing space weather data.
This data is used for atmospheric density calculations and other space environment models.

Space weather data includes:
- Kp and Ap geomagnetic indices
- F10.7 solar radio flux
- International Sunspot Number

Data is typically sourced from CelesTrak's CSSI space weather files.
*/

mod caching_provider;
mod file_provider;
mod global;
mod iterator;
mod parser;
mod provider;
mod static_provider;
pub mod types;

pub use caching_provider::*;
pub use file_provider::*;
pub use global::*;
pub use iterator::*;
pub use parser::*;
pub use provider::*;
pub use static_provider::*;
pub use types::*;
