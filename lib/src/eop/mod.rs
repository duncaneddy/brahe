/*!
The Earth Orientation Parameters (EOP) module provides a set of functions for
loading and accessing EOP data. This data is used as part of time and reference
frame transformations.
*/

mod c04_parser;
mod eop_provider;
mod file_provider;
mod standard_parser;
mod static_provider;
pub mod types;

pub use eop_provider::*;
pub use file_provider::*;
pub use static_provider::*;
pub use types::*;
