/*!
The Earth Orientation Parameters (EOP) module provides a set of functions for
loading and accessing EOP data. This data is used as part of time and reference
frame transformations.
*/

mod c04_parser;
pub mod download;
mod eop_provider;
pub mod eop_types;
mod file_provider;
mod global;
mod standard_parser;
mod static_provider;

pub use download::*;
pub use eop_provider::*;
pub use eop_types::*;
pub use file_provider::*;
pub use global::*;
pub use static_provider::*;
