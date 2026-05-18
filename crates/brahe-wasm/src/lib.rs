//! WASM bindings for the brahe astrodynamics library.
//!
//! Each brahe submodule has a corresponding binding module here (e.g. `constants`).
//! Bindings expose Rust constants and functions as `#[wasm_bindgen]` getters with
//! `__UPPER_SNAKE_CASE` JS names. A hand-written TypeScript wrapper layer (in `js/`)
//! re-exports them as idiomatic TS named exports.

pub mod constants;
