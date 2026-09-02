/*!
 * State and covariance provider traits for accessing state vectors and uncertainties
 *
 * This module defines a two-tier trait hierarchy for state and covariance access:
 *
 * **Base Traits** (frame-agnostic):
 * - [`SStateProvider`] - Provides basic state access for static-sized (6D) vectors
 * - [`DStateProvider`] - Provides basic state access for dynamic-sized vectors
 * - [`SCovarianceProvider`] - Provides basic covariance access for static-sized matrices
 * - [`DCovarianceProvider`] - Provides basic covariance access for dynamic-sized matrices
 *
 * **Orbit-Specific Traits** (frame-aware):
 * - [`SOrbitStateProvider`] - Extends `SStateProvider` with orbital frame conversions
 * - [`DOrbitStateProvider`] - Extends `DStateProvider` with orbital frame conversions
 * - [`SOrbitCovarianceProvider`] - Extends `SCovarianceProvider` with frame conversions
 * - [`DOrbitCovarianceProvider`] - Extends `DCovarianceProvider` with frame conversions
 *
 * This separation allows:
 * - Non-orbital state providers to implement only base traits
 * - Clear distinction between basic state access and orbital-specific operations
 * - Better code reusability across different trajectory types
 *
 * Each provider family lives in its own file; every trait is re-exported
 * here, so `crate::utils::state_providers::X` resolves as before.
 */

mod attitude;
mod combined;
mod covariance;
mod orbit_covariance;
mod orbit_state;
mod state;

pub use combined::{DIdentifiableStateProvider, SIdentifiableStateProvider, ToPropagatorRefs};
pub use covariance::{DCovarianceProvider, SCovarianceProvider};
pub use orbit_covariance::{DOrbitCovarianceProvider, SOrbitCovarianceProvider};
pub use orbit_state::{DOrbitStateProvider, SOrbitStateProvider};
pub use state::{DStateProvider, SStateProvider};
