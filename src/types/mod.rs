/*!
 * Shared types used by both SpaceTrack and Celestrak modules.
 *
 * Contains the canonical [`GPRecord`] type for General Perturbations data,
 * flexible serde deserializers for mixed string/number JSON, and the
 * [`FieldAccessor`] trait for field-name-based record access.
 */

pub mod gp_record;
pub mod serde_flex;

pub use gp_record::GPRecord;
