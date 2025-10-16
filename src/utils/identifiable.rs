/*!
 * Identifiable trait for objects that can have identifying information
 *
 * This module provides a common interface for structs that can be identified
 * using a combination of name, numeric ID, and UUID. All identification fields
 * are optional, allowing flexible usage across different contexts.
 */

use uuid::Uuid;

/// Trait for objects that can be identified by name, ID, and/or UUID.
///
/// All identification fields are optional, allowing flexible usage:
/// - `name`: Human-readable string identifier
/// - `id`: Numeric identifier (non-negative integer)
/// - `uuid`: Universally unique identifier
///
/// This trait provides builder-style methods for setting identity values
/// and standard getters for accessing them.
pub trait Identifiable {
    /// Set the name and return self (consuming constructor pattern)
    fn with_name(self, name: &str) -> Self;

    /// Set the UUID and return self (consuming constructor pattern)
    fn with_uuid(self, uuid: Uuid) -> Self;

    /// Generate a new UUID, set it, and return self (consuming constructor pattern)
    fn with_new_uuid(self) -> Self;

    /// Set the numeric ID and return self (consuming constructor pattern)
    fn with_id(self, id: u64) -> Self;

    /// Set all identity fields at once and return self (consuming constructor pattern)
    fn with_identity(self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) -> Self;

    /// Set all identity fields in-place (mutating)
    fn set_identity(&mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>);

    /// Set the numeric ID in-place (mutating)
    fn set_id(&mut self, id: Option<u64>);

    /// Set the name in-place (mutating)
    fn set_name(&mut self, name: Option<&str>);

    /// Generate a new UUID and set it in-place (mutating)
    fn generate_uuid(&mut self);

    /// Get the current numeric ID
    fn get_id(&self) -> Option<u64>;

    /// Get the current name
    fn get_name(&self) -> Option<&str>;

    /// Get the current UUID
    fn get_uuid(&self) -> Option<Uuid>;
}
