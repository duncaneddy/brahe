/*!
 * Global frame registry: parent chains and orientation providers for
 * registered [`ReferenceFrame`]s.
 *
 * This is the single table backing both [`register_frame`] (arbitrary
 * `Body` frames, e.g. spacecraft body/sensor chains) and
 * [`register_custom_frame`](super::register_custom_frame) (user-defined
 * `CelestialFrame::BodyFixedCustom` orientation models) — one registry, two
 * doorways. Every entry carries the [`OrientationProvider`] supplying its
 * rotation relative to `parent`; `parent` is `None` only for `Custom`
 * entries, whose parent (ICRF axes) and center are implied by the
 * `CelestialFrame::BodyFixedCustom` value at query time rather than stored
 * in the registry.
 */

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

use crate::frames::{BodyFrame, OrientationProvider, ReferenceFrame};
use crate::utils::BraheError;

/// Registry key covering both doorways into the frame registry.
///
/// `Custom(u32)` mirrors `CelestialFrame::BodyFixedCustom`'s `key` field
/// exactly: a registry handle, not a NAIF ID (the NAIF center is the
/// separate `center` field on `BodyFixedCustom`), so it stays `u32` rather
/// than conflating key-space with NAIF space. `Body(ObjectId, BodyFrame)` is
/// required because one object owns many frames (`SC_BODY`, `CSS_1`, ...),
/// so `ObjectId` alone cannot key the table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum FrameKey {
    /// A user-defined body-fixed frame registered under a caller-chosen
    /// handle (`register_custom_frame` / `CelestialFrame::BodyFixedCustom`).
    Custom(u32),
    /// A bound `Body` frame of an object.
    Body(crate::frames::ObjectId, BodyFrame),
}

/// A registered frame's parent link and orientation provider.
///
/// `parent` is `None` only for `Custom` entries; every entry inserted by
/// [`register_frame`] carries `Some`.
#[derive(Clone)]
pub(crate) struct FrameEntry {
    pub(crate) parent: Option<ReferenceFrame>,
    pub(crate) provider: Arc<dyn OrientationProvider>,
}

/// The global frame registry. Shared by [`register_frame`]/[`unregister_frame`]
/// and the `custom` module's `register_custom_frame`/`unregister_custom_frame`.
pub(crate) static FRAME_REGISTRY: Lazy<RwLock<HashMap<FrameKey, FrameEntry>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Converts a bound `Body` frame into its registry key. Returns `None` for
/// every other frame (celestial, orbit-relative, or an unbound body frame),
/// none of which can be registered under [`register_frame`].
fn frame_key(frame: &ReferenceFrame) -> Option<FrameKey> {
    match frame {
        ReferenceFrame::Body {
            frame,
            object: Some(object),
        } => Some(FrameKey::Body(object.clone(), frame.clone())),
        _ => None,
    }
}

/// Registers (or replaces) `frame`'s orientation relative to `parent`.
///
/// `frame` must be a bound `Body` frame (e.g. `ReferenceFrame::SC_BODY("SC")`,
/// `ReferenceFrame::CSS("SC", "1")`); `parent` must resolve to a celestial root by
/// walking the registry — either `parent` is itself `ReferenceFrame::Celestial`, or
/// it is a bound `Body` frame that is already registered and whose own
/// parent chain terminates at one. Re-registering an existing `frame`
/// replaces its entry; the new parent chain is revalidated, so replacing a
/// frame with a parent that would cycle back through `frame` itself is
/// rejected. Validation and insertion happen under a single write-lock
/// acquisition, so two concurrent calls cannot race to co-create a cycle
/// between each other.
///
/// # Arguments
/// * `frame` - The bound `Body` frame being registered
/// * `parent` - The frame `frame`'s orientation is expressed relative to
/// * `provider` - Supplies `frame`'s rotation (and optionally angular
///   velocity) relative to `parent`
///
/// # Returns
/// * `Ok(())`: If `frame` is bound and `parent`'s chain terminates at a
///   celestial frame without passing through `frame` itself
/// * `Err(BraheError)`: If `frame` is not a bound `Body` frame, if the
///   parent chain does not terminate at a celestial frame, or if it cycles
///   back through `frame`
///
/// # Examples
///
/// ```rust
/// use brahe::attitude::Quaternion;
/// use brahe::frames::{CelestialFrame, ReferenceFrame, clear_frame_registry, register_frame};
///
/// clear_frame_registry();
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// register_frame(ReferenceFrame::SC_BODY("SC"), CelestialFrame::GCRF.into(), q).unwrap();
/// register_frame(ReferenceFrame::CSS("SC", "1"), ReferenceFrame::SC_BODY("SC"), q).unwrap();
/// clear_frame_registry();
/// ```
pub fn register_frame(
    frame: ReferenceFrame,
    parent: ReferenceFrame,
    provider: impl OrientationProvider + 'static,
) -> Result<(), BraheError> {
    let key = frame_key(&frame).ok_or_else(|| {
        BraheError::Error(format!(
            "cannot register {}: target must be a bound Body frame (e.g. \
             ReferenceFrame::SC_BODY(\"SC\")); construct with a family constructor \
             or ReferenceFrame::body(object, ..) and bind an object",
            frame
        ))
    })?;

    let mut guard = FRAME_REGISTRY.write().unwrap();
    validate_parent_chain(&frame, &parent, &guard)?;
    guard.insert(
        key,
        FrameEntry {
            parent: Some(parent),
            provider: Arc::new(provider),
        },
    );
    Ok(())
}

/// Walks `parent`'s chain through the registry, requiring it to terminate
/// at `ReferenceFrame::Celestial` without passing through `frame` itself.
fn validate_parent_chain(
    frame: &ReferenceFrame,
    parent: &ReferenceFrame,
    map: &HashMap<FrameKey, FrameEntry>,
) -> Result<(), BraheError> {
    let mut current = parent.clone();
    loop {
        if matches!(current, ReferenceFrame::Celestial(_)) {
            return Ok(());
        }
        if &current == frame {
            return Err(BraheError::Error(format!(
                "cannot register {}: parent chain through {} cycles back to \
                 {} itself",
                frame, parent, frame
            )));
        }
        let key = frame_key(&current).ok_or_else(|| missing_parent_error(frame, &current))?;
        match map.get(&key) {
            Some(entry) => {
                current = entry
                    .parent
                    .clone()
                    .expect("registered Body frame entries always carry a parent");
            }
            None => return Err(missing_parent_error(frame, &current)),
        }
    }
}

/// Builds the D14-style error for a parent link with no registered
/// orientation: names the missing link and the call that fixes it.
fn missing_parent_error(frame: &ReferenceFrame, parent: &ReferenceFrame) -> BraheError {
    BraheError::Error(format!(
        "cannot register {}: parent {} has no registered orientation; \
         register it first with register_frame({}, <parent>, <provider>)",
        frame, parent, parent
    ))
}

/// Removes the registered orientation of a bound `Body` frame.
///
/// # Arguments
/// * `frame` - The bound `Body` frame to unregister
///
/// # Returns
/// `bool`: `true` if `frame` was registered and has been removed
///
/// # Examples
///
/// ```rust
/// use brahe::attitude::Quaternion;
/// use brahe::frames::{CelestialFrame, ReferenceFrame, clear_frame_registry, register_frame, unregister_frame};
///
/// clear_frame_registry();
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// register_frame(ReferenceFrame::SC_BODY("SC"), CelestialFrame::GCRF.into(), q).unwrap();
/// assert!(unregister_frame(&ReferenceFrame::SC_BODY("SC")));
/// assert!(!unregister_frame(&ReferenceFrame::SC_BODY("SC")));
/// ```
pub fn unregister_frame(frame: &ReferenceFrame) -> bool {
    match frame_key(frame) {
        Some(key) => FRAME_REGISTRY.write().unwrap().remove(&key).is_some(),
        None => false,
    }
}

/// Removes every entry from the frame registry, including `Custom` entries
/// registered through `register_custom_frame`.
///
/// Intended for test isolation, matching the crate's other global-registry
/// lifecycle functions.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::clear_frame_registry;
///
/// clear_frame_registry();
/// ```
pub fn clear_frame_registry() {
    FRAME_REGISTRY.write().unwrap().clear();
}

/// Looks up the entry for `key`, cloning the `Arc` provider out of the
/// lock. The lock is released before returning, so no provider evaluation
/// ever runs while it is held.
///
/// # Arguments
/// * `key` - The registry key to look up
///
/// # Returns
/// `Option<FrameEntry>`: The entry, or `None` if `key` is not registered
pub(crate) fn frame_entry(key: &FrameKey) -> Option<FrameEntry> {
    FRAME_REGISTRY.read().unwrap().get(key).cloned()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::attitude::Quaternion;
    use crate::frames::CelestialFrame;

    #[test]
    #[serial]
    fn test_register_frame_validation() {
        clear_frame_registry();
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        // Unbound frame rejected
        let unbound: ReferenceFrame = BodyFrame::SCBody(None).into();
        assert!(register_frame(unbound, CelestialFrame::GCRF.into(), q).is_err());
        // Celestial target rejected
        assert!(
            register_frame(CelestialFrame::ITRF.into(), CelestialFrame::GCRF.into(), q).is_err()
        );
        // Parent chain must exist: CSS -> SC_BODY fails before SC_BODY registered
        let err = register_frame(
            ReferenceFrame::CSS("SC", "1"),
            ReferenceFrame::SC_BODY("SC"),
            q,
        )
        .unwrap_err();
        assert!(err.to_string().contains("SC_BODY@SC"));
        // Valid chain
        register_frame(
            ReferenceFrame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            q,
        )
        .unwrap();
        register_frame(
            ReferenceFrame::CSS("SC", "1"),
            ReferenceFrame::SC_BODY("SC"),
            q,
        )
        .unwrap();
        // Replace that would self-cycle rejected: SC_BODY reparented onto CSS
        assert!(
            register_frame(
                ReferenceFrame::SC_BODY("SC"),
                ReferenceFrame::CSS("SC", "1"),
                q
            )
            .is_err()
        );
        assert!(unregister_frame(&ReferenceFrame::CSS("SC", "1")));
        assert!(!unregister_frame(&ReferenceFrame::CSS("SC", "1")));
        // A frame with no registry key (celestial, or unbound body) is
        // never registered, so unregistering one is always a no-op.
        assert!(!unregister_frame(&CelestialFrame::GCRF.into()));
        let unbound: ReferenceFrame = BodyFrame::SCBody(None).into();
        assert!(!unregister_frame(&unbound));
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_frame_entry_clones_provider_and_clear_wipes_table() {
        clear_frame_registry();
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        register_frame(
            ReferenceFrame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            q,
        )
        .unwrap();

        let key = FrameKey::Body(crate::frames::ObjectId::from("SC"), BodyFrame::SCBody(None));
        let entry = frame_entry(&key).unwrap();
        assert_eq!(entry.parent, Some(CelestialFrame::GCRF.into()));

        clear_frame_registry();
        assert!(frame_entry(&key).is_none());
    }

    #[test]
    #[serial]
    fn test_register_frame_unbound_parent_rejected() {
        clear_frame_registry();
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        // An unbound frame can never satisfy the chain walk: it is not
        // celestial and has no registry key of its own.
        let unbound_parent: ReferenceFrame = BodyFrame::SCBody(None).into();
        assert!(register_frame(ReferenceFrame::CSS("SC", "1"), unbound_parent, q).is_err());
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_register_frame_replace_revalidates_chain() {
        clear_frame_registry();
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        register_frame(
            ReferenceFrame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            q,
        )
        .unwrap();
        // Replace with a still-valid parent chain: succeeds.
        register_frame(
            ReferenceFrame::SC_BODY("SC"),
            CelestialFrame::ITRF.into(),
            q,
        )
        .unwrap();
        let key = FrameKey::Body(crate::frames::ObjectId::from("SC"), BodyFrame::SCBody(None));
        assert_eq!(
            frame_entry(&key).unwrap().parent,
            Some(CelestialFrame::ITRF.into())
        );
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_register_frame_rejects_three_node_cycle_under_single_lock() {
        // Validation and insertion now happen under one write-lock
        // acquisition (see register_frame's doc comment); this exercises a
        // longer chain (SC_BODY -> CSS -> AST) than the two-node self-cycle
        // case above to confirm the walk still rejects a cycle correctly
        // through the combined lock/validate/insert path.
        clear_frame_registry();
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        register_frame(
            ReferenceFrame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            q,
        )
        .unwrap();
        register_frame(
            ReferenceFrame::CSS("SC", "1"),
            ReferenceFrame::SC_BODY("SC"),
            q,
        )
        .unwrap();
        register_frame(
            ReferenceFrame::AST("SC", "1"),
            ReferenceFrame::CSS("SC", "1"),
            q,
        )
        .unwrap();
        // Reparenting SC_BODY onto AST would cycle SC_BODY -> AST -> CSS -> SC_BODY.
        let err = register_frame(
            ReferenceFrame::SC_BODY("SC"),
            ReferenceFrame::AST("SC", "1"),
            q,
        )
        .unwrap_err();
        assert!(err.to_string().contains("cycles back"));
        // The original entry must be untouched: the rejected registration
        // never reached the insert step.
        let key = FrameKey::Body(crate::frames::ObjectId::from("SC"), BodyFrame::SCBody(None));
        assert_eq!(
            frame_entry(&key).unwrap().parent,
            Some(CelestialFrame::GCRF.into())
        );
        clear_frame_registry();
    }
}
