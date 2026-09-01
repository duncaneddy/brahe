/*!
 * Global object registry: named ephemeris providers for translation
 * queries.
 *
 * Objects are identified by [`ObjectId`] (D7), a string-backed name in a
 * space kept separate from NAIF IDs: no self-assigned integers, no
 * ambiguity with NORAD/NAIF/asteroid numbering. Kernel data enters the
 * object space only through the explicit [`SPKStateProvider`] door
 * (`register_object_from_naif`) — there is no implicit path from a NAIF ID
 * to an object.
 *
 * The registry is a single global table (D2), matching the crate's other
 * process-wide providers (EOP, SPICE kernels, the frame registry):
 * [`ObjectId`] maps to a provider and the [`CelestialFrame`] its states are
 * expressed in. Readers clone the `Arc<dyn SStateProvider>` out of the lock
 * and evaluate it after releasing the lock (D11), so no provider ever runs
 * while the registry is held.
 */

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use nalgebra::Vector6;
use once_cell::sync::Lazy;

use crate::frames::{CelestialFrame, ObjectId};
use crate::math::SVector6;
use crate::spice::NAIFId;
use crate::spice::registry::ensure_bodies_loadable;
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::utils::state_providers::{DStateProvider, SStateProvider};

/// A registered object's state provider and the celestial frame its states
/// are expressed in.
#[derive(Clone)]
struct ObjectEntry {
    provider: Arc<dyn SStateProvider + Send + Sync>,
    frame: CelestialFrame,
}

/// The global object registry. Shared by [`register_object`] /
/// [`unregister_object`] / [`object_state`].
static OBJECT_REGISTRY: Lazy<RwLock<HashMap<ObjectId, ObjectEntry>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Registers (or replaces) `name`'s state provider.
///
/// Re-registering an existing `name` replaces its entry, matching the
/// frame registry's register-or-replace precedent (updating an object with
/// a fresher provider is a normal operation).
///
/// # Arguments
/// * `name` - The object's identity (e.g. `"LRO"`, `"2024-123A"`)
/// * `provider` - Supplies `name`'s state at arbitrary epochs, in `frame`
/// * `frame` - The celestial frame `provider`'s states are expressed in
///
/// # Returns
/// * `Ok(())`: Always, on success
///
/// # Examples
///
/// ```rust
/// use brahe::attitude::Quaternion;
/// use brahe::constants::R_EARTH;
/// use brahe::frames::{CelestialFrame, clear_object_registry, register_object};
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::utils::state_providers::SStateProvider;
/// use brahe::utils::BraheError;
/// use nalgebra::Vector6;
///
/// struct ConstantProvider(Vector6<f64>);
/// impl SStateProvider for ConstantProvider {
///     fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
///         Ok(self.0)
///     }
/// }
///
/// clear_object_registry();
/// let x = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0);
/// register_object("SC", ConstantProvider(x), CelestialFrame::GCRF).unwrap();
/// clear_object_registry();
/// # let _ = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// ```
pub fn register_object(
    name: impl Into<ObjectId>,
    provider: impl SStateProvider + Send + Sync + 'static,
    frame: CelestialFrame,
) -> Result<(), BraheError> {
    OBJECT_REGISTRY.write().unwrap().insert(
        name.into(),
        ObjectEntry {
            provider: Arc::new(provider),
            frame,
        },
    );
    Ok(())
}

/// Removes the registered provider for `name`.
///
/// # Arguments
/// * `name` - The object's identity
///
/// # Returns
/// `bool`: `true` if `name` was registered and has been removed
pub fn unregister_object(name: &ObjectId) -> bool {
    OBJECT_REGISTRY.write().unwrap().remove(name).is_some()
}

/// Removes every entry from the object registry.
///
/// Intended for test isolation, matching the crate's other global-registry
/// lifecycle functions.
pub fn clear_object_registry() {
    OBJECT_REGISTRY.write().unwrap().clear();
}

/// Names of every registered object, sorted for stable error messages.
///
/// # Returns
/// `Vec<ObjectId>`: Registered object names, sorted lexicographically by
/// their string representation
pub fn registered_objects() -> Vec<ObjectId> {
    let mut names: Vec<ObjectId> = OBJECT_REGISTRY.read().unwrap().keys().cloned().collect();
    names.sort_by_key(|id| id.to_string());
    names
}

/// Builds the D14-style error for a query against an unregistered object:
/// names the missing object, the currently registered ones, and the calls
/// that fix it.
fn unknown_object_error(name: &ObjectId) -> BraheError {
    let registered = registered_objects()
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    BraheError::Error(format!(
        "object '{name}' is not registered (registered objects: {registered}); \
         register it with register_object(\"{name}\", <state provider>, <frame>) \
         or oem.register_for(\"{name}\")"
    ))
}

/// Looks up `name`'s state at `epoch`.
///
/// Clones the entry's `Arc<dyn SStateProvider>` out of the registry lock
/// and evaluates it after releasing the lock (D11), so no provider ever
/// runs while the registry is held.
///
/// # Arguments
/// * `name` - The object's identity
/// * `epoch` - The epoch at which to compute the state
///
/// # Returns
/// * `Ok((CelestialFrame, SVector6))`: `name`'s state and the celestial
///   frame it is expressed in
/// * `Err(BraheError)`: If `name` is not registered, or the provider fails
///   to compute a state at `epoch`
pub(crate) fn object_state(
    name: &ObjectId,
    epoch: Epoch,
) -> Result<(CelestialFrame, SVector6), BraheError> {
    let entry = OBJECT_REGISTRY
        .read()
        .unwrap()
        .get(name)
        .cloned()
        .ok_or_else(|| unknown_object_error(name))?;
    let state = entry.provider.state(epoch)?;
    Ok((entry.frame, state))
}

/// Adapts a dynamic-sized [`DStateProvider`] into [`SStateProvider`], for
/// providers whose native state is a 6-dimensional `DVector` (e.g.
/// `DOrbitTrajectory`) to register as objects.
pub struct DStateAdapter {
    provider: Box<dyn DStateProvider + Send + Sync>,
}

impl DStateAdapter {
    /// Wraps `provider`, requiring its state dimension to be exactly 6.
    ///
    /// # Arguments
    /// * `provider` - The dynamic-sized state provider to adapt
    ///
    /// # Returns
    /// * `Ok(DStateAdapter)`: If `provider.state_dim() == 6`
    /// * `Err(BraheError)`: If `provider.state_dim() != 6`
    pub fn new(provider: impl DStateProvider + Send + Sync + 'static) -> Result<Self, BraheError> {
        let dim = provider.state_dim();
        if dim != 6 {
            return Err(BraheError::Error(format!(
                "DStateAdapter requires a 6-dimensional state provider, got dimension {dim}"
            )));
        }
        Ok(Self {
            provider: Box::new(provider),
        })
    }
}

impl SStateProvider for DStateAdapter {
    fn state(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let state = self.provider.state(epoch)?;
        Ok(Vector6::from_column_slice(state.as_slice()))
    }
}

/// State provider backed by a loaded SPICE kernel, querying a body's state
/// relative to Earth (NAIF 399) in ICRF axes.
///
/// This is the spec-D7 "door": kernel data enters the object registry only
/// through explicit registration of a `SPKStateProvider` (or the
/// [`register_object_from_naif`] convenience), never implicitly.
pub struct SPKStateProvider {
    naif_id: i32,
}

impl SPKStateProvider {
    /// Creates a provider for the body identified by `naif_id`.
    ///
    /// # Arguments
    /// * `naif_id` - NAIF ID of the body to query
    ///
    /// # Returns
    /// `SPKStateProvider`: A provider querying `naif_id`'s state relative
    /// to Earth from the loaded SPICE kernels
    pub fn new(naif_id: i32) -> Self {
        Self { naif_id }
    }
}

impl SStateProvider for SPKStateProvider {
    fn state(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let earth = NAIFId::Earth.id();
        ensure_bodies_loadable(&[self.naif_id, earth])?;
        crate::spice::spk_state(self.naif_id, NAIFId::Earth, epoch)
    }
}

/// Registers `name` as an `SPKStateProvider` for `naif_id`, in the GCRF
/// frame.
///
/// Convenience wrapper equivalent to
/// `register_object(name, SPKStateProvider::new(naif_id), CelestialFrame::GCRF)`.
///
/// # Arguments
/// * `name` - The object's identity to register
/// * `naif_id` - NAIF ID of the body to query from loaded SPICE kernels
///
/// # Returns
/// * `Ok(())`: Always, on success
///
/// # Examples
///
/// ```rust,no_run
/// use brahe::frames::{clear_object_registry, register_object_from_naif};
///
/// clear_object_registry();
/// register_object_from_naif("MOON", 301).unwrap();
/// clear_object_registry();
/// ```
pub fn register_object_from_naif(
    name: impl Into<ObjectId>,
    naif_id: i32,
) -> Result<(), BraheError> {
    register_object(name, SPKStateProvider::new(naif_id), CelestialFrame::GCRF)
}

/// Closure-backed [`SStateProvider`] for tests: wraps a `Fn(Epoch) ->
/// Result<Vector6<f64>, BraheError>` closure, avoiding a dedicated struct
/// per test fixture.
#[cfg(test)]
pub(crate) struct FnProvider<F: Fn(Epoch) -> Result<Vector6<f64>, BraheError> + Send + Sync>(
    pub(crate) F,
);

#[cfg(test)]
impl<F: Fn(Epoch) -> Result<Vector6<f64>, BraheError> + Send + Sync> SStateProvider
    for FnProvider<F>
{
    fn state(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        (self.0)(epoch)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::constants::R_EARTH;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_spice;

    #[test]
    #[serial]
    fn test_object_registry_round_trip_and_errors() {
        clear_object_registry();
        let x = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0);
        register_object("SC", FnProvider(move |_| Ok(x)), CelestialFrame::GCRF).unwrap();
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let (frame, state) = object_state(&"SC".into(), epc).unwrap();
        assert_eq!(frame, CelestialFrame::GCRF);
        assert_eq!(state, x);
        let err = object_state(&"B".into(), epc).unwrap_err().to_string();
        assert!(err.contains("object 'B' is not registered"));
        assert!(err.contains("registered objects: SC"));
        assert!(err.contains("register_object"));
        assert!(unregister_object(&"SC".into()));
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_unregister_object_missing_returns_false() {
        clear_object_registry();
        assert!(!unregister_object(&"NOPE".into()));
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_registered_objects_sorted() {
        clear_object_registry();
        let x = Vector6::zeros();
        register_object("ZULU", FnProvider(move |_| Ok(x)), CelestialFrame::GCRF).unwrap();
        register_object("ALFA", FnProvider(move |_| Ok(x)), CelestialFrame::GCRF).unwrap();
        let names: Vec<String> = registered_objects()
            .iter()
            .map(|id| id.to_string())
            .collect();
        assert_eq!(names, vec!["ALFA".to_string(), "ZULU".to_string()]);
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_register_object_replace_semantics() {
        clear_object_registry();
        let x1 = Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let x2 = Vector6::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        register_object("SC", FnProvider(move |_| Ok(x1)), CelestialFrame::GCRF).unwrap();
        register_object("SC", FnProvider(move |_| Ok(x2)), CelestialFrame::ITRF).unwrap();
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let (frame, state) = object_state(&"SC".into(), epc).unwrap();
        assert_eq!(frame, CelestialFrame::ITRF);
        assert_eq!(state, x2);
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_dstate_adapter_wraps_dynamic_provider() {
        use crate::utils::state_providers::DStateProvider;
        use nalgebra::DVector;

        struct SixDProvider;
        impl DStateProvider for SixDProvider {
            fn state(&self, _epoch: Epoch) -> Result<DVector<f64>, BraheError> {
                Ok(DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            }
            fn state_dim(&self) -> usize {
                6
            }
        }

        let adapter = DStateAdapter::new(SixDProvider).unwrap();
        let epc = Epoch::from_date(2024, 1, 1, TimeSystem::TAI);
        let state = adapter.state(epc).unwrap();
        assert_eq!(state, Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    }

    #[test]
    #[serial]
    fn test_dstate_adapter_rejects_wrong_dimension() {
        use crate::utils::state_providers::DStateProvider;
        use nalgebra::DVector;

        struct FourDProvider;
        impl DStateProvider for FourDProvider {
            fn state(&self, _epoch: Epoch) -> Result<DVector<f64>, BraheError> {
                Ok(DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]))
            }
            fn state_dim(&self) -> usize {
                4
            }
        }

        let err = DStateAdapter::new(FourDProvider).err().unwrap().to_string();
        assert!(err.contains("6-dimensional"));
    }

    #[test]
    #[serial]
    fn test_register_object_from_naif_moon_matches_direct_spice_query() {
        setup_global_test_spice();
        clear_object_registry();
        register_object_from_naif("MOON", 301).unwrap();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let (frame, state) = object_state(&"MOON".into(), epc).unwrap();
        assert_eq!(frame, CelestialFrame::GCRF);
        let direct = crate::spice::spk_state(301, NAIFId::Earth, epc).unwrap();
        for i in 0..6 {
            assert_eq!(state[i], direct[i]);
        }
        clear_object_registry();
    }
}
