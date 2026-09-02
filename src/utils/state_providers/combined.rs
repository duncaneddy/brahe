/*!
 * Traits combining state provision with object identity.
 *
 * [`SIdentifiableStateProvider`] and [`DIdentifiableStateProvider`] pair an
 * orbital state provider with [`Identifiable`], and [`ToPropagatorRefs`]
 * normalizes the ways a caller can hand a set of such providers to an API.
 */

use crate::utils::identifiable::Identifiable;

use super::orbit_state::{DOrbitStateProvider, SOrbitStateProvider};

/// Combined trait for static-sized state providers with identity tracking.
///
/// This supertrait combines `SOrbitStateProvider` and `Identifiable`, used primarily
/// in access computation where satellite identity needs to be tracked alongside
/// orbital state computation.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for any type that implements both
/// `SOrbitStateProvider` and `Identifiable` via a blanket implementation.
///
/// See also: [`DIdentifiableStateProvider`] for dynamic-sized version
///
/// # Examples
///
/// ```
/// use brahe::propagators::{KeplerianPropagator, SGPPropagator};
/// use brahe::utils::state_providers::SIdentifiableStateProvider;
///
/// // Both propagators implement SIdentifiableStateProvider automatically
/// fn accepts_identified_provider<P: SIdentifiableStateProvider>(provider: &P) {
///     // Can use both SOrbitStateProvider and Identifiable methods
/// }
/// ```
pub trait SIdentifiableStateProvider: SOrbitStateProvider + Identifiable {}

// Blanket implementation for any type implementing both traits
impl<T: SOrbitStateProvider + Identifiable> SIdentifiableStateProvider for T {}

/// Combined trait for dynamic-sized state providers with identity tracking.
///
/// This supertrait combines `DOrbitStateProvider` and `Identifiable`, used primarily
/// in access computation where satellite identity needs to be tracked alongside
/// orbital state computation.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for any type that implements both
/// `DOrbitStateProvider` and `Identifiable` via a blanket implementation.
///
/// See also: [`SIdentifiableStateProvider`] for static-sized version
pub trait DIdentifiableStateProvider: DOrbitStateProvider + Identifiable {}

// Blanket implementation for any type implementing both traits
impl<T: DOrbitStateProvider + Identifiable> DIdentifiableStateProvider for T {}

/// Trait to convert various propagator inputs into a slice of references.
///
/// This trait enables unified functions to accept either single propagators
/// or slices/vectors of propagators.
pub trait ToPropagatorRefs<P: DIdentifiableStateProvider> {
    /// Converts the input into a vector of references to propagators.
    fn to_refs(&self) -> Vec<&P>;
}

// Single propagator reference
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for P {
    fn to_refs(&self) -> Vec<&P> {
        vec![self]
    }
}

// Slice of propagators
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for [P] {
    fn to_refs(&self) -> Vec<&P> {
        self.iter().collect()
    }
}

// Vec of propagators
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for Vec<P> {
    fn to_refs(&self) -> Vec<&P> {
        self.iter().collect()
    }
}

// Slice of propagator references (for non-cloneable propagators like NumericalOrbitPropagator)
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for [&P] {
    fn to_refs(&self) -> Vec<&P> {
        self.to_vec()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::DEGREES;
    use crate::propagators::KeplerianPropagator;
    use crate::propagators::traits::SStatePropagator;
    use crate::time::{Epoch, TimeSystem};
    use crate::traits::{OrbitFrame, OrbitRepresentation};
    use nalgebra::Vector6;

    use serial_test::parallel;
    const TEST_EPOCH_JD: f64 = 2451545.0;

    fn create_test_propagator() -> KeplerianPropagator {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.01, 45.0, 0.0, 0.0, 0.0);
        KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(DEGREES),
            60.0,
        )
        .unwrap()
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_single_propagator() {
        let prop = create_test_propagator();
        let refs = prop.to_refs();
        assert_eq!(refs.len(), 1);
        // Verify the reference points to the original propagator
        assert_eq!(refs[0].initial_epoch(), prop.initial_epoch());
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_slice_of_propagators() {
        let props = [
            create_test_propagator(),
            create_test_propagator(),
            create_test_propagator(),
        ];
        let slice: &[KeplerianPropagator] = &props;
        let refs = slice.to_refs();
        assert_eq!(refs.len(), 3);
        // Verify each reference points to the correct propagator
        for (i, prop_ref) in refs.iter().enumerate() {
            assert_eq!(prop_ref.initial_epoch(), props[i].initial_epoch());
        }
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_vec_of_propagators() {
        let props = vec![create_test_propagator(), create_test_propagator()];
        let refs = props.to_refs();
        assert_eq!(refs.len(), 2);
        // Verify each reference points to the correct propagator
        for (i, prop_ref) in refs.iter().enumerate() {
            assert_eq!(prop_ref.initial_epoch(), props[i].initial_epoch());
        }
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_slice_of_refs() {
        let props = [
            create_test_propagator(),
            create_test_propagator(),
            create_test_propagator(),
            create_test_propagator(),
        ];
        // Create a slice of references
        let prop_refs: Vec<&KeplerianPropagator> = props.iter().collect();
        let slice_of_refs: &[&KeplerianPropagator] = &prop_refs;

        let refs = slice_of_refs.to_refs();
        assert_eq!(refs.len(), 4);
        // Verify each reference points to the correct propagator
        for (i, prop_ref) in refs.iter().enumerate() {
            assert_eq!(prop_ref.initial_epoch(), props[i].initial_epoch());
        }
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_empty_vec() {
        let props: Vec<KeplerianPropagator> = vec![];
        let refs = props.to_refs();
        assert_eq!(refs.len(), 0);
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_empty_slice() {
        let props: Vec<KeplerianPropagator> = vec![];
        let slice: &[KeplerianPropagator] = &props;
        let refs = slice.to_refs();
        assert_eq!(refs.len(), 0);
    }
}
