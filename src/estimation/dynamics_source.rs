/*!
 * Dynamics source abstraction for estimation filters.
 *
 * Wraps either a [`DNumericalOrbitPropagator`] or a [`DNumericalPropagator`]
 * to provide a unified interface for state prediction, STM access, and
 * covariance propagation.
 *
 * [`DNumericalOrbitPropagator`]: crate::propagators::DNumericalOrbitPropagator
 * [`DNumericalPropagator`]: crate::propagators::DNumericalPropagator
 */

use nalgebra::{DMatrix, DVector};

use crate::propagators::traits::DStatePropagator;
use crate::propagators::{DNumericalOrbitPropagator, DNumericalPropagator};
use crate::time::Epoch;

/// Dynamics source for estimation filters.
///
/// Wraps either a built-in orbit propagator (with force models) or a generic
/// propagator (with user-defined dynamics) to provide state prediction,
/// covariance propagation, and STM access needed by the estimation filters.
///
/// # Why an enum instead of a trait?
///
/// The set of numerical propagator types is small and known (2 types). An enum
/// avoids adding `stm()` to the `DStatePropagator` trait, which would force all
/// propagator implementations (Keplerian, SGP4) to deal with it.
pub enum DynamicsSource {
    /// Built-in orbit propagator with force models
    OrbitPropagator(DNumericalOrbitPropagator),
    /// Generic propagator with user-defined dynamics
    GenericPropagator(DNumericalPropagator),
}

impl DynamicsSource {
    /// Propagate to a target epoch.
    pub fn propagate_to(&mut self, epoch: Epoch) {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.propagate_to(epoch),
            DynamicsSource::GenericPropagator(p) => p.propagate_to(epoch),
        }
    }

    /// Get current state vector.
    pub fn current_state(&self) -> DVector<f64> {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.current_state(),
            DynamicsSource::GenericPropagator(p) => p.current_state(),
        }
    }

    /// Get current epoch.
    pub fn current_epoch(&self) -> Epoch {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.current_epoch(),
            DynamicsSource::GenericPropagator(p) => p.current_epoch(),
        }
    }

    /// Get current STM (None if not enabled).
    pub fn stm(&self) -> Option<&DMatrix<f64>> {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.stm(),
            DynamicsSource::GenericPropagator(p) => p.stm(),
        }
    }

    /// Get current propagated covariance P(t) (None if not set).
    ///
    /// The propagator computes `P(t) = Φ(t,t₀)·P₀·Φ(t,t₀)ᵀ` during each step
    /// when a covariance was provided via [`reinitialize`].
    pub fn current_covariance(&self) -> Option<&DMatrix<f64>> {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.current_covariance(),
            DynamicsSource::GenericPropagator(p) => p.current_covariance(),
        }
    }

    /// Returns true if this dynamics source has STM propagation enabled.
    pub fn has_stm(&self) -> bool {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.has_stm(),
            DynamicsSource::GenericPropagator(p) => p.has_stm(),
        }
    }

    /// Reinitialize the propagator with a new state and optional covariance.
    ///
    /// Resets STM to identity while preserving dynamics, integrator, and
    /// force configuration. If covariance is provided, the propagator will
    /// compute `P(t) = Φ·P₀·Φᵀ` during subsequent propagation.
    pub fn reinitialize(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        covariance: Option<DMatrix<f64>>,
    ) {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.reinitialize(epoch, state, covariance),
            DynamicsSource::GenericPropagator(p) => p.reinitialize(epoch, state, covariance),
        }
    }

    /// Get state dimension.
    pub fn state_dim(&self) -> usize {
        match self {
            DynamicsSource::OrbitPropagator(p) => p.state_dim(),
            DynamicsSource::GenericPropagator(p) => p.state_dim(),
        }
    }
}
