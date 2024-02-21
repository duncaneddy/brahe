/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
 */

use nalgebra::{SMatrix, SVector};

pub trait NumericalIntegrator<const S: usize> {
    fn step(&self, t: f64, state: &SVector<f64, S>, dt: f64) -> SVector<f64, S>;
    fn step_with_varmat(&self, t: f64, state: &SVector<f64, S>, phi: &SMatrix<f64, S, S>, dt: f64) -> (SVector<f64, S>, SMatrix<f64, S, S>);
}

pub fn varmat_from_percentage_offset<const S: usize>(t: f64, state: &SVector<f64, S>, f: fn(f64, &SVector<f64, S>) -> SVector<f64, S>, percentage: f64) -> SMatrix<f64, S, S> {
    // Note: The variational matrix development along with the perturbation calculation is
    // defined as a seprate funciton in the RK4Integrator struct instead of being implemented
    // as part of the rk_step_varmat function. This is because the choice of how the variational
    // matrix is calculated is specific

    // Evaluate unperturbed state derivative function
    let fx = f(t, state);

    // Initialize the variational matrix
    let mut phi = SMatrix::<f64, S, S>::zeros();

    // Compute the variational matrix for each state component
    for i in 0..S {
        // Compute the perturbation for the element
        let mut px = *state;
        let offset = state[i] * percentage;
        px[i] += offset;

        let pfx = f(t, &px);
        phi.set_column(i, &((pfx - fx) / offset));
    }

    phi
}

pub fn varmat_from_fixed_offset<const S: usize>(t: f64, state: &SVector<f64, S>, f: fn(f64, &SVector<f64, S>) -> SVector<f64, S>, offset: f64) -> SMatrix<f64, S, S> {
    // Note: The variational matrix development along with the perturbation calculation is
    // defined as a seprate funciton in the RK4Integrator struct instead of being implemented
    // as part of the rk_step_varmat function. This is because the choice of how the variational
    // matrix is calculated is specific

    // Evaluate unperturbed state derivative function
    let fx = f(t, state);

    // Initialize the variational matrix
    let mut phi = SMatrix::<f64, S, S>::zeros();

    // Compute the variational matrix for each state component
    for i in 0..S {
        // Compute the perturbation for the element
        let mut px = *state;
        px[i] += offset;

        let pfx = f(t, &px);
        phi.set_column(i, &((pfx - fx) / offset));
    }

    phi
}

pub fn varmat_from_offset_vector<const S: usize>(t: f64, state: &SVector<f64, S>, f: fn(f64, &SVector<f64, S>) -> SVector<f64, S>, offset: &SVector<f64, S>) -> SMatrix<f64, S, S> {
    // Note: The variational matrix development along with the perturbation calculation is
    // defined as a seprate funciton in the RK4Integrator struct instead of being implemented
    // as part of the rk_step_varmat function. This is because the choice of how the variational
    // matrix is calculated is specific

    // Evaluate unperturbed state derivative function
    let fx = f(t, state);

    // Initialize the variational matrix
    let mut phi = SMatrix::<f64, S, S>::zeros();

    // Compute the perturbed for each state component
    let state_perturbation = state + offset;

    // Compute the variational matrix for each state component
    for i in 0..S {
        let pfx = f(t, &(state + state_perturbation));
        phi.set_column(i, &((pfx - fx) / state_perturbation[i]));
    }

    phi
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_varmat_from_percentage_offset() {
        // todo!()
    }

    #[test]
    fn test_varmat_from_fixed_offset() {
        // todo!()
    }

    #[test]
    fn test_varmat_from_offset_vector() {
        // todo!()
    }
}