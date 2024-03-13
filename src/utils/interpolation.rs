/*!
This module defines traits and methods for interpolating between states.
 */

use nalgebra::SVector;

use crate::time::Epoch;

pub trait StateInterpolator<const S: usize> {
    fn interpolate(&self, t: Epoch) -> SVector<f64, S>;
}

pub fn lagrange_interpolation<const S: usize>(t: Epoch, data: Vec<(Epoch, SVector<f64, S>)>) -> SVector<f64, S> {
    let mut result = SVector::<f64, S>::zeros();
    for i in 0..data.len() {
        let mut term = data[i].1;
        for j in 0..data.len() {
            if i != j {
                term *= (t - data[j].0) / (data[i].0 - data[j].0);
            }
        }
        result += term;
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lagrange_interpolation() {
        // TODO: Implement tests
    }
}