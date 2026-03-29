/*!
 * Probability distributions for Monte Carlo variable sampling.
 *
 * Provides wrapper types that implement the [`MonteCarloDistribution`] trait,
 * producing [`MonteCarloSampledValue`] instances for use in Monte Carlo
 * simulations.
 *
 * Available distributions:
 * - [`Gaussian`]: Univariate normal distribution
 * - [`UniformDist`]: Continuous uniform distribution over \[low, high)
 * - [`TruncatedGaussian`]: Gaussian truncated to \[low, high\] via rejection sampling
 * - [`MultivariateNormal`]: Multivariate normal with full covariance
 */

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::distr::Uniform;

use crate::utils::BraheError;

use super::variables::MonteCarloSampledValue;

/// Trait for distributions that produce sampled values.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// parallel simulation workers.
pub trait MonteCarloDistribution: Send + Sync {
    /// Draw a single sample from this distribution.
    ///
    /// # Arguments
    ///
    /// - `rng` - Random number generator to use for sampling
    ///
    /// # Returns
    ///
    /// `MonteCarloSampledValue`: The sampled value
    fn sample(&self, rng: &mut dyn rand::RngCore) -> MonteCarloSampledValue;

    /// Human-readable name of this distribution.
    ///
    /// # Returns
    ///
    /// `&str`: Distribution name
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Helper: Box-Muller standard normal sampling
// ---------------------------------------------------------------------------

/// Generate a standard normal sample using the Box-Muller transform.
///
/// Uses two uniform samples from (0, 1) to produce a single N(0,1) value.
/// Generates u1, u2 in [0,1) using IEEE 754 double-precision mantissa bits,
/// rejecting u1 == 0 to avoid log(0).
fn standard_normal(rng: &mut dyn rand::RngCore) -> f64 {
    let u1: f64;
    let u2: f64;
    loop {
        let bits1 = rng.next_u64();
        let candidate = (bits1 >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        if candidate > 0.0 {
            u1 = candidate;
            let bits2 = rng.next_u64();
            u2 = (bits2 >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            break;
        }
    }
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Gaussian
// ---------------------------------------------------------------------------

/// Univariate Gaussian (normal) distribution.
///
/// Samples a single scalar value from N(mean, std^2).
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::distributions::{Gaussian, MonteCarloDistribution};
/// use rand::SeedableRng;
///
/// let dist = Gaussian { mean: 0.0, std: 1.0 };
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let sample = dist.sample(&mut rng);
/// ```
#[derive(Clone, Debug)]
pub struct Gaussian {
    /// Mean of the distribution.
    pub mean: f64,
    /// Standard deviation of the distribution (must be non-negative).
    pub std: f64,
}

impl MonteCarloDistribution for Gaussian {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> MonteCarloSampledValue {
        let z = standard_normal(rng);
        MonteCarloSampledValue::Scalar(self.mean + self.std * z)
    }

    fn name(&self) -> &str {
        "Gaussian"
    }
}

// ---------------------------------------------------------------------------
// UniformDist
// ---------------------------------------------------------------------------

/// Continuous uniform distribution over \[low, high).
///
/// Samples a single scalar value uniformly from the interval.
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::distributions::{UniformDist, MonteCarloDistribution};
/// use rand::SeedableRng;
///
/// let dist = UniformDist { low: -1.0, high: 1.0 };
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let sample = dist.sample(&mut rng);
/// ```
#[derive(Clone, Debug)]
pub struct UniformDist {
    /// Lower bound (inclusive).
    pub low: f64,
    /// Upper bound (exclusive).
    pub high: f64,
}

impl MonteCarloDistribution for UniformDist {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> MonteCarloSampledValue {
        let dist = Uniform::new(self.low, self.high).expect("Invalid uniform bounds");
        let value: f64 = rng.sample(dist);
        MonteCarloSampledValue::Scalar(value)
    }

    fn name(&self) -> &str {
        "Uniform"
    }
}

// ---------------------------------------------------------------------------
// TruncatedGaussian
// ---------------------------------------------------------------------------

/// Gaussian distribution truncated to the interval \[low, high\].
///
/// Uses rejection sampling: draws from N(mean, std^2) and discards values
/// outside the bounds. This is efficient when the truncation interval covers
/// a reasonable fraction of the distribution's probability mass.
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::distributions::{TruncatedGaussian, MonteCarloDistribution};
/// use rand::SeedableRng;
///
/// let dist = TruncatedGaussian {
///     mean: 2.2,
///     std: 0.1,
///     low: 1.8,
///     high: 2.6,
/// };
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let sample = dist.sample(&mut rng);
/// ```
#[derive(Clone, Debug)]
pub struct TruncatedGaussian {
    /// Mean of the underlying Gaussian.
    pub mean: f64,
    /// Standard deviation of the underlying Gaussian.
    pub std: f64,
    /// Lower truncation bound (inclusive).
    pub low: f64,
    /// Upper truncation bound (inclusive).
    pub high: f64,
}

impl MonteCarloDistribution for TruncatedGaussian {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> MonteCarloSampledValue {
        loop {
            let z = standard_normal(rng);
            let value = self.mean + self.std * z;
            if value >= self.low && value <= self.high {
                return MonteCarloSampledValue::Scalar(value);
            }
        }
    }

    fn name(&self) -> &str {
        "TruncatedGaussian"
    }
}

// ---------------------------------------------------------------------------
// MultivariateNormal
// ---------------------------------------------------------------------------

/// Multivariate normal distribution.
///
/// Stores the mean vector and the lower-triangular Cholesky factor L of the
/// covariance matrix (cov = L * L^T). Sampling is done via mean + L * z
/// where z ~ N(0, I).
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::distributions::{MultivariateNormal, MonteCarloDistribution};
/// use nalgebra::{DVector, DMatrix};
/// use rand::SeedableRng;
///
/// let mean = DVector::from_vec(vec![0.0, 0.0]);
/// let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
/// let dist = MultivariateNormal::new(mean, cov).unwrap();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let sample = dist.sample(&mut rng);
/// ```
#[derive(Clone, Debug)]
pub struct MultivariateNormal {
    /// Mean vector.
    pub mean: DVector<f64>,
    /// Lower-triangular Cholesky factor of the covariance matrix.
    pub cholesky: DMatrix<f64>,
}

impl MultivariateNormal {
    /// Create a multivariate normal distribution from a mean vector and
    /// covariance matrix.
    ///
    /// Computes the Cholesky decomposition of the covariance matrix. The
    /// covariance must be symmetric positive-definite.
    ///
    /// # Arguments
    ///
    /// - `mean` - Mean vector (n-dimensional)
    /// - `cov` - Covariance matrix (n x n, symmetric positive-definite)
    ///
    /// # Returns
    ///
    /// `MultivariateNormal`: The distribution, ready for sampling
    ///
    /// # Errors
    ///
    /// Returns [`BraheError::NumericalError`] if the covariance matrix is not
    /// square, does not match the mean dimension, or is not positive-definite
    /// (Cholesky decomposition fails).
    pub fn new(mean: DVector<f64>, cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let n = mean.nrows();
        if cov.nrows() != n || cov.ncols() != n {
            return Err(BraheError::NumericalError(format!(
                "Covariance matrix dimensions ({}x{}) do not match mean dimension ({})",
                cov.nrows(),
                cov.ncols(),
                n
            )));
        }

        let cholesky = cov.clone().cholesky().ok_or_else(|| {
            BraheError::NumericalError(
                "Cholesky decomposition failed: covariance matrix is not positive-definite"
                    .to_string(),
            )
        })?;

        Ok(Self {
            mean,
            cholesky: cholesky.l(),
        })
    }
}

impl MonteCarloDistribution for MultivariateNormal {
    fn sample(&self, rng: &mut dyn rand::RngCore) -> MonteCarloSampledValue {
        let n = self.mean.nrows();
        let z = DVector::from_fn(n, |_, _| standard_normal(rng));
        let sample = &self.mean + &self.cholesky * z;
        MonteCarloSampledValue::Vector(sample)
    }

    fn name(&self) -> &str {
        "MultivariateNormal"
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const N_SAMPLES: usize = 10_000;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(12345)
    }

    // -- Gaussian tests --

    #[test]
    fn test_gaussian_name() {
        let dist = Gaussian {
            mean: 0.0,
            std: 1.0,
        };
        assert_eq!(dist.name(), "Gaussian");
    }

    #[test]
    fn test_gaussian_returns_scalar() {
        let dist = Gaussian {
            mean: 5.0,
            std: 1.0,
        };
        let mut rng = make_rng();
        let val = dist.sample(&mut rng);
        assert!(matches!(val, MonteCarloSampledValue::Scalar(_)));
    }

    #[test]
    fn test_gaussian_mean_and_std() {
        let target_mean = 10.0;
        let target_std = 2.0;
        let dist = Gaussian {
            mean: target_mean,
            std: target_std,
        };
        let mut rng = make_rng();

        let samples: Vec<f64> = (0..N_SAMPLES)
            .map(|_| match dist.sample(&mut rng) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        let std = variance.sqrt();

        assert!(
            (mean - target_mean).abs() < 0.1,
            "Sample mean {} too far from target {}",
            mean,
            target_mean
        );
        assert!(
            (std - target_std).abs() < 0.1,
            "Sample std {} too far from target {}",
            std,
            target_std
        );
    }

    #[test]
    fn test_gaussian_zero_std() {
        let dist = Gaussian {
            mean: 42.0,
            std: 0.0,
        };
        let mut rng = make_rng();
        for _ in 0..100 {
            match dist.sample(&mut rng) {
                MonteCarloSampledValue::Scalar(v) => assert_eq!(v, 42.0),
                _ => panic!("Expected scalar"),
            }
        }
    }

    // -- UniformDist tests --

    #[test]
    fn test_uniform_name() {
        let dist = UniformDist {
            low: 0.0,
            high: 1.0,
        };
        assert_eq!(dist.name(), "Uniform");
    }

    #[test]
    fn test_uniform_returns_scalar() {
        let dist = UniformDist {
            low: 0.0,
            high: 1.0,
        };
        let mut rng = make_rng();
        let val = dist.sample(&mut rng);
        assert!(matches!(val, MonteCarloSampledValue::Scalar(_)));
    }

    #[test]
    fn test_uniform_bounds() {
        let low = -5.0;
        let high = 5.0;
        let dist = UniformDist { low, high };
        let mut rng = make_rng();

        for _ in 0..N_SAMPLES {
            match dist.sample(&mut rng) {
                MonteCarloSampledValue::Scalar(v) => {
                    assert!(v >= low, "Sample {} below low bound {}", v, low);
                    assert!(v < high, "Sample {} at or above high bound {}", v, high);
                }
                _ => panic!("Expected scalar"),
            }
        }
    }

    #[test]
    fn test_uniform_mean() {
        let low = 2.0;
        let high = 8.0;
        let expected_mean = (low + high) / 2.0;
        let dist = UniformDist { low, high };
        let mut rng = make_rng();

        let samples: Vec<f64> = (0..N_SAMPLES)
            .map(|_| match dist.sample(&mut rng) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - expected_mean).abs() < 0.1,
            "Sample mean {} too far from expected {}",
            mean,
            expected_mean
        );
    }

    // -- TruncatedGaussian tests --

    #[test]
    fn test_truncated_gaussian_name() {
        let dist = TruncatedGaussian {
            mean: 0.0,
            std: 1.0,
            low: -2.0,
            high: 2.0,
        };
        assert_eq!(dist.name(), "TruncatedGaussian");
    }

    #[test]
    fn test_truncated_gaussian_returns_scalar() {
        let dist = TruncatedGaussian {
            mean: 0.0,
            std: 1.0,
            low: -2.0,
            high: 2.0,
        };
        let mut rng = make_rng();
        let val = dist.sample(&mut rng);
        assert!(matches!(val, MonteCarloSampledValue::Scalar(_)));
    }

    #[test]
    fn test_truncated_gaussian_within_bounds() {
        let low = -1.5;
        let high = 1.5;
        let dist = TruncatedGaussian {
            mean: 0.0,
            std: 1.0,
            low,
            high,
        };
        let mut rng = make_rng();

        for _ in 0..N_SAMPLES {
            match dist.sample(&mut rng) {
                MonteCarloSampledValue::Scalar(v) => {
                    assert!(v >= low, "Sample {} below low bound {}", v, low);
                    assert!(v <= high, "Sample {} above high bound {}", v, high);
                }
                _ => panic!("Expected scalar"),
            }
        }
    }

    #[test]
    fn test_truncated_gaussian_mean_near_center() {
        let dist = TruncatedGaussian {
            mean: 5.0,
            std: 0.5,
            low: 3.0,
            high: 7.0,
        };
        let mut rng = make_rng();

        let samples: Vec<f64> = (0..N_SAMPLES)
            .map(|_| match dist.sample(&mut rng) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        // With wide bounds relative to std, truncated mean should be close to mean
        assert!(
            (mean - 5.0).abs() < 0.1,
            "Sample mean {} too far from 5.0",
            mean
        );
    }

    // -- MultivariateNormal tests --

    #[test]
    fn test_multivariate_normal_name() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::identity(2, 2);
        let dist = MultivariateNormal::new(mean, cov).unwrap();
        assert_eq!(dist.name(), "MultivariateNormal");
    }

    #[test]
    fn test_multivariate_normal_returns_vector() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::identity(2, 2);
        let dist = MultivariateNormal::new(mean, cov).unwrap();
        let mut rng = make_rng();
        let val = dist.sample(&mut rng);
        assert!(matches!(val, MonteCarloSampledValue::Vector(_)));
    }

    #[test]
    fn test_multivariate_normal_dimension() {
        let n = 4;
        let mean = DVector::zeros(n);
        let cov = DMatrix::identity(n, n);
        let dist = MultivariateNormal::new(mean, cov).unwrap();
        let mut rng = make_rng();

        match dist.sample(&mut rng) {
            MonteCarloSampledValue::Vector(v) => assert_eq!(v.nrows(), n),
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn test_multivariate_normal_mean() {
        let target_mean = DVector::from_vec(vec![10.0, -5.0, 3.0]);
        let cov = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let dist = MultivariateNormal::new(target_mean.clone(), cov).unwrap();
        let mut rng = make_rng();

        let mut sum = DVector::zeros(3);
        for _ in 0..N_SAMPLES {
            match dist.sample(&mut rng) {
                MonteCarloSampledValue::Vector(v) => sum += &v,
                _ => panic!("Expected vector"),
            }
        }
        let sample_mean = sum / N_SAMPLES as f64;

        for i in 0..3 {
            assert!(
                (sample_mean[i] - target_mean[i]).abs() < 0.1,
                "Component {} sample mean {} too far from target {}",
                i,
                sample_mean[i],
                target_mean[i]
            );
        }
    }

    #[test]
    fn test_multivariate_normal_correlated() {
        // Test with off-diagonal covariance
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.8, 0.8, 1.0]);
        let dist = MultivariateNormal::new(mean, cov).unwrap();
        let mut rng = make_rng();

        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for _ in 0..N_SAMPLES {
            match dist.sample(&mut rng) {
                MonteCarloSampledValue::Vector(v) => {
                    sum_x += v[0];
                    sum_y += v[1];
                    sum_xy += v[0] * v[1];
                }
                _ => panic!("Expected vector"),
            }
        }
        let n = N_SAMPLES as f64;
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;
        let cov_xy = sum_xy / n - mean_x * mean_y;

        assert!(
            (cov_xy - 0.8).abs() < 0.1,
            "Sample covariance {} too far from 0.8",
            cov_xy
        );
    }

    #[test]
    fn test_multivariate_normal_dimension_mismatch() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::identity(3, 3);
        let result = MultivariateNormal::new(mean, cov);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("dimensions"));
    }

    #[test]
    fn test_multivariate_normal_non_square_cov() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::zeros(2, 3);
        let result = MultivariateNormal::new(mean, cov);
        assert!(result.is_err());
    }

    #[test]
    fn test_multivariate_normal_not_positive_definite() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        // Negative eigenvalue matrix
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let result = MultivariateNormal::new(mean, cov);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("positive-definite"));
    }

    #[test]
    fn test_multivariate_normal_cholesky_stored() {
        let mean = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]);
        let dist = MultivariateNormal::new(mean, cov).unwrap();
        // L should be diag(2, 3)
        assert!((dist.cholesky[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((dist.cholesky[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((dist.cholesky[(0, 1)]).abs() < 1e-10);
    }

    // -- Reproducibility test --

    #[test]
    fn test_gaussian_reproducible() {
        let dist = Gaussian {
            mean: 0.0,
            std: 1.0,
        };
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let v1 = match dist.sample(&mut rng1) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            };
            let v2 = match dist.sample(&mut rng2) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            };
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn test_uniform_reproducible() {
        let dist = UniformDist {
            low: 0.0,
            high: 1.0,
        };
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let v1 = match dist.sample(&mut rng1) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            };
            let v2 = match dist.sample(&mut rng2) {
                MonteCarloSampledValue::Scalar(v) => v,
                _ => panic!("Expected scalar"),
            };
            assert_eq!(v1, v2);
        }
    }

    // -- Trait object tests --

    #[test]
    fn test_distribution_as_trait_object() {
        let dist: Box<dyn MonteCarloDistribution> = Box::new(Gaussian {
            mean: 0.0,
            std: 1.0,
        });
        let mut rng = make_rng();
        let val = dist.sample(&mut rng);
        assert!(matches!(val, MonteCarloSampledValue::Scalar(_)));
        assert_eq!(dist.name(), "Gaussian");
    }

    #[test]
    fn test_multiple_distributions_as_trait_objects() {
        let distributions: Vec<Box<dyn MonteCarloDistribution>> = vec![
            Box::new(Gaussian {
                mean: 0.0,
                std: 1.0,
            }),
            Box::new(UniformDist {
                low: 0.0,
                high: 1.0,
            }),
            Box::new(TruncatedGaussian {
                mean: 0.0,
                std: 1.0,
                low: -2.0,
                high: 2.0,
            }),
            Box::new(
                MultivariateNormal::new(DVector::from_vec(vec![0.0]), DMatrix::identity(1, 1))
                    .unwrap(),
            ),
        ];

        let mut rng = make_rng();
        for dist in &distributions {
            let _ = dist.sample(&mut rng);
        }
    }
}
