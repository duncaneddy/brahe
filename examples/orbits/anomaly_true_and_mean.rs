use approx::assert_abs_diff_eq;
use brahe::orbits::{anomaly_eccentric_to_mean, anomaly_mean_to_eccentric};

fn main() {
    let nu = 45.0; // Starting true anomaly
    let e = 0.01;  // Eccentricity

    // Convert to eccentric anomaly
    let mean_anomaly = anomaly_eccentric_to_mean(nu, e, true);

    // Convert back from eccentric to true anomaly
    let nu_2 = anomaly_mean_to_eccentric(mean_anomaly, e, true).unwrap();

    // Confirm equality to within tolerance
    assert_abs_diff_eq!(nu, nu_2, epsilon=1e-14);
}