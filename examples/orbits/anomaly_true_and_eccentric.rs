use approx::assert_abs_diff_eq;
use brahe::orbits::{anomaly_true_to_eccentric, anomaly_eccentric_to_true};

fn main() {
    let nu = 45.0; // Starting true anomaly
    let e = 0.01;  // Eccentricity

    // Convert to eccentric anomaly
    let ecc_anomaly = anomaly_true_to_eccentric(nu, e, true);

    // Convert back from eccentric to true anomaly
    let nu_2 = anomaly_eccentric_to_true(ecc_anomaly, e, true);

    // Confirm equality to within tolerance
    assert_abs_diff_eq!(nu, nu_2, epsilon=1e-14);
}