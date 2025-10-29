use approx::assert_abs_diff_eq;
use brahe::orbits::{anomaly_eccentric_to_mean, anomaly_mean_to_eccentric};

fn main() {
    let ecc = 45.0; // Starting true anomaly
    let e = 0.01;  // Eccentricity

    // Convert to eccentric anomaly
    let mean_anomaly = anomaly_eccentric_to_mean(ecc, e, true);

    // Convert back from eccentric to true anomaly
    let ecc_2 = anomaly_mean_to_eccentric(mean_anomaly, e, true).unwrap();

    // Confirm equality to within tolerance
    assert_abs_diff_eq!(ecc, ecc_2, epsilon=1e-14);
}