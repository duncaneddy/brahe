use brahe::AngleFormat;
use brahe::attitude::{EulerAngle, EulerAngleOrder, Quaternion, RotationMatrix, ToAttitude};
use brahe::math::SMatrix3;
use std::time::Instant;

pub fn quaternion_to_rotation_matrix(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let quaternions: Vec<Vec<f64>> = serde_json::from_value(params["quaternions"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(quaternions.len());

        for q in &quaternions {
            let quat = Quaternion::new(q[0], q[1], q[2], q[3]);
            let rm: RotationMatrix = quat.to_rotation_matrix();
            let mat = rm.to_matrix();
            // Row-major flattening
            let mut flat = Vec::with_capacity(9);
            for r in 0..3 {
                for c in 0..3 {
                    flat.push(mat[(r, c)]);
                }
            }
            results.push(flat);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn rotation_matrix_to_quaternion(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let matrices: Vec<Vec<Vec<f64>>> = serde_json::from_value(params["matrices"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(matrices.len());

        for mat in &matrices {
            let m = SMatrix3::new(
                mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0],
                mat[2][1], mat[2][2],
            );
            let rm = RotationMatrix::from_matrix(m).unwrap();
            let q: Quaternion = rm.to_quaternion();
            let v = q.to_vector(true); // [w, x, y, z]
            results.push(vec![v[0], v[1], v[2], v[3]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn quaternion_to_euler_angle(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let quaternions: Vec<Vec<f64>> = serde_json::from_value(params["quaternions"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(quaternions.len());

        for q in &quaternions {
            let quat = Quaternion::new(q[0], q[1], q[2], q[3]);
            let ea: EulerAngle = quat.to_euler_angle(EulerAngleOrder::ZYX);
            results.push(vec![ea.phi, ea.theta, ea.psi]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn euler_angle_to_quaternion(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let angles: Vec<Vec<f64>> = serde_json::from_value(params["angles"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(angles.len());

        for a in &angles {
            let ea = EulerAngle::new(EulerAngleOrder::ZYX, a[0], a[1], a[2], AngleFormat::Radians);
            let q: Quaternion = ea.to_quaternion();
            let v = q.to_vector(true); // [w, x, y, z]
            results.push(vec![v[0], v[1], v[2], v[3]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
