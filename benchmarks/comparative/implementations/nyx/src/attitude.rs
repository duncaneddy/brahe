//! Attitude conversions using ANISE's DCM + EulerParameter primitives.
//!
//! Brahe conventions (input/output — both files agree):
//!   - Quaternion: [w, x, y, z] (scalar first)
//!   - Matrix: 3x3 passive rotation matrix, row-major
//!       row0: [1-2(y²+z²), 2(xy+wz), 2(xz-wy)]
//!   - Euler order: ZYX (3-2-1 aerospace) [phi (Z first), theta (Y), psi (X last)] in radians
//!
//! ANISE/Nyx conventions (as observed during implementation):
//!   - Quaternion (EulerParameter): scalar-first `EulerParameter::new(w, x, y, z, from, to)`.
//!     Same Hamiltonian algebra, same scalar-first component order as brahe.
//!   - DCM (EulerParameter ↔ DCM): `DCM::from(ep)` and `EulerParameter::from(dcm)` use the
//!     same passive-rotation matrix formula as brahe's RotationMatrix — element-for-element
//!     identical, no transposing required.
//!   - No built-in Euler-angle extraction. ZYX angles are extracted from the ANISE DCM using
//!     the same Diebel-2006 index formulas brahe uses internally:
//!       theta = asin(-m[0,2]),  phi = atan2(m[0,1], m[0,0]),  psi = atan2(m[1,2], m[2,2])
//!   - Euler→DCM composition: the brahe ZYX passive matrix is `R1(psi)·R2(theta)·R3(phi)`.
//!     ANISE's `A*B` operator evaluates to matrix_A · matrix_B, so the ANISE call chain is:
//!       R1(2→3) * R2(1→2) → r1·r2 (1→3),  then · R3(0→1) → r1·r2·r3 (0→3).
//!   - Frame IDs (NaifId = i32) are required by ANISE but carry no physical meaning here.

use anise::math::rotation::{DCM, EulerParameter};
use anise::math::Matrix3;
use std::time::Instant;

// Dummy frame IDs — ANISE requires them but they carry no physical meaning here.
const FROM: i32 = 0;
const TO: i32 = 1;

/// Extract ZYX Euler angles [phi, theta, psi] (radians) from an ANISE passive rotation DCM.
///
/// ANISE's DCM is element-for-element identical to brahe's RotationMatrix, so the same
/// Diebel-based extraction formulas apply (brahe rotation_matrix.rs, ZYX dispatch, ~line 556):
///
///   theta = asin(-m[0,2])
///   phi   = atan2(m[0,1], m[0,0])
///   psi   = atan2(m[1,2], m[2,2])
///
/// (This is Diebel §8 for "XYZ" in Diebel's matrix-product convention, which corresponds to
///  aerospace ZYX because brahe reverses the order label before dispatching.)
fn dcm_to_zyx_euler(dcm: &DCM) -> [f64; 3] {
    let m = dcm.rot_mat;
    // m[(row, col)] — nalgebra 0-based (row, col) indexing.
    let sin_theta = -m[(0, 2)];
    let theta = sin_theta.clamp(-1.0, 1.0).asin();
    let cos_theta = theta.cos();

    let (phi, psi) = if cos_theta.abs() > 1e-10 {
        let phi = m[(0, 1)].atan2(m[(0, 0)]);
        let psi = m[(1, 2)].atan2(m[(2, 2)]);
        (phi, psi)
    } else {
        // Gimbal lock (theta ≈ ±π/2): set psi = 0, solve for phi.
        let phi = m[(1, 0)].atan2(m[(1, 1)]);
        (phi, 0.0)
    };

    [phi, theta, psi]
}

/// Build a passive ZYX DCM from [phi, theta, psi] in radians whose elements
/// match brahe's RotationMatrix for the same ZYX angles.
///
/// The matrix product that brahe computes for ZYX(phi,theta,psi) is
/// `R1(psi) · R2(theta) · R3(phi)` (innermost rotation right-to-left in
/// the matrix sense). Verified analytically:
///   m[0,2] = -sin(theta)
///   m[0,1] = cos(theta)*sin(phi)
///   m[0,0] = cos(theta)*cos(phi)
///   etc.
///
/// ANISE's `*` operator for DCMs is defined as:
///   `q_A_to_C = q_B_to_C * q_A_to_B`   (LHS.from == RHS.to)
/// So the matrix of `A * B` equals matrix_A · matrix_B.
///
/// To obtain the matrix product `r1 · r2 · r3` via ANISE:
///   Step 1: R1(2→3) * R2(1→2)  → result 1→3, matrix r1·r2
///   Step 2: result(1→3) * R3(0→1) → result 0→3, matrix r1·r2·r3
fn zyx_euler_to_passive_dcm(phi: f64, theta: f64, psi: f64) -> DCM {
    let r3 = DCM::r3(phi, 0, 1);
    let r2 = DCM::r2(theta, 1, 2);
    let r1 = DCM::r1(psi, 2, 3);
    let r12 = (r1 * r2).expect("DCM multiply r1*r2");
    (r12 * r3).expect("DCM multiply (r1*r2)*r3")
}

pub fn quaternion_to_rotation_matrix(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let quaternions: Vec<Vec<f64>> =
        serde_json::from_value(params["quaternions"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(quaternions.len());

        for q in &quaternions {
            // q: [w, x, y, z] — ANISE EulerParameter uses same scalar-first order.
            let ep = EulerParameter::new(q[0], q[1], q[2], q[3], FROM, TO);
            let dcm = DCM::from(ep);
            let m = dcm.rot_mat;
            // ANISE DCM is element-for-element identical to brahe RotationMatrix.
            // Flatten row-major: [r00, r01, r02, r10, r11, r12, r20, r21, r22].
            let flat: Vec<f64> = (0..3)
                .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
                .collect();
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
    // Input: params["matrices"] is Vec<Vec<Vec<f64>>>, each 3x3 row-major rotation matrix
    // (same passive format as ANISE DCM and brahe RotationMatrix).
    // Output: list of [w, x, y, z].
    let matrices: Vec<Vec<Vec<f64>>> =
        serde_json::from_value(params["matrices"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(matrices.len());

        for mat in &matrices {
            // Matrix3::new takes row-major arguments (m11, m12, m13, m21, ...).
            let rot_mat = Matrix3::new(
                mat[0][0], mat[0][1], mat[0][2],
                mat[1][0], mat[1][1], mat[1][2],
                mat[2][0], mat[2][1], mat[2][2],
            );
            let dcm = DCM {
                rot_mat,
                rot_mat_dt: None,
                from: FROM,
                to: TO,
            };
            let ep = EulerParameter::from(dcm);
            // Enforce w ≥ 0 canonical form (short rotation) to match brahe's sign convention.
            let ep = ep.short();
            results.push(vec![ep.w, ep.x, ep.y, ep.z]);
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
    // Input: params["quaternions"], list of [w, x, y, z].
    // Output: list of [phi, theta, psi] in radians (ZYX/3-2-1 aerospace order).
    let quaternions: Vec<Vec<f64>> =
        serde_json::from_value(params["quaternions"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(quaternions.len());

        for q in &quaternions {
            let ep = EulerParameter::new(q[0], q[1], q[2], q[3], FROM, TO);
            let dcm = DCM::from(ep);
            let angles = dcm_to_zyx_euler(&dcm);
            results.push(vec![angles[0], angles[1], angles[2]]);
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
    // Input: params["angles"], list of [phi, theta, psi] in radians (ZYX aerospace).
    // Output: list of [w, x, y, z].
    let angles: Vec<Vec<f64>> =
        serde_json::from_value(params["angles"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(angles.len());

        for a in &angles {
            let phi = a[0];
            let theta = a[1];
            let psi = a[2];
            let dcm = zyx_euler_to_passive_dcm(phi, theta, psi);
            let ep = EulerParameter::from(dcm);
            let ep = ep.short();
            results.push(vec![ep.w, ep.x, ep.y, ep.z]);
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
