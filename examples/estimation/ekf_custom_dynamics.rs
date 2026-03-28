//! Create an EKF with custom dynamics using from_propagator().

#[allow(unused_imports)]
use brahe as bh;
use nalgebra::{DMatrix, DVector};

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::time::TimeSystem::UTC);
    let r = bh::constants::physical::R_EARTH + 500e3;
    let v = (bh::constants::physical::GM_EARTH / r).sqrt();
    let state = DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0]);
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

    // Define custom two-body dynamics
    let dynamics: bh::integrators::traits::DStateDynamics = Box::new(
        |_t, state: &DVector<f64>, _params| {
            let r_vec = state.rows(0, 3);
            let v_vec = state.rows(3, 3);
            let r_mag = r_vec.norm();
            let mu = bh::constants::physical::GM_EARTH;
            let a = -mu / r_mag.powi(3) * &r_vec;
            let mut dx = DVector::zeros(6);
            dx.rows_mut(0, 3).copy_from(&v_vec);
            dx.rows_mut(3, 3).copy_from(&a);
            dx
        },
    );

    // Build a generic propagator with STM enabled
    let mut prop_config = bh::propagators::NumericalPropagationConfig::default();
    prop_config.variational.enable_stm = true;

    let prop = bh::propagators::DNumericalPropagator::new(
        epoch, state.clone(), dynamics, prop_config, None, None, Some(p0),
    ).unwrap();

    // Create EKF from the pre-built propagator
    let models: Vec<Box<dyn bh::estimation::MeasurementModel>> = vec![
        Box::new(bh::estimation::InertialPositionMeasurementModel::new(10.0)),
    ];

    let mut ekf = bh::estimation::ExtendedKalmanFilter::from_propagator(
        bh::estimation::DynamicsSource::GenericPropagator(prop),
        models,
        bh::estimation::EKFConfig::default(),
    ).unwrap();

    // Process a single observation
    let obs = bh::estimation::Observation::new(epoch + 60.0, state.rows(0, 3).into_owned(), 0);
    let record = ekf.process_observation(&obs).unwrap();

    println!("Custom dynamics EKF:");
    println!("  Prefit residual norm: {:.3} m", record.prefit_residual.norm());
    println!("  Postfit residual norm: {:.6} m", record.postfit_residual.norm());
    println!("  State dim: {}", ekf.current_state().len());
}
