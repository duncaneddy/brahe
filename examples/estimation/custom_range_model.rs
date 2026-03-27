//! Define a custom measurement model and use it with the EKF.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra::{DMatrix, DVector};

/// Custom range measurement model: distance from a station to the satellite.
struct RangeModel {
    station_eci: DVector<f64>,
    noise_cov: DMatrix<f64>,
}

impl RangeModel {
    fn new(station_eci: DVector<f64>, sigma: f64) -> Self {
        let noise_cov = DMatrix::from_element(1, 1, sigma * sigma);
        Self { station_eci, noise_cov }
    }
}

impl bh::estimation::MeasurementModel for RangeModel {
    fn predict(
        &self,
        _epoch: &bh::time::Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, bh::utils::errors::BraheError> {
        let range = (state.rows(0, 3) - &self.station_eci).norm();
        Ok(DVector::from_vec(vec![range]))
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Range"
    }
}

fn main() {
    bh::initialize_eop().unwrap();

    // Set up orbit and truth propagator
    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::time::TimeSystem::UTC);
    let r = bh::constants::physical::R_EARTH + 500e3;
    let v = (bh::constants::physical::GM_EARTH / r).sqrt();
    let true_state = DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0]);

    let mut truth_prop = bh::propagators::DNumericalOrbitPropagator::new(
        epoch,
        true_state.clone(),
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None, None,
    ).unwrap();

    // Ground station on the equator
    let station = DVector::from_vec(vec![bh::constants::physical::R_EARTH, 0.0, 0.0]);

    // Create EKF with both a built-in model and a custom range model
    let position_model = bh::estimation::InertialPositionMeasurementModel::new(10.0);
    let range_model = RangeModel::new(station.clone(), 100.0);

    let mut initial_state = true_state.clone();
    initial_state[0] += 500.0;
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![
        1e6, 1e6, 1e6, 1e2, 1e2, 1e2,
    ]));

    let models: Vec<Box<dyn bh::estimation::MeasurementModel>> = vec![
        Box::new(position_model),
        Box::new(range_model),
    ];

    let mut ekf = bh::estimation::ExtendedKalmanFilter::new(
        epoch,
        initial_state,
        p0,
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None,
        models,
        bh::estimation::EKFConfig::default(),
    ).unwrap();

    // Alternate between position and range observations
    let dt = 60.0;
    for i in 1..=20 {
        let obs_epoch = epoch + dt * i as f64;
        truth_prop.propagate_to(obs_epoch);
        let truth_st = truth_prop.current_state();

        let obs = if i % 2 == 0 {
            // Position observation (model_index=0)
            bh::estimation::Observation::new(obs_epoch, truth_st.rows(0, 3).into_owned(), 0)
        } else {
            // Range observation (model_index=1)
            let true_range = (truth_st.rows(0, 3) - &station).norm();
            bh::estimation::Observation::new(obs_epoch, DVector::from_vec(vec![true_range]), 1)
        };

        let record = ekf.process_observation(&obs).unwrap();
        println!("  {:20} prefit residual norm: {:.3}",
            record.measurement_name, record.prefit_residual.norm());
    }

    // Summary
    use bh::propagators::traits::DStatePropagator;
    truth_prop.propagate_to(ekf.current_epoch());
    let pos_error = (ekf.current_state().rows(0, 3) - truth_prop.current_state().rows(0, 3)).norm();
    println!("\nFinal position error: {:.2} m", pos_error);
    println!("Records: {} (InertialPosition: {}, Range: {})",
        ekf.records().len(),
        ekf.records().iter().filter(|r| r.measurement_name == "InertialPosition").count(),
        ekf.records().iter().filter(|r| r.measurement_name == "Range").count());
}
