use brahe::AngleFormat;
use brahe::coordinates::state_koe_to_eci;
use brahe::integrators::IntegratorConfig;
use brahe::propagators::traits::{DStatePropagator, SStatePropagator, SStateProvider};
use brahe::propagators::{
    AtmosphericModel, DNumericalOrbitPropagator, DragConfiguration, EclipseModel,
    EphemerisSource, ForceModelConfig, GravityConfiguration, GravityModelSource,
    IntegratorMethod, KeplerianPropagator, NumericalPropagationConfig, ParameterSource,
    SGPPropagator, SolarRadiationPressureConfiguration, ThirdBody, ThirdBodyConfiguration,
};
use brahe::traits::DOrbitStateProvider;
use brahe::time::{Epoch, TimeSystem};
use brahe::TrajectoryMode;
use nalgebra::{DVector, SVector};
use std::time::Instant;

pub fn keplerian_single(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    #[derive(serde::Deserialize)]
    struct Case {
        jd: f64,
        elements: Vec<f64>,
        dt: f64,
    }
    let cases: Vec<Case> = serde_json::from_value(params["cases"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(cases.len());

        for case in &cases {
            let epc = Epoch::from_jd(case.jd, TimeSystem::UTC);
            let oe = SVector::<f64, 6>::new(
                case.elements[0],
                case.elements[1],
                case.elements[2],
                case.elements[3],
                case.elements[4],
                case.elements[5],
            );
            let target = epc + case.dt;

            let prop = KeplerianPropagator::from_keplerian(epc, oe, AngleFormat::Degrees, 60.0);
            let state = DOrbitStateProvider::state_eci(&prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn keplerian_trajectory(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
    let elements: Vec<f64> = serde_json::from_value(params["elements"].clone()).unwrap();
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    let epc = Epoch::from_jd(jd, TimeSystem::UTC);
    let oe = SVector::<f64, 6>::new(
        elements[0],
        elements[1],
        elements[2],
        elements[3],
        elements[4],
        elements[5],
    );

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let prop = KeplerianPropagator::from_keplerian(epc, oe, AngleFormat::Degrees, step_size);
        let mut results = Vec::with_capacity(n_steps);

        for step_idx in 0..n_steps {
            let target = epc + (step_idx as f64 + 1.0) * step_size;
            let state = DOrbitStateProvider::state_eci(&prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn sgp4_single(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let line1: String = serde_json::from_value(params["line1"].clone()).unwrap();
    let line2: String = serde_json::from_value(params["line2"].clone()).unwrap();
    let offsets: Vec<f64> = serde_json::from_value(params["time_offsets_seconds"].clone()).unwrap();

    let base_prop = SGPPropagator::from_tle(&line1, &line2, 60.0).unwrap();
    let base_epoch = base_prop.initial_epoch();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(offsets.len());

        for dt in &offsets {
            let target = base_epoch + *dt;
            // Use SStateProvider::state() to get TEME output directly (matches Java's TEME frame)
            let state = SStateProvider::state(&base_prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn sgp4_trajectory(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let line1: String = serde_json::from_value(params["line1"].clone()).unwrap();
    let line2: String = serde_json::from_value(params["line2"].clone()).unwrap();
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    let prop = SGPPropagator::from_tle(&line1, &line2, step_size).unwrap();
    let base_epoch = prop.initial_epoch();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(n_steps);

        for step_idx in 0..n_steps {
            let target = base_epoch + (step_idx as f64 + 1.0) * step_size;
            // Use SStateProvider::state() to get TEME output directly (matches Java's TEME frame)
            let state = SStateProvider::state(&prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn numerical_twobody(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
    let elements: Vec<f64> = serde_json::from_value(params["elements"].clone()).unwrap();
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    let epc = Epoch::from_jd(jd, TimeSystem::UTC);
    let oe = SVector::<f64, 6>::new(
        elements[0],
        elements[1],
        elements[2],
        elements[3],
        elements[4],
        elements[5],
    );
    let cart = state_koe_to_eci(oe, AngleFormat::Degrees);
    let state_dv = DVector::from_vec(vec![cart[0], cart[1], cart[2], cart[3], cart[4], cart[5]]);

    let prop_config = NumericalPropagationConfig::default();
    let force_config = ForceModelConfig::two_body_gravity();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut prop = DNumericalOrbitPropagator::new(
            epc,
            state_dv.clone(),
            prop_config.clone(),
            force_config.clone(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        prop.set_trajectory_mode(TrajectoryMode::Disabled);

        let mut results = Vec::with_capacity(n_steps);
        for step_idx in 0..n_steps {
            let target = epc + (step_idx as f64 + 1.0) * step_size;
            prop.propagate_to(target);
            let state = prop.current_state();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

#[derive(serde::Deserialize)]
struct Rk4ForceParams {
    jd: f64,
    elements_deg: Vec<f64>,
    step_size: f64,
    n_steps: usize,
    params: Vec<f64>,
    gravity_degree: usize,
    gravity_order: usize,
    #[serde(default)]
    third_body_sun: bool,
    #[serde(default)]
    third_body_moon: bool,
    #[serde(default)]
    drag: bool,
    #[serde(default)]
    srp: bool,
}

fn force_config_from_params(p: &Rk4ForceParams) -> ForceModelConfig {
    let gravity = GravityConfiguration::SphericalHarmonic {
        source: GravityModelSource::default(),
        degree: p.gravity_degree,
        order: p.gravity_order,
    };

    let mut bodies = Vec::new();
    if p.third_body_sun {
        bodies.push(ThirdBody::Sun);
    }
    if p.third_body_moon {
        bodies.push(ThirdBody::Moon);
    }
    let third_body = if !bodies.is_empty() {
        Some(ThirdBodyConfiguration {
            ephemeris_source: EphemerisSource::DE440s,
            bodies,
        })
    } else {
        None
    };

    let drag = if p.drag {
        Some(DragConfiguration {
            model: AtmosphericModel::NRLMSISE00,
            area: ParameterSource::ParameterIndex(1),
            cd: ParameterSource::ParameterIndex(2),
        })
    } else {
        None
    };

    let srp = if p.srp {
        Some(SolarRadiationPressureConfiguration {
            area: ParameterSource::ParameterIndex(3),
            cr: ParameterSource::ParameterIndex(4),
            eclipse_model: EclipseModel::Conical,
        })
    } else {
        None
    };

    let mass = if drag.is_some() || srp.is_some() {
        Some(ParameterSource::ParameterIndex(0))
    } else {
        None
    };

    ForceModelConfig {
        gravity,
        drag,
        srp,
        third_body,
        relativity: false,
        mass,
        frame_transform: Default::default(),
    }
}

fn numerical_rk4_run(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let p: Rk4ForceParams = serde_json::from_value(params.clone()).unwrap();

    let epc = Epoch::from_jd(p.jd, TimeSystem::UTC);
    let oe = SVector::<f64, 6>::new(
        p.elements_deg[0],
        p.elements_deg[1],
        p.elements_deg[2],
        p.elements_deg[3],
        p.elements_deg[4],
        p.elements_deg[5],
    );
    let cart = state_koe_to_eci(oe, AngleFormat::Degrees);
    let state_dv = DVector::from_vec(vec![cart[0], cart[1], cart[2], cart[3], cart[4], cart[5]]);
    let param_vec = DVector::from_vec(p.params.clone());

    let force_config = force_config_from_params(&p);
    let prop_config = NumericalPropagationConfig::new(
        IntegratorMethod::RK4,
        IntegratorConfig::fixed_step(p.step_size),
        Default::default(),
    );

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut prop = DNumericalOrbitPropagator::new(
            epc,
            state_dv.clone(),
            prop_config.clone(),
            force_config.clone(),
            Some(param_vec.clone()),
            None,
            None,
            None,
        )
        .unwrap();
        prop.set_trajectory_mode(TrajectoryMode::Disabled);

        let mut results = Vec::with_capacity(p.n_steps);
        // Use step_by rather than propagate_to: with fixed-step RK4,
        // propagate_to can hit a float-drift path in brahe that
        // permanently shrinks dt_next when the target epoch isn't exactly
        // representable as a multiple of step_size.
        for _ in 0..p.n_steps {
            prop.step_by(p.step_size);
            let state = prop.current_state();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn numerical_rk4_grav5x5(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    numerical_rk4_run(params, iterations)
}

pub fn numerical_rk4_grav20x20_sun_moon(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    numerical_rk4_run(params, iterations)
}

pub fn numerical_rk4_grav80x80_full(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    numerical_rk4_run(params, iterations)
}
