use brahe::time::{Epoch, TimeSystem};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct DateTimeParams {
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
}

fn parse_datetimes(params: &serde_json::Value) -> Vec<DateTimeParams> {
    serde_json::from_value(params["datetimes"].clone()).unwrap()
}

fn make_epochs(dts: &[DateTimeParams]) -> Vec<Epoch> {
    dts.iter()
        .map(|dt| {
            Epoch::from_datetime(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                dt.nanosecond,
                TimeSystem::UTC,
            )
        })
        .collect()
}

pub fn epoch_creation(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(dts.len());

        for dt in &dts {
            let epc = Epoch::from_datetime(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                dt.nanosecond,
                TimeSystem::UTC,
            );
            results.push(epc.jd());
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn utc_to_tai(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);
    let epochs = make_epochs(&dts);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            results.push(epc.jd_as_time_system(TimeSystem::TAI));
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn utc_to_tt(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);
    let epochs = make_epochs(&dts);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            results.push(epc.jd_as_time_system(TimeSystem::TT));
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn utc_to_gps(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);
    let epochs = make_epochs(&dts);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            results.push(epc.jd_as_time_system(TimeSystem::GPS));
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn utc_to_ut1(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);
    let epochs = make_epochs(&dts);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            results.push(epc.jd_as_time_system(TimeSystem::UT1));
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
