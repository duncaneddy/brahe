//! Time-system conversion benchmarks. Uses hifitime directly (no Nyx wrapper)
//! because Nyx adds no value for raw timescale conversions and direct usage
//! locks the version dependency explicitly.

use hifitime::{Duration, Epoch, Unit};
use hifitime::ut1::Ut1Provider;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct DateTimeParams {
    year: i32,
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

/// Construct a hifitime `Epoch` from broken-down UTC components.
///
/// The input gives `second: f64` and `nanosecond: f64`. hifitime's
/// `from_gregorian_utc` requires integer seconds in [0, 59] and nanos < 1e9.
/// Rather than carry overflow through minute/hour/day boundaries manually,
/// we build the epoch at the truncated integer second with zero nanos and then
/// add the combined sub-second residual as a `Duration`. This avoids any
/// boundary carry logic entirely.
fn make_epoch(dt: &DateTimeParams) -> Epoch {
    let int_sec = dt.second.floor() as u8;
    let frac_ns = (dt.second - dt.second.floor()) * 1.0e9;
    let total_sub_ns = frac_ns + dt.nanosecond;
    let base = Epoch::from_gregorian_utc(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        int_sec,
        0,
    );
    base + Duration::from_nanoseconds(total_sub_ns)
}

pub fn utc_to_tai(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);
    let epochs: Vec<Epoch> = dts.iter().map(make_epoch).collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<f64> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            results.push(epc.to_jde_tai_days());
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn epoch_creation(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let dts = parse_datetimes(params);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<f64> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(dts.len());

        for dt in &dts {
            // Construct the epoch and report its UTC JD — both construction
            // and the JD extraction are inside the timed region, matching
            // the brahe-Rust reference implementation.
            let epc = make_epoch(dt);
            results.push(epc.to_jde_utc_days());
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
    let epochs: Vec<Epoch> = dts.iter().map(make_epoch).collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<f64> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            results.push(epc.to_jde_tt_days());
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
    let epochs: Vec<Epoch> = dts.iter().map(make_epoch).collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<f64> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            // hifitime 4.3 has no to_jde_gpst_days(). GPS = TAI - 19 s, so
            // derive JD in GPST from the TAI JDE by subtracting 19 seconds.
            results.push(epc.to_jde_tai_days() - 19.0 / 86400.0);
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
    let epochs: Vec<Epoch> = dts.iter().map(make_epoch).collect();

    // hifitime 4.3 UT1 requires a Ut1Provider (the `ut1` feature). An empty
    // default provider has zero DUT1 records, so UT1 ≈ UTC. Until Task 9
    // wires in EOP, this produces UT1 JD == UTC JD (matching brahe's
    // zero-EOP behavior).
    //
    // Compute UT1 JD as: UTC_JD + DUT1_days.
    // `ut1_offset` returns TAI - UT1; UTC ≈ TAI - leap_seconds, so
    // DUT1 = UTC - UT1 = -(TAI - UT1 - leap_seconds) ≈ -(offset - leaps).
    // With an empty provider, ut1_offset returns None → DUT1 = 0 → UT1 JD == UTC JD.
    //
    // Previous approach (`to_ut1_duration() + 2415020.0`) was wrong because
    // `to_ut1_duration` returns seconds since J1900 *midnight* (JD 2415020.0),
    // but the JDE reference is J1900 *noon* (JD 2415020.5) — a 0.5-day error.
    let provider = Ut1Provider::default();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<f64> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epochs.len());

        for epc in &epochs {
            // ut1_offset = TAI - UT1. UTC = TAI - leap_seconds, so
            // UT1 = UTC + (leap_seconds - ut1_offset).
            // Equivalently: UT1_JD = UTC_JD + (utc_tai_offset - ut1_tai_offset) / 86400
            // With empty provider, ut1_offset is None → DUT1 ≈ 0 → UT1 JD ≡ UTC JD.
            let utc_jd = epc.to_jde_utc_days();
            let dut1_days = match epc.ut1_offset(&provider) {
                Some(tai_minus_ut1) => {
                    // DUT1 = UTC - UT1 = -(TAI-UT1) + (TAI-UTC)
                    // TAI-UTC = leap seconds (from hifitime utc duration difference)
                    let tai_minus_utc = epc.to_tai_duration() - epc.to_utc_duration();
                    let dut1 = tai_minus_utc - tai_minus_ut1;
                    dut1.to_unit(Unit::Day)
                }
                None => 0.0,
            };
            results.push(utc_jd + dut1_days);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
