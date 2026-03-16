use brahe::access::{
    location_accesses, AccessSearchConfig, ElevationConstraint, PointLocation,
};
use brahe::propagators::SGPPropagator;
use brahe::propagators::traits::SStatePropagator;
use std::time::Instant;

pub fn sgp4_access(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let line1: String = serde_json::from_value(params["line1"].clone()).unwrap();
    let line2: String = serde_json::from_value(params["line2"].clone()).unwrap();
    let min_el: f64 = serde_json::from_value(params["min_elevation_deg"].clone()).unwrap();
    let duration: f64 =
        serde_json::from_value(params["search_duration_seconds"].clone()).unwrap();

    #[derive(serde::Deserialize)]
    struct Location {
        lon: f64,
        lat: f64,
        alt: f64,
    }
    let locations: Vec<Location> =
        serde_json::from_value(params["locations"].clone()).unwrap();

    let prop = SGPPropagator::from_tle(&line1, &line2, 60.0).unwrap();
    let search_start = prop.initial_epoch();
    let search_end = search_start + duration;

    let constraint = ElevationConstraint::new(Some(min_el), None).unwrap();
    let config = AccessSearchConfig {
        parallel: false,
        ..Default::default()
    };

    // Build PointLocation objects
    let point_locations: Vec<PointLocation> = locations
        .iter()
        .map(|loc| PointLocation::new(loc.lon, loc.lat, loc.alt))
        .collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<serde_json::Value>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut all_windows: Vec<Vec<serde_json::Value>> =
            Vec::with_capacity(point_locations.len());

        for pl in &point_locations {
            let windows = location_accesses(
                pl,
                &prop,
                search_start,
                search_end,
                &constraint,
                None,
                Some(&config),
            )
            .unwrap();

            let loc_windows: Vec<serde_json::Value> = windows
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "start_jd": w.window_open.jd(),
                        "end_jd": w.window_close.jd(),
                    })
                })
                .collect();
            all_windows.push(loc_windows);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = all_windows;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
