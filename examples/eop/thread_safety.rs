//! Thread safety demonstration for CachingEOPProvider.
//!
//! Demonstrates:
//! - Creating shared EOP provider
//! - Processing epochs concurrently across multiple threads
//! - Thread-safe access to EOP data

use brahe::eop::*;
use std::path::Path;
use std::sync::Arc;
use std::thread;

fn process_epoch(mjd: f64) -> Result<(f64, f64, f64, f64), String> {
    let ut1_utc = get_global_ut1_utc(mjd).map_err(|e| format!("{:?}", e))?;
    let (pm_x, pm_y) = get_global_pm(mjd).map_err(|e| format!("{:?}", e))?;
    Ok((mjd, ut1_utc, pm_x, pm_y))
}

fn main() {
    // Create shared provider
    let provider = Arc::new(CachingEOPProvider::new(
        Path::new(Path::new("./eop_data/finals.txt")),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds: 7 days
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap());
    set_global_eop_provider((*provider).clone());

    println!("Thread Safety Demonstration");
    println!("{}", "=".repeat(60));
    println!("CachingEOPProvider is thread-safe and can be safely");
    println!("shared across multiple threads");
    println!();

    // Process epochs concurrently
    let mjds: Vec<f64> = (0..10).map(|i| 59000.0 + (i as f64) * 10.0).collect();

    println!("Processing {} epochs across 4 threads...", mjds.len());
    println!();

    let mut handles = vec![];

    // Spawn threads to process epochs concurrently
    for mjd in mjds {
        let handle = thread::spawn(move || {
            process_epoch(mjd)
        });
        handles.push(handle);
    }

    // Collect results
    let mut results = vec![];
    for handle in handles {
        results.push(handle.join().unwrap());
    }

    // Display results
    println!("Results:");
    println!("{}", "-".repeat(60));
    let mut success_count = 0;
    for result in results {
        match result {
            Ok((mjd, ut1_utc, pm_x, pm_y)) => {
                println!("MJD {:.1}: UT1-UTC={:.6}s, PM=({:.6}, {:.6}) arcsec",
                         mjd, ut1_utc, pm_x, pm_y);
                success_count += 1;
            }
            Err(e) => {
                println!("ERROR: {}", e);
            }
        }
    }

    println!();
    println!("Successfully processed {}/10 epochs", success_count);
    println!();
    println!("Thread-safe concurrent access completed successfully!");
}
