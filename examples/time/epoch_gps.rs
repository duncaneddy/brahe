//! Create Epoch instances from GPS week and seconds

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch from GPS week and seconds
    // Week 2390, day 2 (October 28, 2025)
    let week = 2390;
    let seconds = 2.0 * 86400.0; // 3 days + 12 hours
    let epc1 = bh::Epoch::from_gps_date(week, seconds);
    println!("GPS Week {}, Seconds {}: {}", week, seconds, epc1);
    // GPS Week 2390, Seconds 172800: 2025-10-28 00:00:00.000 GPS

    // Verify round-trip conversion
    let (week_out, sec_out) = epc1.gps_date();
    println!("Round-trip: Week {}, Seconds {:.1}", week_out, sec_out);
    // Round-trip: Week 2390, Seconds 172800.0

    // Create from GPS seconds since GPS epoch
    let gps_seconds = week as f64 * 7.0 * 86400.0 + seconds;
    let epc2 = bh::Epoch::from_gps_seconds(gps_seconds);
    println!("GPS Seconds {}: {}", gps_seconds, epc2);
    // 1445644800: 2025-10-28 00:00:00.000 GPS
}
