//! Build an SP3-d file in memory, write it to text, and read it back.

#[allow(unused_imports)]
use brahe as bh;
use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
use brahe::time::{Epoch, TimeSystem};

fn main() {
    let g01 = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    let g02 = SP3SatelliteId::new(GNSSSystem::GPS, 2).unwrap();

    let options = SP3WriteOptions::new()
        .with_agency("BRHE")
        .with_comments(vec!["Example orbit file".to_string()]);
    let mut sp3 = SP3::new(vec![g01, g02], 900.0, options).unwrap();

    let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    for k in 0..3 {
        let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
        let record = SP3Record::new([2.0e7 + 1.0e3 * k as f64, 1.0e7, 1.0e7])
            .with_velocity([100.0, 2.0e3, -5.0e2])
            .with_clock(2.5e-5);
        sp3.set_record(g01, index, record).unwrap();
    }

    // G02 is left absent at every epoch except the first
    sp3.set_record(g02, 0, SP3Record::new([-1.5e7, 2.0e7, 5.0e6])).unwrap();

    let text = sp3.to_string().unwrap();
    let lines: Vec<&str> = text.lines().collect();
    println!("{}", lines[..3].join("\n"));
    println!("...");
    println!("{}", lines[lines.len() - 6..].join("\n"));

    let parsed = SP3::from_str(&text).unwrap();
    println!(
        "Read back {} satellites and {} epochs",
        parsed.num_satellites(),
        parsed.num_epochs()
    );
    println!("G02 at epoch 1: {:?}", parsed.record(g02, 1).unwrap());
}
