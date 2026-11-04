//! Read an SP3-d precise orbit file and inspect the first record of satellite G01.

#[allow(unused_imports)]
use brahe as bh;
use brahe::datafiles::sp3::{SP3, SP3SatelliteId};

fn main() {
    let sp3 = SP3::from_file("test_assets/sp3/example-d-1.sp3").unwrap();
    let header = sp3.header();

    println!("Version:     {}", header.version);
    println!("Agency:      {}", header.agency);
    println!("Time system: {}", header.time_system);
    println!("Satellites:  {}", sp3.num_satellites());
    println!("Epochs:      {}", sp3.num_epochs());
    println!("Interval:    {:.1} s", header.epoch_interval);
    println!("Span:        {} to {}", sp3.start_epoch(), sp3.end_epoch());

    let g01 = SP3SatelliteId::parse("G01").unwrap();
    let record = sp3.record(g01, 0).unwrap().unwrap();
    println!(
        "G01 position (m):    [{:.3}, {:.3}, {:.3}]",
        record.position[0], record.position[1], record.position[2]
    );
    println!("G01 clock (us):      {:.6}", record.clock.unwrap() * 1e6);
    println!("G01 velocity:        {:?}", record.velocity);

    // A zero position is the format's bad-value marker and is read as absent
    println!("G01 at epoch 1:      {:?}", sp3.record(g01, 1).unwrap());
}
