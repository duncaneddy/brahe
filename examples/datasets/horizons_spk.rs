//! Generate and load a targeted SPK from JPL Horizons.
//!
//! Requests an SPK for Ceres (SPK-ID 20000001) over a time span, loads it
//! into the SPICE registry, and prints the cached kernel path. The .bsp is
//! cached under $BRAHE_CACHE/horizons and reused for the same request on
//! repeat runs.
//!
//! FLAGS = ["NETWORK"]

use brahe as bh;
use bh::datasets::horizons::{HorizonsClient, HorizonsSPKRequest};
use bh::time::{Epoch, TimeSystem};

fn main() {
    let t0 = Epoch::from_datetime(2015, 12, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
    let t1 = Epoch::from_datetime(2016, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);

    let request = HorizonsSPKRequest::for_spkid(20000001, t0, t1);
    let response = HorizonsClient::new().get_spk(&request).unwrap();

    response.load().unwrap();
    println!("Cached at: {}", response.path().display());
}
