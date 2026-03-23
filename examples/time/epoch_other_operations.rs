//! Equality and comparison operations with Epoch instances

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc_1 = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc_2 = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 1.0, bh::TimeSystem::UTC);
    let epc_3 = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // You can compare two Epoch instances for equality
    println!("epc_1 == epc_2: {}", epc_1 == epc_2);
    println!("epc_1 == epc_3: {}", epc_1 == epc_3);

    // You can also use inequality and comparison operators
    println!("epc_1 != epc_2: {}", epc_1 != epc_2);
    println!("epc_1 < epc_2: {}", epc_1 < epc_2);
    println!("epc_2 < epc_1: {}", epc_2 < epc_1);
    println!("epc_2 > epc_1: {}", epc_2 > epc_1);
    println!("epc_1 <= epc_3: {}", epc_1 <= epc_3);
    println!("epc_2 >= epc_1: {}", epc_2 >= epc_1);
}

