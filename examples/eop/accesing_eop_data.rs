//! Initialize EOP Providers with simpliest way possible

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::now();
    let (xp, yp, dut1, lod, dX, dY) = bh::get_global_eop(epc.mjd()).unwrap();

    println!("At epoch {}:", epc.to_string());
    println!("xp:  {}", xp);
    println!("yp:  {}", yp);
    println!("dut1: {}", dut1);
    println!("lod:  {}", lod);
    println!("dX:   {}", dX);
    println!("dY:   {}", dY);
}
