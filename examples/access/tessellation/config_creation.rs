//! Create tessellation configurations with default and custom parameters.
//! Demonstrates the builder pattern for OrbitGeometryTessellatorConfig.

#[allow(unused_imports)]
use brahe as bh;
use bh::access::constraints::AscDsc;
use bh::access::tessellation::OrbitGeometryTessellatorConfig;

fn main() {
    bh::initialize_eop().unwrap();

    // Default configuration
    let config = OrbitGeometryTessellatorConfig::default();
    println!("Default image_width: {} m", config.image_width);
    println!("Default image_length: {} m", config.image_length);
    println!("Default crosstrack_overlap: {} m", config.crosstrack_overlap);
    println!("Default alongtrack_overlap: {} m", config.alongtrack_overlap);
    println!("Default min_image_length: {} m", config.min_image_length);
    println!("Default max_image_length: {} m", config.max_image_length);

    // Custom configuration for ascending passes with larger tiles
    let custom_config = OrbitGeometryTessellatorConfig::new(10000.0, 15000.0)
        .with_crosstrack_overlap(300.0)
        .with_alongtrack_overlap(300.0)
        .with_asc_dsc(AscDsc::Ascending)
        .with_min_image_length(8000.0)
        .with_max_image_length(25000.0);
    println!("\nCustom image_width: {} m", custom_config.image_width);
    println!("Custom image_length: {} m", custom_config.image_length);
    println!("Custom asc_dsc: {:?}", custom_config.asc_dsc);

}

