//! List the IGS precise products published for one GPS week and filter them by center, solution, and content.
//!
//! Uses the Garner (SIO) mirror, which needs no credentials. Garner is served over
//! plain HTTP, so the downloaded files are not transport-secured.

// FLAGS = ["NETWORK"]

use brahe as bh;
use bh::igs::{
    IGSContentType, IGSFileFormat, IGSProductFilter, IGSProductSource, IGSProductsClient,
    IGSSolutionType,
};
use bh::time::{Epoch, TimeSystem};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = IGSProductsClient::with_source(IGSProductSource::Garner);
    println!("Source: {}, base URL: {}", client.source(), client.base_url());

    // Any epoch inside the week selects it; this one falls in GPS week 2340
    let epoch = Epoch::from_datetime(2024, 11, 12, 12, 0, 0.0, 0.0, TimeSystem::GPS);
    println!("Files published for the week: {}", client.list_files(epoch)?.len());

    // Final orbit products in SP3 format from the CODE and IGS analysis centers
    for center in ["COD", "IGS"] {
        let filter = IGSProductFilter::new()
            .with_center(center)
            .with_solution(IGSSolutionType::Final)
            .with_content(IGSContentType::Orbit)
            .with_format(IGSFileFormat::SP3);
        let products = client.list_products(epoch, &filter)?;
        println!("\n{center} final SP3 orbits: {}", products.len());
        for product in products.iter().take(3) {
            let name = &product.file_name;
            println!(
                "  {}: starts {}, span {:.0} h, sampling {:.0} s",
                product.name,
                name.start_epoch(),
                name.span().unwrap() / 3600.0,
                name.sampling().unwrap()
            );
        }
    }

    Ok(())
}
