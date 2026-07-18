//! Convert the Hipparcos catalog to a Polars DataFrame, and convert a
//! magnitude-filtered catalog to inspect its reduced row count.
//!
//! FLAGS = ["NETWORK"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::star_catalogs;

fn main() {
    let hipparcos = star_catalogs::get_hipparcos_catalog(None).unwrap();

    let df = hipparcos.to_dataframe().unwrap();
    println!("DataFrame shape: {:?}", df.shape());
    println!("Columns: {:?}", df.get_column_names());

    // The `polars` dependency here is built with only the `lazy` feature and
    // isn't itself a nameable crate for this example, so filter with the
    // catalog's own `filter_by_magnitude` before exporting, rather than the
    // Python example's Polars-level `.str.contains()` expression.
    let bright = hipparcos.filter_by_magnitude(4.0);
    let bright_df = bright.to_dataframe().unwrap();
    println!("Stars brighter than Vmag 4.0: {}", bright_df.height());
}
