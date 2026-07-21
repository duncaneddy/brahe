/*!
 * JPL Horizons SPK client.
 *
 * Generates, caches, and loads targeted SPK (`.bsp`) kernels for small bodies
 * without packaged DE-kernel coverage via the Horizons API
 * (`https://ssd.jpl.nasa.gov/api/horizons.api`).
 */

pub mod request;
pub mod response;

pub use request::HorizonsSPKRequest;
pub use response::HorizonsSPKResponse;
