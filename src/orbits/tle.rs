/*!
*
* The `tle` module provides functionality for working with NORAD Two-Line Element (TLE) data.
*
*/

use crate::orbits::traits::{OrbitalState, OrbitalStateInterpolator};
use crate::time::Epoch;
use nalgebra as na;
use sgp;

struct TLE {
    pub epoch: Epoch,
    pub line1: String,
    pub line2: String,
    elements: sgp::Elements,
    constants: sgp::Constants,
}

impl TLE {
    /// Creates a new TLE from the given epoch and two lines of TLE data.
    ///
    /// # Arguments
    /// * `line1` - The first line of the TLE data.
    /// * `line2` - The second line of the TLE data.
    pub fn new(line1: String, line2: String) -> Self {
        let elements = sgp::Elements::from_tle(&line1, &line2).unwrap();
        let constants = sgp::Constants::from_elements(&elements)?;
        TLE {
            epoch,
            line1,
            line2,
            elements,
            constants,
        }
    }
}
