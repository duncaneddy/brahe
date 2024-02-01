/*!
 * Provides the `TimeSeries` struct, a custom iterator for `Epoch` that allows for direct iteration over
 * a range of `Epoch`s. The iteration can be in the positive (forward) or negative (backward) direction.
 */

use crate::time::epoch::Epoch;

// TimeSeries

/// `TimeSeries` is a custom iterator that enables direct iteration times between
/// two `Epoch`s. The iteration can either be in the positive (forward) or negative
/// (backward) direction.
///
/// The `TimeSeries` iterator will return a new `Epoch` for each iteration it is
/// called. The iteration is exclusive so the `epoch_end` will not be reached.
/// The last value will be one whole or partial step from the iterator end.
pub struct TimeSeries {
    epoch_current: Epoch,
    epoch_end: Epoch,
    step: f64,
    positive_step: bool,
}

impl TimeSeries {
    /// Create an `Epoch` from a Julian date and time system. The time system is needed
    /// to make the instant unambiguous.
    ///
    /// # Arguments
    /// - `jd`: Julian date as a floating point number
    /// - `eop` Earth orientation data loading structure.
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // Epochs specifying start and end of iteration
    /// let epcs = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
    /// let epcf = Epoch::from_datetime(2022, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::TAI);
    ///
    /// // Vector to confirm equivalence of iterator to addition of time
    /// let mut epc = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
    ///
    /// // Use `TimeSeries` iterator to generate Epochs over range
    /// for e in TimeSeries::new(epcs, epcf, 1.0) {
    ///     assert_eq!(epc, e);
    ///     epc += 1;
    /// }
    /// ```
    pub fn new(epoch_start: Epoch, epoch_end: Epoch, step: f64) -> Self {
        Self {
            epoch_current: epoch_start.clone(),
            epoch_end,
            step: step.abs(),
            positive_step: epoch_end > epoch_start,
        }
    }
}

impl Iterator for TimeSeries {
    type Item = Epoch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.epoch_end != self.epoch_current {
            // Grab current epoch to return prior to advancing
            let epc = self.epoch_current.clone();

            let rem = (self.epoch_end - self.epoch_current).abs();
            let h = if self.step < rem { self.step } else { rem };

            if self.positive_step {
                self.epoch_current += h;
            } else {
                self.epoch_current -= h;
            }

            Some(epc)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::testing::setup_global_test_eop;

    use crate::time::epoch::Epoch;
    use crate::time::time_types::TimeSystem;
    use super::*;

    #[test]
    fn test_time_series() {
        setup_global_test_eop();

        let mut epcv: Vec<Epoch> = Vec::new();
        let epcs = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let epcf = Epoch::from_datetime(2022, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let mut epc = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);

        for e in TimeSeries::new(epcs, epcf, 1.0) {
            assert_eq!(epc, e);
            epc += 1;
            epcv.push(e);
        }

        let epcl = Epoch::from_datetime(2022, 1, 1, 23, 59, 59.0, 0.0, TimeSystem::TAI);
        assert_eq!(epcv.len(), 86400);
        assert_eq!(epcv[epcv.len() - 1] != epcf, true);
        assert!((epcv[epcv.len() - 1] - epcl).abs() < 1.0e-9);
    }

    #[test]
    fn test_time_series_negative() {
        setup_global_test_eop();

        let mut epcv: Vec<Epoch> = Vec::new();
        let epcs = Epoch::from_datetime(2022, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let epcf = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let mut epc = Epoch::from_datetime(2022, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::TAI);

        for e in TimeSeries::new(epcs, epcf, 1.0) {
            assert_eq!(epc, e);
            epc -= 1;
            epcv.push(e);
        }

        let epcl = Epoch::from_datetime(2022, 1, 1, 0, 0, 1.0, 0.0, TimeSystem::TAI);
        assert_eq!(epcv.len(), 86400);
        assert_eq!(epcv[epcv.len() - 1] != epcf, true);
        assert!((epcv[epcv.len() - 1] - epcl).abs() < 1.0e-9);
    }
}