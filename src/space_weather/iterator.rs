/*!
 * Iterators for space weather data.
 */

use std::sync::Arc;

use crate::space_weather::provider::SpaceWeatherProvider;
use crate::time::Epoch;
use crate::utils::BraheError;

/// Iterator over 3-hourly Kp values within a time range.
///
/// Each iteration returns a tuple of (Epoch, Kp value) where the Epoch
/// is at the start of the 3-hour UT interval.
///
/// # Example
///
/// ```
/// use brahe::space_weather::{FileSpaceWeatherProvider, SpaceWeatherKpIterator};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
/// let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let end = Epoch::from_datetime(2023, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let iter = SpaceWeatherKpIterator::new(provider, start, end).unwrap();
/// for (epoch, kp) in iter {
///     println!("Epoch: {}, Kp: {}", epoch, kp);
/// }
/// ```
pub struct SpaceWeatherKpIterator<P: SpaceWeatherProvider> {
    provider: Arc<P>,
    current_mjd: f64,
    current_index: usize,
    end_mjd: f64,
}

impl<P: SpaceWeatherProvider> SpaceWeatherKpIterator<P> {
    /// Create a new Kp iterator over the specified time range.
    ///
    /// # Arguments
    /// - `provider`: Space weather provider to iterate over
    /// - `start_epoch`: Start of the iteration range
    /// - `end_epoch`: End of the iteration range
    ///
    /// # Returns
    /// - Iterator over (Epoch, Kp) tuples
    pub fn new(provider: P, start_epoch: Epoch, end_epoch: Epoch) -> Result<Self, BraheError> {
        let start_mjd = start_epoch.mjd();
        let end_mjd = end_epoch.mjd();

        if start_mjd > end_mjd {
            return Err(BraheError::SpaceWeatherError(
                "Start epoch must be before end epoch".to_string(),
            ));
        }

        // Calculate the starting interval
        let start_floor = start_mjd.floor();
        let fraction = start_mjd - start_floor;
        let hours = fraction * 24.0;
        let index = (hours / 3.0).floor() as usize;

        Ok(Self {
            provider: Arc::new(provider),
            current_mjd: start_floor,
            current_index: index.min(7),
            end_mjd,
        })
    }

    /// Get the current MJD at the start of the current 3-hour interval.
    fn current_interval_mjd(&self) -> f64 {
        self.current_mjd + (self.current_index as f64 * 3.0 / 24.0)
    }
}

impl<P: SpaceWeatherProvider> Iterator for SpaceWeatherKpIterator<P> {
    type Item = (Epoch, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let interval_mjd = self.current_interval_mjd();

        // Check if we've reached or passed the end (end is exclusive)
        if interval_mjd >= self.end_mjd {
            return None;
        }

        // Get the Kp value for this interval
        let kp = match self.provider.get_kp(interval_mjd) {
            Ok(kp) => kp,
            Err(_) => return None,
        };

        // Create epoch at start of this interval
        let epoch = Epoch::from_mjd(interval_mjd, crate::time::TimeSystem::UTC);

        // Advance to next interval
        self.current_index += 1;
        if self.current_index > 7 {
            self.current_index = 0;
            self.current_mjd += 1.0;
        }

        Some((epoch, kp))
    }
}

/// Iterator over 3-hourly Ap values within a time range.
///
/// Each iteration returns a tuple of (Epoch, Ap value) where the Epoch
/// is at the start of the 3-hour UT interval.
///
/// # Example
///
/// ```
/// use brahe::space_weather::{FileSpaceWeatherProvider, SpaceWeatherApIterator};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
/// let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let end = Epoch::from_datetime(2023, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let iter = SpaceWeatherApIterator::new(provider, start, end).unwrap();
/// for (epoch, ap) in iter {
///     println!("Epoch: {}, Ap: {}", epoch, ap);
/// }
/// ```
pub struct SpaceWeatherApIterator<P: SpaceWeatherProvider> {
    provider: Arc<P>,
    current_mjd: f64,
    current_index: usize,
    end_mjd: f64,
}

impl<P: SpaceWeatherProvider> SpaceWeatherApIterator<P> {
    /// Create a new Ap iterator over the specified time range.
    ///
    /// # Arguments
    /// - `provider`: Space weather provider to iterate over
    /// - `start_epoch`: Start of the iteration range
    /// - `end_epoch`: End of the iteration range
    ///
    /// # Returns
    /// - Iterator over (Epoch, Ap) tuples
    pub fn new(provider: P, start_epoch: Epoch, end_epoch: Epoch) -> Result<Self, BraheError> {
        let start_mjd = start_epoch.mjd();
        let end_mjd = end_epoch.mjd();

        if start_mjd > end_mjd {
            return Err(BraheError::SpaceWeatherError(
                "Start epoch must be before end epoch".to_string(),
            ));
        }

        // Calculate the starting interval
        let start_floor = start_mjd.floor();
        let fraction = start_mjd - start_floor;
        let hours = fraction * 24.0;
        let index = (hours / 3.0).floor() as usize;

        Ok(Self {
            provider: Arc::new(provider),
            current_mjd: start_floor,
            current_index: index.min(7),
            end_mjd,
        })
    }

    /// Get the current MJD at the start of the current 3-hour interval.
    fn current_interval_mjd(&self) -> f64 {
        self.current_mjd + (self.current_index as f64 * 3.0 / 24.0)
    }
}

impl<P: SpaceWeatherProvider> Iterator for SpaceWeatherApIterator<P> {
    type Item = (Epoch, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let interval_mjd = self.current_interval_mjd();

        // Check if we've reached or passed the end (end is exclusive)
        if interval_mjd >= self.end_mjd {
            return None;
        }

        // Get the Ap value for this interval
        let ap = match self.provider.get_ap(interval_mjd) {
            Ok(ap) => ap,
            Err(_) => return None,
        };

        // Create epoch at start of this interval
        let epoch = Epoch::from_mjd(interval_mjd, crate::time::TimeSystem::UTC);

        // Advance to next interval
        self.current_index += 1;
        if self.current_index > 7 {
            self.current_index = 0;
            self.current_mjd += 1.0;
        }

        Some((epoch, ap))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::space_weather::FileSpaceWeatherProvider;
    use crate::time::TimeSystem;

    #[test]
    fn test_kp_iterator_basic() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let iter = SpaceWeatherKpIterator::new(provider, start, end).unwrap();
        let values: Vec<_> = iter.collect();

        // Should have 4 intervals (00:00, 03:00, 06:00, 09:00)
        // 12:00 is the end, so the interval starting at 12:00 is not included
        assert_eq!(values.len(), 4);

        // Check each value is valid
        for (epoch, kp) in &values {
            assert!(*kp >= 0.0 && *kp <= 9.0);
            assert!(epoch.mjd() >= start.mjd());
            assert!(epoch.mjd() <= end.mjd());
        }
    }

    #[test]
    fn test_ap_iterator_basic() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let iter = SpaceWeatherApIterator::new(provider, start, end).unwrap();
        let values: Vec<_> = iter.collect();

        // Should have 4 intervals
        assert_eq!(values.len(), 4);

        // Check each value is valid
        for (epoch, ap) in &values {
            assert!(*ap >= 0.0);
            assert!(epoch.mjd() >= start.mjd());
            assert!(epoch.mjd() <= end.mjd());
        }
    }

    #[test]
    fn test_kp_iterator_full_day() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let iter = SpaceWeatherKpIterator::new(provider, start, end).unwrap();
        let values: Vec<_> = iter.collect();

        // Should have 8 intervals for a full day
        assert_eq!(values.len(), 8);
    }

    #[test]
    fn test_kp_iterator_multiple_days() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 3, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let iter = SpaceWeatherKpIterator::new(provider, start, end).unwrap();
        let values: Vec<_> = iter.collect();

        // Should have 16 intervals for 2 days
        assert_eq!(values.len(), 16);
    }

    #[test]
    fn test_iterator_invalid_range() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        let start = Epoch::from_datetime(2023, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let result = SpaceWeatherKpIterator::new(provider, start, end);
        assert!(result.is_err());
    }

    #[test]
    fn test_iterator_mid_interval_start() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        // Start at 01:30 (middle of first interval)
        let start = Epoch::from_datetime(2023, 1, 1, 1, 30, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let iter = SpaceWeatherKpIterator::new(provider, start, end).unwrap();
        let values: Vec<_> = iter.collect();

        // Should still start from the 00:00 interval since we're in that interval
        assert_eq!(values.len(), 4);
    }

    #[test]
    fn test_iterator_epoch_values() {
        let provider = FileSpaceWeatherProvider::from_default_file().unwrap();
        let start = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let end = Epoch::from_datetime(2023, 1, 2, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let iter = SpaceWeatherKpIterator::new(provider, start, end).unwrap();
        let values: Vec<_> = iter.collect();

        // Check that epochs are at the start of each 3-hour interval
        let expected_hours = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0];
        for (i, (epoch, _)) in values.iter().enumerate() {
            let (_, _, _, hour, _, _, _) = epoch.to_datetime();
            assert_eq!(hour as f64, expected_hours[i]);
        }
    }
}
