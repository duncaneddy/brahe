/*!
 * SpaceTrack rate limiter
 *
 * Implements rate limiting for SpaceTrack API requests.
 * SpaceTrack enforces 30 requests per minute and 300 requests per hour.
 */

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Rate limiter for SpaceTrack API requests.
///
/// SpaceTrack enforces two rate limits:
/// - 30 requests per minute
/// - 300 requests per hour
///
/// This implementation uses a sliding window approach with in-memory
/// timestamp tracking.
#[derive(Clone)]
pub struct RateLimiter {
    /// Timestamps of requests within the last minute
    minute_requests: Arc<Mutex<VecDeque<Instant>>>,
    /// Timestamps of requests within the last hour
    hour_requests: Arc<Mutex<VecDeque<Instant>>>,
    /// Maximum requests per minute (default: 30)
    per_minute_limit: u32,
    /// Maximum requests per hour (default: 300)
    per_hour_limit: u32,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter {
    /// Create a new rate limiter with default SpaceTrack limits.
    ///
    /// Default limits:
    /// - 30 requests per minute
    /// - 300 requests per hour
    pub fn new() -> Self {
        Self {
            minute_requests: Arc::new(Mutex::new(VecDeque::new())),
            hour_requests: Arc::new(Mutex::new(VecDeque::new())),
            per_minute_limit: 30,
            per_hour_limit: 300,
        }
    }

    /// Create a rate limiter with custom limits.
    ///
    /// # Arguments
    ///
    /// * `per_minute` - Maximum requests per minute
    /// * `per_hour` - Maximum requests per hour
    pub fn with_limits(per_minute: u32, per_hour: u32) -> Self {
        Self {
            minute_requests: Arc::new(Mutex::new(VecDeque::new())),
            hour_requests: Arc::new(Mutex::new(VecDeque::new())),
            per_minute_limit: per_minute,
            per_hour_limit: per_hour,
        }
    }

    /// Check if a request is allowed, returning wait duration if rate limited.
    ///
    /// This method cleans up old entries and checks both minute and hour limits.
    ///
    /// # Returns
    ///
    /// * `None` if the request is allowed
    /// * `Some(Duration)` if rate limited, indicating how long to wait
    pub fn check(&self) -> Option<Duration> {
        let now = Instant::now();
        self.clean_old_entries(now);

        let minute_count = self.minute_requests.lock().unwrap().len();
        let hour_count = self.hour_requests.lock().unwrap().len();

        // Check minute limit
        if minute_count >= self.per_minute_limit as usize {
            let oldest = self.minute_requests.lock().unwrap().front().copied();
            if let Some(oldest) = oldest {
                let elapsed = now.duration_since(oldest);
                if elapsed < Duration::from_secs(60) {
                    return Some(Duration::from_secs(60) - elapsed);
                }
            }
        }

        // Check hour limit
        if hour_count >= self.per_hour_limit as usize {
            let oldest = self.hour_requests.lock().unwrap().front().copied();
            if let Some(oldest) = oldest {
                let elapsed = now.duration_since(oldest);
                if elapsed < Duration::from_secs(3600) {
                    return Some(Duration::from_secs(3600) - elapsed);
                }
            }
        }

        None
    }

    /// Record a request timestamp.
    ///
    /// Call this after successfully making a request to track it for rate limiting.
    pub fn record_request(&self) {
        let now = Instant::now();
        self.minute_requests.lock().unwrap().push_back(now);
        self.hour_requests.lock().unwrap().push_back(now);
    }

    /// Clean up entries older than their respective windows.
    fn clean_old_entries(&self, now: Instant) {
        let minute_cutoff = now - Duration::from_secs(60);
        let hour_cutoff = now - Duration::from_secs(3600);

        // Clean minute entries
        {
            let mut minute = self.minute_requests.lock().unwrap();
            while let Some(&front) = minute.front() {
                if front < minute_cutoff {
                    minute.pop_front();
                } else {
                    break;
                }
            }
        }

        // Clean hour entries
        {
            let mut hour = self.hour_requests.lock().unwrap();
            while let Some(&front) = hour.front() {
                if front < hour_cutoff {
                    hour.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Get the current number of requests in the minute window.
    pub fn minute_count(&self) -> usize {
        self.clean_old_entries(Instant::now());
        self.minute_requests.lock().unwrap().len()
    }

    /// Get the current number of requests in the hour window.
    pub fn hour_count(&self) -> usize {
        self.clean_old_entries(Instant::now());
        self.hour_requests.lock().unwrap().len()
    }

    /// Reset all rate limit counters.
    pub fn reset(&self) {
        self.minute_requests.lock().unwrap().clear();
        self.hour_requests.lock().unwrap().clear();
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_new_rate_limiter() {
        let limiter = RateLimiter::new();
        assert_eq!(limiter.per_minute_limit, 30);
        assert_eq!(limiter.per_hour_limit, 300);
        assert_eq!(limiter.minute_count(), 0);
        assert_eq!(limiter.hour_count(), 0);
    }

    #[test]
    fn test_with_limits() {
        let limiter = RateLimiter::with_limits(10, 100);
        assert_eq!(limiter.per_minute_limit, 10);
        assert_eq!(limiter.per_hour_limit, 100);
    }

    #[test]
    fn test_record_request() {
        let limiter = RateLimiter::new();
        assert_eq!(limiter.minute_count(), 0);

        limiter.record_request();
        assert_eq!(limiter.minute_count(), 1);
        assert_eq!(limiter.hour_count(), 1);

        limiter.record_request();
        assert_eq!(limiter.minute_count(), 2);
        assert_eq!(limiter.hour_count(), 2);
    }

    #[test]
    fn test_check_allows_under_limit() {
        let limiter = RateLimiter::with_limits(5, 50);

        // Should allow requests under limit
        for _ in 0..4 {
            assert!(limiter.check().is_none());
            limiter.record_request();
        }

        assert_eq!(limiter.minute_count(), 4);
    }

    #[test]
    fn test_check_blocks_at_limit() {
        let limiter = RateLimiter::with_limits(3, 50);

        // Fill up to limit
        for _ in 0..3 {
            assert!(limiter.check().is_none());
            limiter.record_request();
        }

        // Should be rate limited now
        let wait = limiter.check();
        assert!(wait.is_some());
        assert!(wait.unwrap() <= Duration::from_secs(60));
    }

    #[test]
    fn test_reset() {
        let limiter = RateLimiter::new();

        limiter.record_request();
        limiter.record_request();
        assert_eq!(limiter.minute_count(), 2);

        limiter.reset();
        assert_eq!(limiter.minute_count(), 0);
        assert_eq!(limiter.hour_count(), 0);
    }

    #[test]
    fn test_clone() {
        let limiter = RateLimiter::new();
        limiter.record_request();

        let cloned = limiter.clone();
        // Cloned limiter shares the same Arc, so counts should match
        assert_eq!(cloned.minute_count(), 1);

        limiter.record_request();
        assert_eq!(cloned.minute_count(), 2);
    }

    #[test]
    fn test_default() {
        let limiter = RateLimiter::default();
        assert_eq!(limiter.per_minute_limit, 30);
        assert_eq!(limiter.per_hour_limit, 300);
    }
}
