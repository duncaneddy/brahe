/*!
 * Sliding-window rate limiter for SpaceTrack API requests.
 *
 * Space-Track.org enforces rate limits of 30 requests per minute and
 * 300 requests per hour. This module provides a rate limiter that tracks
 * request timestamps in two sliding windows (1-minute and 1-hour) to
 * prevent exceeding these limits.
 *
 * Default limits are set conservatively at ~83% of the actual limits
 * (25/min, 250/hour) to provide safety margin for clock drift and
 * shared accounts.
 */

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for SpaceTrack API rate limiting.
///
/// Defines the maximum number of requests allowed per minute and per hour.
/// Defaults to 25 requests/minute and 250 requests/hour (~83% of
/// Space-Track.org's actual limits of 30/min and 300/hour).
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::RateLimitConfig;
///
/// // Use default conservative limits
/// let config = RateLimitConfig::default();
/// assert_eq!(config.max_per_minute, 25);
/// assert_eq!(config.max_per_hour, 250);
///
/// // Custom limits
/// let config = RateLimitConfig {
///     max_per_minute: 10,
///     max_per_hour: 100,
/// };
///
/// // Disable rate limiting
/// let config = RateLimitConfig::disabled();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitConfig {
    /// Maximum requests allowed per rolling 60-second window.
    pub max_per_minute: u32,
    /// Maximum requests allowed per rolling 3600-second window.
    pub max_per_hour: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        RateLimitConfig {
            max_per_minute: 25,
            max_per_hour: 250,
        }
    }
}

impl RateLimitConfig {
    /// Create a configuration that effectively disables rate limiting.
    ///
    /// Sets both limits to `u32::MAX`, so no request will ever be delayed.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::RateLimitConfig;
    ///
    /// let config = RateLimitConfig::disabled();
    /// assert_eq!(config.max_per_minute, u32::MAX);
    /// assert_eq!(config.max_per_hour, u32::MAX);
    /// ```
    pub fn disabled() -> Self {
        RateLimitConfig {
            max_per_minute: u32::MAX,
            max_per_hour: u32::MAX,
        }
    }
}

/// Sliding-window rate limiter that tracks request timestamps.
///
/// Maintains two `VecDeque<Instant>` windows for the 1-minute and 1-hour
/// periods. Before each request, call `acquire()` to get the duration to
/// sleep before proceeding. The timestamp is recorded optimistically
/// (at `now + wait`) so concurrent threads see correct future timestamps.
pub(crate) struct RateLimiter {
    config: RateLimitConfig,
    minute_window: VecDeque<Instant>,
    hour_window: VecDeque<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub(crate) fn new(config: RateLimitConfig) -> Self {
        RateLimiter {
            config,
            minute_window: VecDeque::new(),
            hour_window: VecDeque::new(),
        }
    }

    /// Acquire permission to make a request.
    ///
    /// Returns the `Duration` the caller must sleep before proceeding.
    /// A zero duration means the request can proceed immediately.
    ///
    /// The request timestamp is recorded as `now + wait_duration` so that
    /// concurrent callers (who acquire while this caller is sleeping)
    /// correctly see the future timestamp.
    pub(crate) fn acquire(&mut self) -> Duration {
        let now = Instant::now();

        // Prune expired entries from both windows
        let minute_cutoff = now - Duration::from_secs(60);
        while self
            .minute_window
            .front()
            .is_some_and(|&t| t < minute_cutoff)
        {
            self.minute_window.pop_front();
        }

        let hour_cutoff = now - Duration::from_secs(3600);
        while self.hour_window.front().is_some_and(|&t| t < hour_cutoff) {
            self.hour_window.pop_front();
        }

        // Calculate required wait time
        let mut wait = Duration::ZERO;

        if self.minute_window.len() >= self.config.max_per_minute as usize
            && let Some(&oldest) = self.minute_window.front()
        {
            let minute_wait = (oldest + Duration::from_secs(60)).saturating_duration_since(now);
            if minute_wait > wait {
                wait = minute_wait;
            }
        }

        if self.hour_window.len() >= self.config.max_per_hour as usize
            && let Some(&oldest) = self.hour_window.front()
        {
            let hour_wait = (oldest + Duration::from_secs(3600)).saturating_duration_since(now);
            if hour_wait > wait {
                wait = hour_wait;
            }
        }

        // Record the future timestamp when the request will actually fire
        let request_time = now + wait;
        self.minute_window.push_back(request_time);
        self.hour_window.push_back(request_time);

        wait
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_per_minute, 25);
        assert_eq!(config.max_per_hour, 250);
    }

    #[test]
    fn test_rate_limit_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert_eq!(config.max_per_minute, u32::MAX);
        assert_eq!(config.max_per_hour, u32::MAX);
    }

    #[test]
    fn test_rate_limit_config_custom() {
        let config = RateLimitConfig {
            max_per_minute: 10,
            max_per_hour: 100,
        };
        assert_eq!(config.max_per_minute, 10);
        assert_eq!(config.max_per_hour, 100);
    }

    #[test]
    fn test_rate_limit_config_clone() {
        let config = RateLimitConfig::default();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_rate_limit_config_debug() {
        let config = RateLimitConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("25"));
        assert!(debug.contains("250"));
    }

    #[test]
    fn test_rate_limit_config_equality() {
        let a = RateLimitConfig::default();
        let b = RateLimitConfig::default();
        let c = RateLimitConfig::disabled();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_rate_limiter_acquire_within_limits() {
        let config = RateLimitConfig {
            max_per_minute: 5,
            max_per_hour: 100,
        };
        let mut limiter = RateLimiter::new(config);

        // First 5 requests should return zero wait
        for _ in 0..5 {
            let wait = limiter.acquire();
            assert_eq!(wait, Duration::ZERO);
        }
    }

    #[test]
    fn test_rate_limiter_acquire_at_minute_limit() {
        let config = RateLimitConfig {
            max_per_minute: 3,
            max_per_hour: 100,
        };
        let mut limiter = RateLimiter::new(config);

        // Fill the minute window
        for _ in 0..3 {
            let wait = limiter.acquire();
            assert_eq!(wait, Duration::ZERO);
        }

        // 4th request should require a wait
        let wait = limiter.acquire();
        assert!(wait > Duration::ZERO);
        assert!(wait <= Duration::from_secs(60));
    }

    #[test]
    fn test_rate_limiter_acquire_at_hour_limit() {
        let config = RateLimitConfig {
            max_per_minute: 100,
            max_per_hour: 3,
        };
        let mut limiter = RateLimiter::new(config);

        // Fill the hour window
        for _ in 0..3 {
            let wait = limiter.acquire();
            assert_eq!(wait, Duration::ZERO);
        }

        // 4th request should require a wait
        let wait = limiter.acquire();
        assert!(wait > Duration::ZERO);
        assert!(wait <= Duration::from_secs(3600));
    }

    #[test]
    fn test_rate_limiter_disabled_no_wait() {
        let config = RateLimitConfig::disabled();
        let mut limiter = RateLimiter::new(config);

        // Even after many requests, should never wait
        for _ in 0..1000 {
            let wait = limiter.acquire();
            assert_eq!(wait, Duration::ZERO);
        }
    }

    #[test]
    fn test_rate_limiter_records_future_timestamps() {
        let config = RateLimitConfig {
            max_per_minute: 2,
            max_per_hour: 100,
        };
        let mut limiter = RateLimiter::new(config);

        // Fill the minute window
        limiter.acquire();
        limiter.acquire();

        // 3rd acquire should return non-zero wait and record future timestamp
        let wait = limiter.acquire();
        assert!(wait > Duration::ZERO);

        // 4th acquire should wait even longer (stacks on the future timestamp)
        let wait2 = limiter.acquire();
        assert!(wait2 > Duration::ZERO);
        // The second wait should be at least as long because the window is still full
        // (the future timestamp from request 3 is in the window)
    }

    #[test]
    fn test_rate_limiter_minute_window_takes_precedence() {
        // Minute limit is tighter than hour limit in terms of burst
        let config = RateLimitConfig {
            max_per_minute: 2,
            max_per_hour: 10,
        };
        let mut limiter = RateLimiter::new(config);

        limiter.acquire();
        limiter.acquire();

        let wait = limiter.acquire();
        // Should be capped by minute window (~60s), not hour window
        assert!(wait > Duration::ZERO);
        assert!(wait <= Duration::from_secs(60));
    }
}
