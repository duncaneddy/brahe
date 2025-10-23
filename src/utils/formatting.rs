/*!
 * Formatting utilities for human-readable output.
 */

/// Format a time duration in seconds to a human-readable string.
///
/// Converts a duration in seconds to either a long format (e.g., "6 minutes and 2.00 seconds")
/// or a short format (e.g., "6m 2s").
///
/// # Arguments
///
/// * `seconds` - Time duration in seconds
/// * `short` - If true, use short format; otherwise use long format
///
/// # Returns
///
/// A human-readable string representation of the time duration
///
/// # Examples
///
/// ```
/// use brahe::utils::format_time_string;
///
/// // Long format
/// assert_eq!(format_time_string(45.5, false), "45.50 seconds");
/// assert_eq!(format_time_string(90.0, false), "1 minutes and 30.00 seconds");
/// assert_eq!(format_time_string(3665.0, false), "1 hours, 1 minutes, and 5.00 seconds");
/// assert_eq!(format_time_string(86400.0, false), "1 days, 0 hours, 0 minutes, and 0.00 seconds");
///
/// // Short format
/// assert_eq!(format_time_string(45.5, true), "46s");
/// assert_eq!(format_time_string(90.0, true), "1m 30s");
/// assert_eq!(format_time_string(3665.0, true), "1h 1m 5s");
/// assert_eq!(format_time_string(86400.0, true), "1d");
/// ```
pub fn format_time_string(seconds: f64, short: bool) -> String {
    if short {
        format_time_string_short(seconds)
    } else {
        format_time_string_long(seconds)
    }
}

/// Format time in short format (e.g., "6m 2s", "1h 30m 15s", "2d 3h 45m")
fn format_time_string_short(seconds: f64) -> String {
    let days = (seconds / 86400.0).floor() as i64;
    let hours = ((seconds / 3600.0).floor() as i64) % 24;
    let minutes = ((seconds / 60.0).floor() as i64) % 60;
    let secs = (seconds % 60.0).floor() as i64;

    let mut parts = Vec::new();

    if days > 0 {
        parts.push(format!("{}d", days));
    }
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{}m", minutes));
    }
    if secs > 0 || parts.is_empty() {
        parts.push(format!("{}s", secs));
    }

    parts.join(" ")
}

/// Format time in long format (e.g., "6 minutes and 2.00 seconds")
fn format_time_string_long(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.2} seconds", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let remaining_seconds = seconds % 60.0;
        format!(
            "{} minutes and {:.2} seconds",
            minutes as i64, remaining_seconds
        )
    } else if seconds < 86400.0 {
        let hours = (seconds / 3600.0).floor();
        let minutes = ((seconds / 60.0).floor() as i64) % 60;
        let remaining_seconds = seconds % 60.0;
        format!(
            "{} hours, {} minutes, and {:.2} seconds",
            hours as i64, minutes, remaining_seconds
        )
    } else {
        let days = (seconds / 86400.0).floor();
        let hours = ((seconds / 3600.0).floor() as i64) % 24;
        let minutes = ((seconds / 60.0).floor() as i64) % 60;
        let remaining_seconds = seconds % 60.0;
        format!(
            "{} days, {} hours, {} minutes, and {:.2} seconds",
            days as i64, hours, minutes, remaining_seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_time_string_short_seconds() {
        assert_eq!(format_time_string(45.5, true), "45s");
        assert_eq!(format_time_string(0.5, true), "0s");
        assert_eq!(format_time_string(30.0, true), "30s");
    }

    #[test]
    fn test_format_time_string_short_minutes() {
        assert_eq!(format_time_string(90.0, true), "1m 30s");
        assert_eq!(format_time_string(125.75, true), "2m 5s");
    }

    #[test]
    fn test_format_time_string_short_hours() {
        assert_eq!(format_time_string(3665.0, true), "1h 1m 5s");
        assert_eq!(format_time_string(7200.0, true), "2h");
    }

    #[test]
    fn test_format_time_string_short_days() {
        assert_eq!(format_time_string(86400.0, true), "1d");
        assert_eq!(format_time_string(90061.5, true), "1d 1h 1m 1s");
    }

    #[test]
    fn test_format_time_string_long_seconds() {
        assert_eq!(format_time_string(45.5, false), "45.50 seconds");
        assert_eq!(format_time_string(0.5, false), "0.50 seconds");
    }

    #[test]
    fn test_format_time_string_long_minutes() {
        assert_eq!(
            format_time_string(90.0, false),
            "1 minutes and 30.00 seconds"
        );
        assert_eq!(
            format_time_string(125.75, false),
            "2 minutes and 5.75 seconds"
        );
    }

    #[test]
    fn test_format_time_string_long_hours() {
        assert_eq!(
            format_time_string(3665.0, false),
            "1 hours, 1 minutes, and 5.00 seconds"
        );
        assert_eq!(
            format_time_string(7200.0, false),
            "2 hours, 0 minutes, and 0.00 seconds"
        );
    }

    #[test]
    fn test_format_time_string_long_days() {
        assert_eq!(
            format_time_string(86400.0, false),
            "1 days, 0 hours, 0 minutes, and 0.00 seconds"
        );
        assert_eq!(
            format_time_string(90061.5, false),
            "1 days, 1 hours, 1 minutes, and 1.50 seconds"
        );
    }
}
