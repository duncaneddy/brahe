/*!
 * SpaceTrack API error types
 */

use std::fmt;
use std::time::Duration;

use crate::utils::BraheError;

/// SpaceTrack API error types for consistent error handling.
#[derive(Debug)]
pub enum SpaceTrackError {
    /// Authentication failed - invalid credentials or expired session.
    AuthenticationError(String),
    /// Rate limit exceeded - includes retry duration.
    RateLimitError {
        /// Duration to wait before retrying.
        retry_after: Duration,
    },
    /// Network/HTTP error during API request.
    NetworkError(String),
    /// Invalid query parameters or request configuration.
    QueryError(String),
    /// Failed to parse API response.
    ParseError(String),
    /// Session expired and needs re-authentication.
    SessionExpired,
    /// Unknown or unsupported request class.
    UnknownRequestClass(String),
    /// Unknown or unsupported predicate.
    UnknownPredicate(String),
}

impl fmt::Display for SpaceTrackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SpaceTrackError::AuthenticationError(msg) => {
                write!(f, "SpaceTrack authentication failed: {}", msg)
            }
            SpaceTrackError::RateLimitError { retry_after } => {
                write!(
                    f,
                    "SpaceTrack rate limit exceeded, retry after {:?}",
                    retry_after
                )
            }
            SpaceTrackError::NetworkError(msg) => {
                write!(f, "SpaceTrack network error: {}", msg)
            }
            SpaceTrackError::QueryError(msg) => {
                write!(f, "SpaceTrack query error: {}", msg)
            }
            SpaceTrackError::ParseError(msg) => {
                write!(f, "SpaceTrack parse error: {}", msg)
            }
            SpaceTrackError::SessionExpired => {
                write!(f, "SpaceTrack session expired")
            }
            SpaceTrackError::UnknownRequestClass(name) => {
                write!(f, "Unknown SpaceTrack request class: {}", name)
            }
            SpaceTrackError::UnknownPredicate(name) => {
                write!(f, "Unknown SpaceTrack predicate: {}", name)
            }
        }
    }
}

impl std::error::Error for SpaceTrackError {}

impl From<SpaceTrackError> for BraheError {
    fn from(e: SpaceTrackError) -> Self {
        match e {
            SpaceTrackError::AuthenticationError(msg) => BraheError::Error(msg),
            SpaceTrackError::RateLimitError { retry_after } => BraheError::Error(format!(
                "Rate limit exceeded, retry after {:?}",
                retry_after
            )),
            SpaceTrackError::NetworkError(msg) => BraheError::IoError(msg),
            SpaceTrackError::QueryError(msg) => BraheError::Error(msg),
            SpaceTrackError::ParseError(msg) => BraheError::ParseError(msg),
            SpaceTrackError::SessionExpired => {
                BraheError::Error("SpaceTrack session expired".to_string())
            }
            SpaceTrackError::UnknownRequestClass(name) => {
                BraheError::Error(format!("Unknown request class: {}", name))
            }
            SpaceTrackError::UnknownPredicate(name) => {
                BraheError::Error(format!("Unknown predicate: {}", name))
            }
        }
    }
}

impl From<reqwest::Error> for SpaceTrackError {
    fn from(e: reqwest::Error) -> Self {
        SpaceTrackError::NetworkError(e.to_string())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_authentication_error_display() {
        let e = SpaceTrackError::AuthenticationError("Invalid credentials".to_string());
        assert!(e.to_string().contains("authentication failed"));
        assert!(e.to_string().contains("Invalid credentials"));
    }

    #[test]
    fn test_rate_limit_error_display() {
        let e = SpaceTrackError::RateLimitError {
            retry_after: Duration::from_secs(60),
        };
        assert!(e.to_string().contains("rate limit exceeded"));
    }

    #[test]
    fn test_network_error_display() {
        let e = SpaceTrackError::NetworkError("Connection refused".to_string());
        assert!(e.to_string().contains("network error"));
    }

    #[test]
    fn test_query_error_display() {
        let e = SpaceTrackError::QueryError("Invalid parameter".to_string());
        assert!(e.to_string().contains("query error"));
    }

    #[test]
    fn test_parse_error_display() {
        let e = SpaceTrackError::ParseError("Invalid JSON".to_string());
        assert!(e.to_string().contains("parse error"));
    }

    #[test]
    fn test_session_expired_display() {
        let e = SpaceTrackError::SessionExpired;
        assert!(e.to_string().contains("session expired"));
    }

    #[test]
    fn test_unknown_request_class_display() {
        let e = SpaceTrackError::UnknownRequestClass("invalid_class".to_string());
        assert!(e.to_string().contains("Unknown"));
        assert!(e.to_string().contains("invalid_class"));
    }

    #[test]
    fn test_unknown_predicate_display() {
        let e = SpaceTrackError::UnknownPredicate("invalid_pred".to_string());
        assert!(e.to_string().contains("Unknown"));
        assert!(e.to_string().contains("invalid_pred"));
    }

    #[test]
    fn test_from_spacetrack_error_to_brahe_error() {
        let st_error = SpaceTrackError::AuthenticationError("test".to_string());
        let brahe_error: BraheError = st_error.into();
        assert!(matches!(brahe_error, BraheError::Error(_)));

        let st_error = SpaceTrackError::NetworkError("test".to_string());
        let brahe_error: BraheError = st_error.into();
        assert!(matches!(brahe_error, BraheError::IoError(_)));

        let st_error = SpaceTrackError::ParseError("test".to_string());
        let brahe_error: BraheError = st_error.into();
        assert!(matches!(brahe_error, BraheError::ParseError(_)));
    }
}
