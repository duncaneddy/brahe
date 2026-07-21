//! Horizons SPK generation request model.

use crate::time::Epoch;
use crate::utils::cache::short_hash;
use crate::utils::download::urlencode;

/// A request to generate an SPK kernel for a small body over a time span.
///
/// `center` defaults to `"500@0"` (Solar System Barycenter), which produces an
/// SSB-relative segment that chains cleanly with the SSB-relative `de440s`
/// ephemeris for third-body and SRP force resolution.
#[derive(Debug, Clone)]
pub struct HorizonsSPKRequest {
    /// Horizons `COMMAND` target, e.g. `"DES=2000001;"`.
    pub command: String,
    /// SPK span start.
    pub start: Epoch,
    /// SPK span stop.
    pub stop: Epoch,
    /// Horizons `CENTER`, e.g. `"500@0"` (SSB).
    pub center: String,
}

impl HorizonsSPKRequest {
    /// Create a request from a raw `COMMAND` string and time span.
    ///
    /// # Arguments
    ///
    /// * `command` - Horizons target command (e.g. `"DES=2000001;"`).
    /// * `start` - SPK span start epoch.
    /// * `stop` - SPK span stop epoch.
    pub fn new(command: &str, start: Epoch, stop: Epoch) -> Self {
        HorizonsSPKRequest {
            command: command.to_string(),
            start,
            stop,
            center: "500@0".to_string(),
        }
    }

    /// Create a request targeting a small body by SPK-ID (`"DES=<spkid>;"`).
    ///
    /// # Arguments
    ///
    /// * `spkid` - Small-body SPK-ID (NAIF ID), e.g. `2000001` for Ceres.
    /// * `start` - SPK span start epoch.
    /// * `stop` - SPK span stop epoch.
    pub fn for_spkid(spkid: i32, start: Epoch, stop: Epoch) -> Self {
        Self::new(&format!("DES={};", spkid), start, stop)
    }

    /// Override the `CENTER` body (default `"500@0"`, the SSB).
    ///
    /// # Arguments
    ///
    /// * `center` - Horizons center specification.
    pub fn with_center(mut self, center: &str) -> Self {
        self.center = center.to_string();
        self
    }

    /// Format an epoch as a Horizons calendar string (`YYYY-MM-DD HH:MM:SS`).
    fn horizons_time(epoch: Epoch) -> String {
        let (y, mo, d, h, mi, s, _ns) = epoch.to_datetime();
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            y, mo, d, h, mi, s as u32
        )
    }

    /// Build the URL query string (leading `?`) for the SPK request.
    pub(crate) fn query(&self) -> String {
        let start = Self::horizons_time(self.start);
        let stop = Self::horizons_time(self.stop);
        format!(
            "?format=json&EPHEM_TYPE=SPK&MAKE_EPHEM=YES&OBJ_DATA=NO\
             &COMMAND={}&CENTER={}&START_TIME={}&STOP_TIME={}",
            urlencode(&format!("'{}'", self.command)),
            urlencode(&format!("'{}'", self.center)),
            urlencode(&format!("'{}'", start)),
            urlencode(&format!("'{}'", stop)),
        )
    }

    /// Build the deterministic `.bsp` cache filename for this request.
    pub(crate) fn cache_key(&self) -> String {
        let sanitized = self
            .command
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>();
        let digest = short_hash(&format!(
            "{}|{}|{}|{}",
            self.command,
            Self::horizons_time(self.start),
            Self::horizons_time(self.stop),
            self.center
        ));
        format!("{}_{}.bsp", sanitized.trim_matches('_'), digest)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use serial_test::parallel;

    fn span() -> (Epoch, Epoch) {
        let t0 = Epoch::from_datetime(2015, 12, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
        let t1 = Epoch::from_datetime(2016, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
        (t0, t1)
    }

    #[test]
    #[parallel]
    fn test_for_spkid_command() {
        let (t0, t1) = span();
        let req = HorizonsSPKRequest::for_spkid(2000001, t0, t1);
        assert_eq!(req.command, "DES=2000001;");
        assert_eq!(req.center, "500@0");
    }

    #[test]
    #[parallel]
    fn test_with_center_overrides() {
        let (t0, t1) = span();
        let req = HorizonsSPKRequest::for_spkid(2000001, t0, t1).with_center("10");
        assert_eq!(req.center, "10");
    }

    #[test]
    #[parallel]
    fn test_query_contains_encoded_params() {
        let (t0, t1) = span();
        let q = HorizonsSPKRequest::for_spkid(2000001, t0, t1).query();
        assert!(q.starts_with('?'));
        assert!(q.contains("format=json"));
        assert!(q.contains("EPHEM_TYPE=SPK"));
        assert!(q.contains("MAKE_EPHEM=YES"));
        assert!(q.contains("OBJ_DATA=NO"));
        // COMMAND='DES=2000001;' url-encoded (quotes %27, = %3D, ; %3B).
        assert!(q.contains("COMMAND=%27DES%3D2000001%3B%27"));
        assert!(q.contains("CENTER=%27500%400%27"));
        assert!(q.contains("START_TIME=%272015-12-01"));
    }

    #[test]
    #[parallel]
    fn test_cache_key_is_bsp_and_stable() {
        let (t0, t1) = span();
        let a = HorizonsSPKRequest::for_spkid(2000001, t0, t1).cache_key();
        let b = HorizonsSPKRequest::for_spkid(2000001, t0, t1).cache_key();
        assert_eq!(a, b);
        assert!(a.ends_with(".bsp"));
        assert!(a.contains("DES_2000001"));
        let c = HorizonsSPKRequest::for_spkid(2000001, t0, t1)
            .with_center("10")
            .cache_key();
        assert_ne!(a, c);
    }
}
