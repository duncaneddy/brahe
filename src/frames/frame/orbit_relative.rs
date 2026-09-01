/*!
 * Local orbital frame axes definitions and the rotating/inertial variant
 * they pair with.
 */

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::utils::errors::BraheError;

/// Local orbital frame axes definitions.
///
/// `RTN` is the frame the SANA registries call `RSW`; brahe uses its
/// existing RTN vocabulary (`state_eci_to_rtn`, `covariance_rtn`).
///
/// Every kind is a valid frame identity, which is what parsing a data file
/// needs, but only `RTN` has an axes derivation today. A transform through
/// any other kind errors until issue #452 adds the remaining derivations.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::OrbitRelativeFrameKind;
///
/// assert_eq!(OrbitRelativeFrameKind::RTN.to_string(), "RTN");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRelativeFrameKind {
    /// Local-Vertical Local-Horizontal.
    LVLH,
    /// Radial / transverse (along-track) / normal (cross-track). SANA: RSW.
    RTN,
    /// Normal / tangential / cross-track.
    NTW,
    /// Tangential / normal / cross-track.
    TNW,
    /// Perifocal. SANA-registered only as an inertial-snapshot frame.
    PQW,
    /// Equinoctial. SANA-registered only as an inertial-snapshot frame.
    EQW,
    /// Topocentric south / east / zenith.
    SEZ,
    /// Velocity / normal / co-normal.
    VNC,
    /// Nadir / Sun / normal.
    NSW,
}

impl fmt::Display for OrbitRelativeFrameKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let token = match self {
            Self::LVLH => "LVLH",
            Self::RTN => "RTN",
            Self::NTW => "NTW",
            Self::TNW => "TNW",
            Self::PQW => "PQW",
            Self::EQW => "EQW",
            Self::SEZ => "SEZ",
            Self::VNC => "VNC",
            Self::NSW => "NSW",
        };
        write!(f, "{}", token)
    }
}

/// Rotating vs. quasi-inertial snapshot variant of a local orbital frame.
///
/// - **Rotating**: True local orbital frame, rotating with the orbit. It
///   carries the orbital angular velocity, so a state transform through it
///   picks up the corresponding velocity transport term.
/// - **Inertial**: The same axes, taken as an instantaneous snapshot of the
///   orbit state at the evaluation epoch and then treated as non-rotating.
///   Its rate is zero, so a state transform through it applies no transport
///   term. The axes still differ from epoch to epoch, since each evaluation
///   takes a fresh snapshot.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::OrbitRelativeFrameVariant;
///
/// assert_eq!(OrbitRelativeFrameVariant::Rotating.to_string(), "rotating");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRelativeFrameVariant {
    /// True local orbital frame, rotating with the orbit.
    Rotating,
    /// Quasi-inertial snapshot: the orbit-relative axes at the evaluation
    /// epoch, with the frame's rate taken as zero.
    Inertial,
}

impl fmt::Display for OrbitRelativeFrameVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rotating => write!(f, "rotating"),
            Self::Inertial => write!(f, "inertial"),
        }
    }
}

/// A local orbital frame: a kind plus rotating/inertial-snapshot variant.
///
/// Represents a frame that rotates with or tracks the orbit, composed of:
/// - A frame construction type (e.g., RTN, LVLH) defining the axes
/// - A variant indicating whether the frame rotates with the orbit or is
///   frozen
///
/// Fields are private: per the SANA registry, `EQW` and `PQW` exist only as
/// inertial-snapshot frames, so construction goes through
/// [`OrbitRelativeFrame::new`] to reject that combination rather than
/// allowing it and erroring later.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::{OrbitRelativeFrame, OrbitRelativeFrameKind, OrbitRelativeFrameVariant};
///
/// let rtn = OrbitRelativeFrame::new(OrbitRelativeFrameKind::RTN, OrbitRelativeFrameVariant::Rotating);
/// assert!(rtn.is_ok());
/// assert_eq!(rtn.unwrap().to_string(), "RTN (rotating)");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrbitRelativeFrame {
    kind: OrbitRelativeFrameKind,
    variant: OrbitRelativeFrameVariant,
}

impl OrbitRelativeFrame {
    /// Constructs an orbit-relative frame, validating the kind/variant
    /// combination.
    ///
    /// Per the SANA orbit-relative frame registry, `EQW` and `PQW` are
    /// defined only as inertial-snapshot frames; they have no rotating
    /// variant.
    ///
    /// # Arguments
    /// * `kind` - The frame construction (axes definition)
    /// * `variant` - Rotating (true local orbital frame) or inertial
    ///   snapshot
    ///
    /// # Returns
    /// * `Ok(OrbitRelativeFrame)`: If the combination is valid
    /// * `Err(BraheError)`: If `kind` is `EQW` or `PQW` and `variant` is
    ///   `Rotating`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::frames::{OrbitRelativeFrame, OrbitRelativeFrameKind, OrbitRelativeFrameVariant};
    ///
    /// let rtn = OrbitRelativeFrame::new(OrbitRelativeFrameKind::RTN, OrbitRelativeFrameVariant::Rotating);
    /// assert!(rtn.is_ok());
    ///
    /// let eqw_rotating = OrbitRelativeFrame::new(OrbitRelativeFrameKind::EQW, OrbitRelativeFrameVariant::Rotating);
    /// assert!(eqw_rotating.is_err());
    /// ```
    pub fn new(
        kind: OrbitRelativeFrameKind,
        variant: OrbitRelativeFrameVariant,
    ) -> Result<Self, BraheError> {
        if matches!(
            kind,
            OrbitRelativeFrameKind::EQW | OrbitRelativeFrameKind::PQW
        ) && variant == OrbitRelativeFrameVariant::Rotating
        {
            return Err(BraheError::Error(format!(
                "orbit-relative frame {} exists only as an inertial SANA frame and cannot be \
                 constructed with the rotating variant",
                kind
            )));
        }
        Ok(Self { kind, variant })
    }

    /// Returns the frame construction (axes definition).
    ///
    /// # Returns
    /// `OrbitRelativeFrameKind`: The frame construction
    pub fn kind(&self) -> OrbitRelativeFrameKind {
        self.kind
    }

    /// Returns the rotating/inertial-snapshot variant.
    ///
    /// # Returns
    /// `OrbitRelativeFrameVariant`: Rotating or inertial
    pub fn variant(&self) -> OrbitRelativeFrameVariant {
        self.variant
    }
}

impl fmt::Display for OrbitRelativeFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.kind, self.variant)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn test_orbit_relative_kind_display_all_variants() {
        let cases = [
            (OrbitRelativeFrameKind::LVLH, "LVLH"),
            (OrbitRelativeFrameKind::RTN, "RTN"),
            (OrbitRelativeFrameKind::NTW, "NTW"),
            (OrbitRelativeFrameKind::TNW, "TNW"),
            (OrbitRelativeFrameKind::PQW, "PQW"),
            (OrbitRelativeFrameKind::EQW, "EQW"),
            (OrbitRelativeFrameKind::SEZ, "SEZ"),
            (OrbitRelativeFrameKind::VNC, "VNC"),
            (OrbitRelativeFrameKind::NSW, "NSW"),
        ];
        for (kind, expected) in cases {
            assert_eq!(kind.to_string(), expected);
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_variant_display() {
        assert_eq!(OrbitRelativeFrameVariant::Rotating.to_string(), "rotating");
        assert_eq!(OrbitRelativeFrameVariant::Inertial.to_string(), "inertial");
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_new_rejects_eqw_pqw_rotating() {
        assert!(
            OrbitRelativeFrame::new(
                OrbitRelativeFrameKind::EQW,
                OrbitRelativeFrameVariant::Rotating
            )
            .is_err()
        );
        assert!(
            OrbitRelativeFrame::new(
                OrbitRelativeFrameKind::PQW,
                OrbitRelativeFrameVariant::Rotating
            )
            .is_err()
        );
        assert!(
            OrbitRelativeFrame::new(
                OrbitRelativeFrameKind::EQW,
                OrbitRelativeFrameVariant::Inertial
            )
            .is_ok()
        );
        assert!(
            OrbitRelativeFrame::new(
                OrbitRelativeFrameKind::PQW,
                OrbitRelativeFrameVariant::Inertial
            )
            .is_ok()
        );
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_new_accepts_valid_combos() {
        for kind in [
            OrbitRelativeFrameKind::LVLH,
            OrbitRelativeFrameKind::RTN,
            OrbitRelativeFrameKind::NTW,
            OrbitRelativeFrameKind::TNW,
            OrbitRelativeFrameKind::SEZ,
            OrbitRelativeFrameKind::VNC,
            OrbitRelativeFrameKind::NSW,
        ] {
            assert!(OrbitRelativeFrame::new(kind, OrbitRelativeFrameVariant::Rotating).is_ok());
            assert!(OrbitRelativeFrame::new(kind, OrbitRelativeFrameVariant::Inertial).is_ok());
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_kind_variant_accessors() {
        let frame = OrbitRelativeFrame::new(
            OrbitRelativeFrameKind::RTN,
            OrbitRelativeFrameVariant::Rotating,
        )
        .unwrap();
        assert_eq!(frame.kind(), OrbitRelativeFrameKind::RTN);
        assert_eq!(frame.variant(), OrbitRelativeFrameVariant::Rotating);
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_display() {
        let frame = OrbitRelativeFrame::new(
            OrbitRelativeFrameKind::VNC,
            OrbitRelativeFrameVariant::Rotating,
        )
        .unwrap();
        assert_eq!(frame.to_string(), "VNC (rotating)");
    }
}
