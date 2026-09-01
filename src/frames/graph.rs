/*!
 * Frame-graph resolution: reducing any [`Frame`] to a celestial root plus
 * the rotation (and, where available, the angular velocity) relating the
 * two.
 *
 * Every frame in the graph ultimately hangs off a [`CelestialFrame`]:
 *
 * - A `Celestial` frame is its own root, with an identity rotation.
 * - A `Body` frame is resolved by walking the frame registry's parent links
 *   from the frame up to the celestial frame that terminates its chain,
 *   composing each link's orientation provider along the way.
 * - An `OrbitRelative` frame is built from its bound object's registered
 *   state, expressed in the ICRF-aligned inertial frame sharing the center
 *   of the frame the object's states are declared in.
 *
 * Reducing both endpoints of a query to `(root, rotation)` pairs turns an
 * arbitrary frame-to-frame rotation into the celestial rotation between the
 * two roots, pre- and post-multiplied by the two chains.
 */

use nalgebra::Vector3;

use crate::frames::object_registry::object_state;
use crate::frames::registry::{FrameKey, frame_entry};
use crate::frames::{
    BodyFrame, CelestialFrame, Frame, ObjectId, OrbitRelativeKind, OrbitRelativeVariant,
};
use crate::math::SMatrix3;
use crate::relative_motion::{omega_rtn, rotation_eci_to_rtn};
use crate::spice::NAIFId;
use crate::time::Epoch;
use crate::utils::BraheError;

use super::transform::{rotation_celestial, state_frame_to_frame};

/// A frame reduced to its celestial root.
///
/// `dcm` rotates vectors expressed in `root` into the resolved frame's own
/// axes. `omega` is the resolved frame's angular velocity relative to
/// `root`, expressed in the resolved frame; it is `None` when any link in
/// the chain carries no rate data, since a partially known chain rate is
/// not a usable rate.
pub(crate) struct Resolved {
    /// Celestial frame terminating the resolved frame's chain.
    pub(crate) root: CelestialFrame,
    /// Rotation matrix from `root` axes into the resolved frame's axes.
    pub(crate) dcm: SMatrix3,
    /// Angular velocity of the resolved frame relative to `root`, expressed
    /// in the resolved frame. Units: (*rad/s*)
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) omega: Option<Vector3<f64>>,
}

/// Computes the rotation matrix transforming `from` axes into `to` axes at
/// `epc`, for arbitrary frames.
///
/// Both frames are reduced to their celestial roots, and the two chains are
/// joined through the celestial rotation between those roots:
/// `R = dcm_to * R(root_from -> root_to) * dcm_fromᵀ`.
///
/// # Arguments
/// - `from`: Source frame
/// - `to`: Target frame
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `Ok(SMatrix3)`: 3x3 rotation matrix transforming `from` -> `to`
/// - `Err(BraheError)`: If either frame is unbound, is missing a registered
///   link, or cannot be evaluated at `epc`
pub(crate) fn resolve_rotation(
    from: &Frame,
    to: &Frame,
    epc: Epoch,
) -> Result<SMatrix3, BraheError> {
    let from = resolve_orientation(from, epc)?;
    let to = resolve_orientation(to, epc)?;
    let roots = rotation_celestial(from.root, to.root, epc)?;
    Ok(to.dcm * roots * from.dcm.transpose())
}

/// Reduces `frame` to its celestial root at `epc`.
///
/// # Arguments
/// - `frame`: The frame to resolve
/// - `epc`: Epoch instant for evaluation of the frame's orientation
///
/// # Returns
/// - `Ok(Resolved)`: The celestial root, the `root` -> `frame` rotation
///   matrix, and the frame's angular velocity relative to `root` (*rad/s*)
///   when every link supplies one
/// - `Err(BraheError)`: If `frame` is unbound, is missing a registered
///   link, or cannot be evaluated at `epc`
pub(crate) fn resolve_orientation(frame: &Frame, epc: Epoch) -> Result<Resolved, BraheError> {
    match frame {
        Frame::Celestial(celestial) => Ok(Resolved {
            root: *celestial,
            dcm: SMatrix3::identity(),
            omega: Some(Vector3::zeros()),
        }),
        Frame::Body {
            object: Some(object),
            frame: body,
        } => resolve_body(frame, object, body, epc),
        Frame::Body { object: None, .. } => Err(BraheError::Error(format!(
            "cannot evaluate {frame}: frame is not bound to an object; construct with an \
             object (e.g. Frame::SC_BODY(\"SC\")) or bind via register_for"
        ))),
        Frame::OrbitRelative {
            kind,
            variant,
            object: Some(object),
        } => resolve_orbit_relative(*kind, *variant, object, epc),
        Frame::OrbitRelative {
            kind, object: None, ..
        } => Err(BraheError::Error(format!(
            "cannot evaluate {frame}: frame is not bound to an object; construct with \
             Frame::{kind}(object)"
        ))),
    }
}

/// Walks a bound `Body` frame's registered parent links up to a celestial
/// root, composing each link's rotation and angular velocity.
///
/// # Arguments
/// - `frame`: The bound `Body` frame being resolved (named in errors)
/// - `object`: The object `frame` is bound to
/// - `body`: `frame`'s body-frame kind and designator
/// - `epc`: Epoch instant for evaluation of each link's orientation
///
/// # Returns
/// - `Ok(Resolved)`: The chain's celestial root, composed rotation, and
///   composed angular velocity (*rad/s*)
/// - `Err(BraheError)`: If a link is unregistered or fails to evaluate
fn resolve_body(
    frame: &Frame,
    object: &ObjectId,
    body: &BodyFrame,
    epc: Epoch,
) -> Result<Resolved, BraheError> {
    let mut dcm = SMatrix3::identity();
    let mut omega = Some(Vector3::zeros());
    let mut link = frame.clone();
    let mut key = FrameKey::Body(object.clone(), body.clone());

    loop {
        let entry = frame_entry(&key).ok_or_else(|| missing_link_error(frame, &link, object))?;
        let provider = entry.provider;
        let r_link = provider
            .rotation_matrix(epc)
            .map_err(|e| provider_error(&link, e))?
            .to_matrix();
        let omega_link = provider
            .angular_velocity(epc)
            .map_err(|e| provider_error(&link, e))?;

        // The contribution of this link is expressed in the frame being
        // resolved, so it is rotated by the product accumulated so far
        // (which maps this link's axes into the resolved frame's axes)
        // before the product absorbs the link itself.
        omega = match (omega, omega_link) {
            (Some(total), Some(w)) => Some(total + dcm * w),
            _ => None,
        };
        dcm *= r_link;

        let parent = entry
            .parent
            .expect("registered Body frame entries always carry a parent");
        match parent {
            Frame::Celestial(root) => {
                return Ok(Resolved { root, dcm, omega });
            }
            Frame::Body {
                object: Some(ref parent_object),
                frame: ref parent_body,
            } => {
                key = FrameKey::Body(parent_object.clone(), parent_body.clone());
                link = parent;
            }
            other => return Err(missing_link_error(frame, &other, object)),
        }
    }
}

/// Builds the D14-style error for a chain link with no registered
/// orientation: names the frame being resolved, the missing link, and the
/// calls that fix it.
///
/// # Arguments
/// - `frame`: The frame being resolved
/// - `link`: The chain link that has no registered orientation
/// - `object`: The object `frame` is bound to
///
/// # Returns
/// - `BraheError`: The formatted error
fn missing_link_error(frame: &Frame, link: &Frame, object: &ObjectId) -> BraheError {
    BraheError::Error(format!(
        "cannot resolve {frame}: parent {link} has no registered orientation; register one \
         with register_frame({link}, <parent's parent>, <provider>) or load an AEM and call \
         aem.register_for(\"{object}\")"
    ))
}

/// Wraps an orientation provider's evaluation failure (e.g. an epoch
/// outside its coverage) with the frame it was evaluated for.
///
/// # Arguments
/// - `frame`: The frame whose provider failed
/// - `err`: The provider's error
///
/// # Returns
/// - `BraheError`: The wrapped error
fn provider_error(frame: &Frame, err: BraheError) -> BraheError {
    BraheError::Error(format!("orientation for {frame}: {err}"))
}

/// Builds a bound orbit-relative frame's axes from its object's registered
/// state.
///
/// The object's state is converted into the ICRF-aligned inertial frame
/// sharing the center of the frame its states are declared in, and that
/// inertial frame becomes the resolved root.
///
/// # Arguments
/// - `kind`: The orbit-relative axes definition
/// - `variant`: Rotating (true local orbital frame) or inertial snapshot
/// - `object`: The object the frame is bound to
/// - `epc`: Epoch instant for evaluation of the object's state
///
/// # Returns
/// - `Ok(Resolved)`: The inertial root, the `root` -> frame rotation, and
///   the frame's angular velocity relative to `root` (*rad/s*) — the
///   orbital rate for the rotating variant, zero for the inertial snapshot
/// - `Err(BraheError)`: If `kind` has no axes derivation, `object` is not
///   registered, or its state cannot be evaluated at `epc`
fn resolve_orbit_relative(
    kind: OrbitRelativeKind,
    variant: OrbitRelativeVariant,
    object: &ObjectId,
    epc: Epoch,
) -> Result<Resolved, BraheError> {
    if kind != OrbitRelativeKind::RTN {
        return Err(BraheError::Error(format!(
            "orbit-relative kind {kind} does not yet have an axes derivation (tracked in \
             issue #452); RTN is supported"
        )));
    }

    let (declared, x) = object_state(object, epc)?;
    let root = icrf_aligned_inertial(declared);
    let x_root = state_frame_to_frame(declared, root, epc, x)?;

    Ok(Resolved {
        root,
        dcm: rotation_eci_to_rtn(x_root),
        omega: Some(match variant {
            OrbitRelativeVariant::Rotating => omega_rtn(x_root),
            OrbitRelativeVariant::Inertial => Vector3::zeros(),
        }),
    })
}

/// The ICRF-aligned inertial frame sharing `frame`'s center.
///
/// Returns `frame` itself when it is already ICRF-aligned, so a state
/// declared in such a frame is used without conversion.
///
/// # Arguments
/// - `frame`: The celestial frame whose center to match
///
/// # Returns
/// - `CelestialFrame`: The ICRF-aligned frame centered on `frame`'s center
fn icrf_aligned_inertial(frame: CelestialFrame) -> CelestialFrame {
    match frame {
        CelestialFrame::GCRF
        | CelestialFrame::LCI
        | CelestialFrame::MCI
        | CelestialFrame::EMBI
        | CelestialFrame::SSBI
        | CelestialFrame::BodyCenteredICRF(_) => frame,
        other => {
            let center = other.center_naif_id();
            if center == NAIFId::Earth.id() {
                CelestialFrame::GCRF
            } else if center == NAIFId::Moon.id() {
                CelestialFrame::LCI
            } else if center == NAIFId::Mars.id() {
                CelestialFrame::MCI
            } else if center == NAIFId::EarthMoonBarycenter.id() {
                CelestialFrame::EMBI
            } else if center == NAIFId::SolarSystemBarycenter.id() {
                CelestialFrame::SSBI
            } else {
                CelestialFrame::BodyCenteredICRF(center)
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::Arc;

    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::attitude::{EulerAxis, FromAttitude, Quaternion, ToAttitude};
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::coordinates::state_koe_to_eci;
    use crate::frames::object_registry::FnProvider;
    use crate::frames::registry::{FRAME_REGISTRY, FrameEntry};
    use crate::frames::{
        CallbackOrientation, clear_frame_registry, clear_object_registry, register_frame,
        register_object, rotation_frame_to_frame,
    };
    use crate::math::SVector6;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;

    #[test]
    #[serial]
    fn test_rotation_celestial_bit_identity() {
        setup_global_test_eop();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let direct =
            rotation_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();
        let via_frame = rotation_frame_to_frame(
            Frame::from(CelestialFrame::GCRF),
            Frame::from(CelestialFrame::ITRF),
            epc,
        )
        .unwrap();
        assert_eq!(direct, via_frame);
    }

    #[test]
    #[serial]
    fn test_body_chain_matches_manual_composition() {
        setup_global_test_eop();
        clear_frame_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let q_body = Quaternion::from_euler_axis(EulerAxis::new(
            Vector3::z_axis().into_inner(),
            0.3,
            AngleFormat::Radians,
        ));
        let q_css = Quaternion::from_euler_axis(EulerAxis::new(
            Vector3::x_axis().into_inner(),
            1.1,
            AngleFormat::Radians,
        ));
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), q_body).unwrap();
        register_frame(Frame::CSS("SC", "1"), Frame::SC_BODY("SC"), q_css).unwrap();
        let r = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::CSS("SC", "1"), epc).unwrap();
        let expected =
            q_css.to_rotation_matrix().to_matrix() * q_body.to_rotation_matrix().to_matrix();
        assert_abs_diff_eq!(r, expected, epsilon = 1e-14);
        // Inverse direction is the transpose
        let r_inv =
            rotation_frame_to_frame(Frame::CSS("SC", "1"), CelestialFrame::GCRF, epc).unwrap();
        assert_abs_diff_eq!(r_inv, expected.transpose(), epsilon = 1e-14);
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_missing_link_error_names_fix() {
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::CSS("SC", "1"), epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("CSS_1@SC"));
        assert!(err.contains("register_frame"));
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::RTN("A"), epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("object 'A' is not registered"));
        let unbound: Frame = BodyFrame::SCBody(None).into();
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, unbound, epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not bound to an object"));
    }

    #[test]
    #[serial]
    fn test_rtn_rotation_matches_relative_motion() {
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let x = state_koe_to_eci(oe, AngleFormat::Degrees);
        register_object("A", FnProvider(move |_| Ok(x)), CelestialFrame::GCRF).unwrap();
        let r = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::RTN("A"), epc).unwrap();
        assert_abs_diff_eq!(r, rotation_eci_to_rtn(x), epsilon = 1e-14);
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_body_chain_composes_angular_velocity() {
        clear_frame_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let w_body = Vector3::new(0.0, 0.0, 1.0e-3);
        let q_css = Quaternion::from_euler_axis(EulerAxis::new(
            Vector3::x_axis().into_inner(),
            1.1,
            AngleFormat::Radians,
        ));
        let spin = CallbackOrientation::new(
            |_epc: Epoch| Ok(SMatrix3::identity()),
            Some(Box::new(move |_epc: Epoch| Ok(w_body))),
        );
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), spin).unwrap();
        register_frame(Frame::CSS("SC", "1"), Frame::SC_BODY("SC"), q_css).unwrap();

        // omega_css = omega_link(css) + R_css * omega_body, and the constant
        // q_css link contributes no rate of its own.
        let resolved = resolve_orientation(&Frame::CSS("SC", "1"), epc).unwrap();
        let expected = q_css.to_rotation_matrix().to_matrix() * w_body;
        assert_abs_diff_eq!(resolved.omega.unwrap(), expected, epsilon = 1e-15);
        assert_eq!(resolved.root, CelestialFrame::GCRF);
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_body_chain_angular_velocity_none_when_link_lacks_rates() {
        clear_frame_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let rotation_only = CallbackOrientation::new(|_epc: Epoch| Ok(SMatrix3::identity()), None);
        register_frame(
            Frame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            rotation_only,
        )
        .unwrap();
        register_frame(Frame::CSS("SC", "1"), Frame::SC_BODY("SC"), q).unwrap();

        // A single rate-less link poisons the whole chain's angular velocity
        assert!(
            resolve_orientation(&Frame::CSS("SC", "1"), epc)
                .unwrap()
                .omega
                .is_none()
        );
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_cross_root_rotation_joins_chains_through_celestial() {
        setup_global_test_eop();
        clear_frame_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::ITRF.into(), q).unwrap();

        // Identity link off an ITRF root: the chain reduces to GCRF -> ITRF
        let r = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::SC_BODY("SC"), epc).unwrap();
        let expected =
            rotation_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::ITRF, epc).unwrap();
        assert_abs_diff_eq!(r, expected, epsilon = 1e-15);
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_rtn_variants_angular_velocity() {
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let x = state_koe_to_eci(oe, AngleFormat::Degrees);
        register_object("A", FnProvider(move |_| Ok(x)), CelestialFrame::GCRF).unwrap();

        let rotating = resolve_orientation(&Frame::RTN("A"), epc).unwrap();
        assert_abs_diff_eq!(rotating.omega.unwrap(), omega_rtn(x), epsilon = 1e-15);

        let inertial = Frame::orbit_relative(
            OrbitRelativeKind::RTN,
            OrbitRelativeVariant::Inertial,
            Some("A".into()),
        )
        .unwrap();
        let snapshot = resolve_orientation(&inertial, epc).unwrap();
        assert_eq!(snapshot.omega.unwrap(), Vector3::zeros());
        // Both variants share the same axes at any single epoch
        assert_eq!(snapshot.dcm, rotating.dcm);
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_rtn_non_icrf_declared_frame_converts_to_inertial_root() {
        setup_global_test_eop();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let x_gcrf = state_koe_to_eci(oe, AngleFormat::Degrees);
        let x_itrf =
            state_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::ITRF, epc, x_gcrf).unwrap();
        register_object("A", FnProvider(move |_| Ok(x_itrf)), CelestialFrame::ITRF).unwrap();

        // Declared in ITRF, so the state is converted to the ICRF-aligned
        // frame sharing ITRF's center before the RTN axes are built.
        let resolved = resolve_orientation(&Frame::RTN("A"), epc).unwrap();
        assert_eq!(resolved.root, CelestialFrame::GCRF);
        assert_abs_diff_eq!(resolved.dcm, rotation_eci_to_rtn(x_gcrf), epsilon = 1e-9);
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_unsupported_orbit_relative_kind_names_issue() {
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::LVLH("A"), epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("LVLH"));
        assert!(err.contains("issue #452"));
        assert!(err.contains("RTN is supported"));

        let unbound =
            Frame::orbit_relative(OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating, None)
                .unwrap();
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, unbound, epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not bound to an object"));
        assert!(err.contains("Frame::RTN(object)"));
    }

    #[test]
    #[serial]
    fn test_provider_failure_is_wrapped_with_frame_context() {
        clear_frame_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let failing = CallbackOrientation::new(
            |_epc: Epoch| Err(BraheError::Error("epoch outside coverage".to_string())),
            None,
        );
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), failing).unwrap();
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::SC_BODY("SC"), epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("orientation for SC_BODY@SC"));
        assert!(err.contains("epoch outside coverage"));
        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_malformed_parent_link_reports_missing_orientation() {
        clear_frame_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        // `register_frame` rejects a non-celestial, non-registered parent, so
        // the entry is inserted directly to exercise the defensive branch.
        FRAME_REGISTRY.write().unwrap().insert(
            FrameKey::Body("SC".into(), BodyFrame::SCBody(None)),
            FrameEntry {
                parent: Some(BodyFrame::CSS(Some("1".to_string())).into()),
                provider: Arc::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            },
        );
        let err = rotation_frame_to_frame(CelestialFrame::GCRF, Frame::SC_BODY("SC"), epc)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot resolve SC_BODY@SC"));
        assert!(err.contains("register_frame"));
        clear_frame_registry();
    }
}
