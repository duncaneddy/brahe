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
 *
 * Position and state transforms add a translation: each endpoint also has an
 * [`Origin`] — a celestial center for a `Celestial` frame, and the bound
 * object for an `OrbitRelative` or `Body` frame. The two origins are
 * differenced in ICRF axes by [`origin_offset_state`], between the two
 * chains' de-rotation and re-rotation steps.
 */

use nalgebra::Vector3;

use crate::frames::object_registry::object_state;
use crate::frames::registry::{FrameKey, frame_entry};
use crate::frames::{
    BodyFrame, CelestialFrame, Frame, ObjectId, OrbitRelativeKind, OrbitRelativeVariant,
};
use crate::math::{SMatrix3, SVector6};
use crate::relative_motion::{omega_rtn, rotation_eci_to_rtn};
use crate::spice::NAIFId;
use crate::time::Epoch;
use crate::utils::BraheError;

use super::transform::{
    center_offset_state, rotation_celestial, state_celestial, state_frame_to_frame,
};

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
    pub(crate) omega: Option<Vector3<f64>>,
    /// The first chain link that supplied no angular velocity, when `omega`
    /// is `None`. Names the offending link in the state-transform error.
    pub(crate) rateless_link: Option<Frame>,
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
    if from == to {
        return Ok(SMatrix3::identity());
    }
    let from = resolve_orientation(from, epc, false)?;
    let to = resolve_orientation(to, epc, false)?;
    let roots = rotation_celestial(from.root, to.root, epc)?;
    Ok(to.dcm * roots * from.dcm.transpose())
}

/// Transforms a Cartesian position from `from` to `to` at `epc`, for
/// arbitrary frames.
///
/// Both frames are reduced to their celestial roots and rotated into ICRF
/// axes, where the two origins are differenced (see
/// [`origin_offset_state`]) before the target chain is applied in reverse.
/// No angular-velocity data is required.
///
/// # Arguments
/// - `from`: Source frame
/// - `to`: Target frame
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian position in `from` axes/origin. Units: (*m*)
///
/// # Returns
/// - `Ok(Vector3<f64>)`: Cartesian position in `to` axes/origin (*m*)
/// - `Err(BraheError)`: If either frame is unbound, is missing a registered
///   link or object, or cannot be evaluated at `epc`
pub(crate) fn resolve_position(
    from: &Frame,
    to: &Frame,
    epc: Epoch,
    x: Vector3<f64>,
) -> Result<Vector3<f64>, BraheError> {
    if from == to {
        return Ok(x);
    }
    let resolved_from = resolve_orientation(from, epc, false)?;
    let resolved_to = resolve_orientation(to, epc, false)?;
    let offset = origin_offset_state(&from.origin()?, &to.origin()?, epc)?;

    let icrf_from = icrf_aligned_inertial(resolved_from.root);
    let icrf_to = icrf_aligned_inertial(resolved_to.root);

    // `from` axes -> its root's axes -> ICRF axes, still about `from`'s
    // origin; then re-centered onto `to`'s origin and rotated back down
    // through `to`'s root into `to`'s axes.
    let p_root = resolved_from.dcm.transpose() * x;
    let p_icrf = rotation_celestial(resolved_from.root, icrf_from, epc)? * p_root
        + offset.fixed_rows::<3>(0);
    let p_root_to = rotation_celestial(icrf_to, resolved_to.root, epc)? * p_icrf;
    Ok(resolved_to.dcm * p_root_to)
}

/// Transforms a Cartesian state from `from` to `to` at `epc`, for arbitrary
/// frames.
///
/// Follows [`resolve_position`]'s pipeline with the velocity transport
/// terms restored: the state is de-rotated out of `from`'s axes
/// (`p_root = Rᵀ p`, `v_root = Rᵀ (v + ω × p)`), carried into ICRF axes
/// through the celestial state machinery — which handles the celestial
/// frames' own rotation rates exactly — re-centered by
/// [`origin_offset_state`], and rotated into `to`'s axes (`p = R p_root`,
/// `v = R v_root − ω × p`).
///
/// # Arguments
/// - `from`: Source frame
/// - `to`: Target frame
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian state (position, velocity) in `from` axes/origin. Units: (*m*; *m/s*)
///
/// # Returns
/// - `Ok(SVector6)`: Cartesian state in `to` axes/origin (*m*; *m/s*)
/// - `Err(BraheError)`: If either frame is unbound, is missing a registered
///   link or object, cannot be evaluated at `epc`, or has an orientation
///   link carrying no angular velocity
pub(crate) fn resolve_state(
    from: &Frame,
    to: &Frame,
    epc: Epoch,
    x: SVector6,
) -> Result<SVector6, BraheError> {
    if from == to {
        return Ok(x);
    }
    let resolved_from = resolve_orientation(from, epc, true)?;
    let resolved_to = resolve_orientation(to, epc, true)?;
    let omega_from = chain_rate(from, &resolved_from)?;
    let omega_to = chain_rate(to, &resolved_to)?;
    let offset = origin_offset_state(&from.origin()?, &to.origin()?, epc)?;

    let dcm_from = resolved_from.dcm.transpose();
    let p: Vector3<f64> = x.fixed_rows::<3>(0).into_owned();
    let v: Vector3<f64> = x.fixed_rows::<3>(3).into_owned();
    let p_root = dcm_from * p;
    let v_root = dcm_from * (v + omega_from.cross(&p));
    let x_root = SVector6::new(
        p_root[0], p_root[1], p_root[2], v_root[0], v_root[1], v_root[2],
    );

    // Both celestial legs share their frame's center, so they rotate only;
    // the whole translation is carried by `offset`.
    let x_icrf = state_celestial(
        resolved_from.root,
        icrf_aligned_inertial(resolved_from.root),
        epc,
        x_root,
    )? + offset;
    let x_root_to = state_celestial(
        icrf_aligned_inertial(resolved_to.root),
        resolved_to.root,
        epc,
        x_icrf,
    )?;

    let p_to: Vector3<f64> = resolved_to.dcm * x_root_to.fixed_rows::<3>(0);
    let v_to: Vector3<f64> = resolved_to.dcm * x_root_to.fixed_rows::<3>(3) - omega_to.cross(&p_to);
    Ok(SVector6::new(
        p_to[0], p_to[1], p_to[2], v_to[0], v_to[1], v_to[2],
    ))
}

/// A resolved chain's angular velocity, or the D12 error naming the link
/// that supplies no rates.
///
/// A constant orientation reports a zero rate, so this only fails for a
/// provider that genuinely carries no rate data.
///
/// # Arguments
/// - `frame`: The frame that was resolved (named in the error)
/// - `resolved`: The frame's resolution
///
/// # Returns
/// - `Ok(Vector3<f64>)`: The frame's angular velocity relative to its root,
///   expressed in the frame (*rad/s*)
/// - `Err(BraheError)`: If any link in the chain carries no angular velocity
fn chain_rate(frame: &Frame, resolved: &Resolved) -> Result<Vector3<f64>, BraheError> {
    match resolved.omega {
        Some(omega) => Ok(omega),
        None => {
            let link = resolved.rateless_link.as_ref().unwrap_or(frame);
            Err(BraheError::Error(format!(
                "cannot compute state transform through {frame}: orientation link {link} \
                 carries no angular velocity; provide rates or wrap the provider with \
                 with_numerical_rates"
            )))
        }
    }
}

/// Reduces `frame` to its celestial root at `epc`.
///
/// # Arguments
/// - `frame`: The frame to resolve
/// - `epc`: Epoch instant for evaluation of the frame's orientation
/// - `need_rate`: Whether to evaluate each chain link's angular velocity.
///   Rotation-only and position-only callers pass `false` so a provider
///   whose rotation is valid but whose rate query fails or is unsupported
///   does not spuriously break them; `resolved.omega` is `None` when
///   `false`.
///
/// # Returns
/// - `Ok(Resolved)`: The celestial root, the `root` -> `frame` rotation
///   matrix, and — when `need_rate` is `true` — the frame's angular
///   velocity relative to `root` (*rad/s*) if every link supplies one
/// - `Err(BraheError)`: If `frame` is unbound, is missing a registered
///   link, or cannot be evaluated at `epc`
pub(crate) fn resolve_orientation(
    frame: &Frame,
    epc: Epoch,
    need_rate: bool,
) -> Result<Resolved, BraheError> {
    match frame {
        Frame::Celestial(celestial) => Ok(Resolved {
            root: *celestial,
            dcm: SMatrix3::identity(),
            omega: Some(Vector3::zeros()),
            rateless_link: None,
        }),
        Frame::Body {
            object: Some(object),
            frame: body,
        } => resolve_body(frame, object, body, epc, need_rate),
        Frame::OrbitRelative {
            kind,
            variant,
            object: Some(object),
        } => resolve_orbit_relative(*kind, *variant, object, epc),
        Frame::Body { object: None, .. } | Frame::OrbitRelative { object: None, .. } => {
            Err(unbound_frame_error(frame))
        }
    }
}

/// Builds the D14-style error for a frame with no bound object: names the
/// frame and the construction that binds one.
///
/// # Arguments
/// - `frame`: The unbound frame
///
/// # Returns
/// - `BraheError`: The formatted error
fn unbound_frame_error(frame: &Frame) -> BraheError {
    let fix = match frame {
        Frame::OrbitRelative { kind, .. } => format!("construct with Frame::{kind}(object)"),
        _ => "construct with an object (e.g. Frame::SC_BODY(\"SC\")) or bind via register_for"
            .to_string(),
    };
    BraheError::Error(format!(
        "cannot evaluate {frame}: frame is not bound to an object; {fix}"
    ))
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
    need_rate: bool,
) -> Result<Resolved, BraheError> {
    let mut dcm = SMatrix3::identity();
    let mut omega = need_rate.then(Vector3::zeros);
    let mut rateless_link: Option<Frame> = None;
    let mut link = frame.clone();
    let mut key = FrameKey::Body(object.clone(), body.clone());

    loop {
        let entry = frame_entry(&key).ok_or_else(|| missing_link_error(frame, &link, object))?;
        let provider = entry.provider;
        let r_link = provider
            .rotation_matrix(epc)
            .map_err(|e| provider_error(&link, e))?
            .to_matrix();

        // Rate evaluation is skipped entirely when the caller does not need
        // it (rotation-only and position-only queries), so a provider whose
        // rotation is valid but whose rate query fails does not spuriously
        // break those queries.
        if need_rate {
            let omega_link = provider
                .angular_velocity(epc)
                .map_err(|e| provider_error(&link, e))?;

            // The contribution of this link is expressed in the frame being
            // resolved, so it is rotated by the product accumulated so far
            // (which maps this link's axes into the resolved frame's axes)
            // before the product absorbs the link itself.
            if omega_link.is_none() && rateless_link.is_none() {
                rateless_link = Some(link.clone());
            }
            omega = match (omega, omega_link) {
                (Some(total), Some(w)) => Some(total + dcm * w),
                _ => None,
            };
        }
        dcm *= r_link;

        let parent = entry
            .parent
            .expect("registered Body frame entries always carry a parent");
        match parent {
            Frame::Celestial(root) => {
                return Ok(Resolved {
                    root,
                    dcm,
                    omega,
                    rateless_link,
                });
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
/// calls that fix it. When the queried frame is itself the unregistered
/// link, it is named as the frame rather than as its own parent.
///
/// # Arguments
/// - `frame`: The frame being resolved
/// - `link`: The chain link that has no registered orientation
/// - `object`: The object `frame` is bound to
///
/// # Returns
/// - `BraheError`: The formatted error
fn missing_link_error(frame: &Frame, link: &Frame, object: &ObjectId) -> BraheError {
    if link == frame {
        return BraheError::Error(format!(
            "frame {frame} has no registered orientation; register one with \
             register_frame({frame}, <parent>, <provider>) or load an AEM and call \
             aem.register_for(\"{object}\")"
        ));
    }
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
        rateless_link: None,
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

/// The point a frame's positions are measured from.
///
/// A celestial frame's origin is its own center. Every non-celestial frame
/// is bound to an object, and its origin is that object's origin: an
/// object's body and sensor frames share its origin exactly, with no lever
/// arm between the object's center and the frames mounted on it.
#[derive(Debug, PartialEq)]
pub(crate) enum Origin {
    /// The center of a celestial frame.
    Celestial(CelestialFrame),
    /// The origin of a registered object.
    Object(ObjectId),
}

impl Frame {
    /// The point this frame's positions are measured from.
    ///
    /// # Returns
    /// - `Ok(Origin)`: The frame's origin
    /// - `Err(BraheError)`: If the frame is not bound to an object, so it
    ///   has no origin
    pub(crate) fn origin(&self) -> Result<Origin, BraheError> {
        match self {
            Frame::Celestial(celestial) => Ok(Origin::Celestial(*celestial)),
            Frame::Body {
                object: Some(object),
                ..
            }
            | Frame::OrbitRelative {
                object: Some(object),
                ..
            } => Ok(Origin::Object(object.clone())),
            Frame::Body { object: None, .. } | Frame::OrbitRelative { object: None, .. } => {
                Err(unbound_frame_error(self))
            }
        }
    }
}

/// State of `from`'s origin relative to `to`'s origin at `epc`, in ICRF
/// axes.
///
/// The translation seam of the generalized position and state transforms:
/// `x_about_to = x_about_from + origin_offset_state(from, to, epc)`. A
/// celestial origin contributes only its center; an object origin
/// contributes its registered state, rotated into ICRF axes, on top of its
/// declared frame's center. Two centers are related through the celestial
/// [`center_offset_state`] seam.
///
/// # Arguments
/// - `from`: Origin the input state is measured from
/// - `to`: Origin the output state is measured from
/// - `epc`: Epoch instant for evaluation of the origins
///
/// # Returns
/// - `Ok(SVector6)`: State `[x, y, z, vx, vy, vz]` of `from`'s origin
///   relative to `to`'s origin, in ICRF axes. Units: (*m*; *m/s*)
/// - `Err(BraheError)`: If an object is not registered or its state cannot
///   be evaluated at `epc`, or the centers cannot be related
pub(crate) fn origin_offset_state(
    from: &Origin,
    to: &Origin,
    epc: Epoch,
) -> Result<SVector6, BraheError> {
    // Identical origins (e.g. two body/sensor frames of the same object)
    // have an exactly zero offset by construction; short-circuiting avoids
    // evaluating an object's state twice and works even when that object
    // is not registered, since no state is needed to answer "zero".
    if from == to {
        return Ok(SVector6::zeros());
    }
    let (center_from, x_from) = origin_state_about_center(from, epc)?;
    let (center_to, x_to) = origin_state_about_center(to, epc)?;
    let centers = if center_from == center_to {
        SVector6::zeros()
    } else {
        center_offset_state(center_to, center_from, epc)?
    };
    Ok(x_from + centers - x_to)
}

/// An origin's celestial center and its state about that center, in ICRF
/// axes.
///
/// # Arguments
/// - `origin`: The origin to evaluate
/// - `epc`: Epoch instant for evaluation of an object origin's state
///
/// # Returns
/// - `Ok((i32, SVector6))`: NAIF ID of the origin's center, and the
///   origin's state relative to that center in ICRF axes (*m*; *m/s*)
/// - `Err(BraheError)`: If the object is not registered or its state cannot
///   be evaluated at `epc`
fn origin_state_about_center(origin: &Origin, epc: Epoch) -> Result<(i32, SVector6), BraheError> {
    match origin {
        Origin::Celestial(frame) => Ok((frame.center_naif_id(), SVector6::zeros())),
        Origin::Object(object) => {
            let (declared, x) = object_state(object, epc)?;
            let icrf = icrf_aligned_inertial(declared);
            Ok((
                declared.center_naif_id(),
                state_celestial(declared, icrf, epc, x)?,
            ))
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::Arc;

    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    use super::*;
    use crate::attitude::{EulerAxis, FromAttitude, Quaternion, ToAttitude};
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::coordinates::state_koe_to_eci;
    use crate::frames::object_registry::FnProvider;
    use crate::frames::registry::{FRAME_REGISTRY, FrameEntry};
    use crate::frames::{
        CallbackOrientation, OrientationProvider, OrientationProviderExt, clear_frame_registry,
        clear_object_registry, position_frame_to_frame, register_frame, register_object,
        rotation_frame_to_frame,
    };
    use crate::math::SVector6;
    use crate::orbit_dynamics::ephemerides::sun_position;
    use crate::relative_motion::state_eci_to_rtn;
    use crate::time::TimeSystem;
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_spice};

    #[test]
    #[parallel]
    fn test_icrf_aligned_inertial_maps_every_center() {
        // Earth-centered rotating frame.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::ITRF),
            CelestialFrame::GCRF
        );
        // Already-aligned frames pass through unchanged.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::GCRF),
            CelestialFrame::GCRF
        );
        // Moon-centered rotating frame.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::LFPA),
            CelestialFrame::LCI
        );
        // Mars-centered rotating frame.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::MCMF),
            CelestialFrame::MCI
        );
        // Earth-Moon-barycenter-centered rotating frame.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::EMR),
            CelestialFrame::EMBI
        );
        // Solar-system-barycenter-centered non-aligned frame.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::BodyFixedIAU(
                NAIFId::SolarSystemBarycenter.id()
            )),
            CelestialFrame::SSBI
        );
        // Any other center falls back to a generic ICRF-aligned frame at
        // that center.
        assert_eq!(
            icrf_aligned_inertial(CelestialFrame::BodyFixedIAU(-20001)),
            CelestialFrame::BodyCenteredICRF(-20001)
        );
    }

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
        // The queried frame is itself the unregistered link, so it is named
        // as the frame rather than as its own parent.
        assert!(err.contains("frame CSS_1@SC has no registered orientation"));
        assert!(!err.contains("parent CSS_1@SC"));
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
        let resolved = resolve_orientation(&Frame::CSS("SC", "1"), epc, true).unwrap();
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
            resolve_orientation(&Frame::CSS("SC", "1"), epc, true)
                .unwrap()
                .omega
                .is_none()
        );
        clear_frame_registry();
    }

    /// Rotation-only provider whose rate query always fails, to verify
    /// rotation-only and position-only queries never evaluate it.
    struct RotationOnlyErroringRates;

    impl OrientationProvider for RotationOnlyErroringRates {
        fn quaternion(&self, _epoch: Epoch) -> Result<Quaternion, BraheError> {
            Ok(Quaternion::new(1.0, 0.0, 0.0, 0.0))
        }

        fn angular_velocity(&self, _epoch: Epoch) -> Result<Option<Vector3<f64>>, BraheError> {
            Err(BraheError::Error("rate query unsupported".to_string()))
        }
    }

    #[test]
    #[serial]
    fn test_rotation_and_position_do_not_evaluate_angular_velocity() {
        // resolve_body must not call angular_velocity for rotation-only or
        // position-only queries: a provider whose rotation is valid but
        // whose rate query errors must not break those two.
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let oe = SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let x = state_koe_to_eci(oe, AngleFormat::Degrees);
        register_object("SC", FnProvider(move |_| Ok(x)), CelestialFrame::GCRF).unwrap();
        register_frame(
            Frame::SC_BODY("SC"),
            CelestialFrame::GCRF.into(),
            RotationOnlyErroringRates,
        )
        .unwrap();

        assert!(rotation_frame_to_frame(CelestialFrame::GCRF, Frame::SC_BODY("SC"), epc).is_ok());
        assert!(
            position_frame_to_frame(
                CelestialFrame::GCRF,
                Frame::SC_BODY("SC"),
                epc,
                Vector3::new(R_EARTH + 500e3, 0.0, 0.0),
            )
            .is_ok()
        );
        // The state router does need rates, so it surfaces the provider's error.
        let err = state_frame_to_frame(
            CelestialFrame::GCRF,
            Frame::SC_BODY("SC"),
            epc,
            SVector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("rate query unsupported"));

        clear_frame_registry();
        clear_object_registry();
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

        let rotating = resolve_orientation(&Frame::RTN("A"), epc, true).unwrap();
        assert_abs_diff_eq!(rotating.omega.unwrap(), omega_rtn(x), epsilon = 1e-15);

        let inertial = Frame::orbit_relative(
            OrbitRelativeKind::RTN,
            OrbitRelativeVariant::Inertial,
            Some("A".into()),
        )
        .unwrap();
        let snapshot = resolve_orientation(&inertial, epc, true).unwrap();
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
        let resolved = resolve_orientation(&Frame::RTN("A"), epc, true).unwrap();
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

    #[test]
    #[serial]
    fn test_sun_vector_in_sensor_frame() {
        setup_global_test_eop();
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let x_sc = state_koe_to_eci(
            SVector6::new(R_EARTH + 500e3, 0.0, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        );
        register_object("SC", FnProvider(move |_| Ok(x_sc)), CelestialFrame::GCRF).unwrap();
        let q_body = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let q_css = Quaternion::from_euler_axis(EulerAxis::new(
            Vector3::y_axis().into_inner(),
            0.7,
            AngleFormat::Radians,
        ));
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), q_body).unwrap();
        register_frame(Frame::CSS("SC", "1"), Frame::SC_BODY("SC"), q_css).unwrap();

        // The GCRF sun direction is re-centered on the object and rotated
        // through SC_BODY -> CSS_1; equal to the manual recipe.
        let sun_gcrf = sun_position(epc);
        let got =
            position_frame_to_frame(CelestialFrame::GCRF, Frame::CSS("SC", "1"), epc, sun_gcrf)
                .unwrap();
        let manual = q_css.to_rotation_matrix().to_matrix()
            * q_body.to_rotation_matrix().to_matrix()
            * (sun_gcrf - x_sc.fixed_rows::<3>(0));
        // Tolerance is ~1e-9 relative at the Sun's distance; the residual is
        // exactly zero, since both paths apply the same rotations.
        assert_abs_diff_eq!(got, manual, epsilon = 150.0);
        clear_frame_registry();
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_same_object_body_frames_share_origin_without_object_registered() {
        // Two body/sensor frames of the same object share their origin
        // exactly (no lever arm), so position/state transforms between them
        // never need the object's own state -- and so succeed even when the
        // object itself was never registered.
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let q_body = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let q_css = Quaternion::from_euler_axis(EulerAxis::new(
            Vector3::y_axis().into_inner(),
            0.7,
            AngleFormat::Radians,
        ));
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), q_body).unwrap();
        register_frame(Frame::CSS("SC", "1"), Frame::SC_BODY("SC"), q_css).unwrap();

        let x = Vector3::new(1.0, 2.0, 3.0);
        let got =
            position_frame_to_frame(Frame::SC_BODY("SC"), Frame::CSS("SC", "1"), epc, x).unwrap();
        let expected = q_css.to_rotation_matrix().to_matrix() * x;
        assert_abs_diff_eq!(got, expected, epsilon = 1e-15);

        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_two_object_rtn_matches_state_eci_to_rtn() {
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let x_a = state_koe_to_eci(
            SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        );
        let x_b = state_koe_to_eci(
            SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.2),
            AngleFormat::Degrees,
        );
        register_object("A", FnProvider(move |_| Ok(x_a)), CelestialFrame::GCRF).unwrap();

        // Routing B's GCRF state into A's RTN frame is the same arithmetic
        // as the dedicated relative-motion transform, so the two agree bit
        // for bit.
        let got = state_frame_to_frame(CelestialFrame::GCRF, Frame::RTN("A"), epc, x_b).unwrap();
        assert_eq!(got, state_eci_to_rtn(x_a, x_b));
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_state_transform_missing_rates_errors() {
        setup_global_test_eop();
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let t0 = epc - 10.0;
        let spinning = CallbackOrientation::new(
            move |e: Epoch| {
                let dt: f64 = e - t0;
                let (s, c) = (0.001 * dt).sin_cos();
                Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
            },
            None,
        );
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), spinning).unwrap();
        register_object(
            "SC",
            FnProvider(move |_| Ok(SVector6::zeros())),
            CelestialFrame::GCRF,
        )
        .unwrap();

        // The rotation is time-varying but the provider supplies no rates, so
        // the state transform cannot form the velocity transport term.
        let err = state_frame_to_frame(
            CelestialFrame::GCRF,
            Frame::SC_BODY("SC"),
            epc,
            SVector6::zeros(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no angular velocity"));
        assert!(err.contains("with_numerical_rates"));

        // The position transform needs no rates and still succeeds.
        assert!(
            position_frame_to_frame(
                CelestialFrame::GCRF,
                Frame::SC_BODY("SC"),
                epc,
                Vector3::new(1.0, 2.0, 3.0)
            )
            .is_ok()
        );
        clear_frame_registry();
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_body_frame_velocity_matches_finite_difference() {
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let t0 = epc - 10.0;
        let rate = 1.0e-3;
        let spinning = CallbackOrientation::new(
            move |e: Epoch| {
                let dt: f64 = e - t0;
                let (s, c) = (rate * dt).sin_cos();
                Ok(SMatrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
            },
            None,
        )
        .with_numerical_rates(0.1);
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF.into(), spinning).unwrap();
        register_object(
            "SC",
            FnProvider(move |_| Ok(SVector6::zeros())),
            CelestialFrame::GCRF,
        )
        .unwrap();

        // A target held fixed in GCRF sweeps through the spinning body frame
        // purely by the transport term, so the body-frame velocity must match
        // a central difference of the body-frame position.
        let p_gcrf = Vector3::new(9.0e3, -4.0e3, 2.0e3);
        let x_gcrf = SVector6::new(p_gcrf[0], p_gcrf[1], p_gcrf[2], 0.0, 0.0, 0.0);
        let body = Frame::SC_BODY("SC");
        let x_body = state_frame_to_frame(CelestialFrame::GCRF, body.clone(), epc, x_gcrf).unwrap();

        let delta = 0.5;
        let p_plus =
            position_frame_to_frame(CelestialFrame::GCRF, body.clone(), epc + delta, p_gcrf)
                .unwrap();
        let p_minus =
            position_frame_to_frame(CelestialFrame::GCRF, body, epc - delta, p_gcrf).unwrap();
        let v_numerical = (p_plus - p_minus) / (2.0 * delta);

        // The tolerance is set by the central difference's own truncation
        // error, ~(delta^2/6) * rate^3 * |p|, against a 9.85 m/s transport
        // velocity.
        assert_abs_diff_eq!(
            x_body.fixed_rows::<3>(3).into_owned(),
            v_numerical,
            epsilon = 1e-6
        );
        clear_frame_registry();
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_object_origin_offset_crosses_centers() {
        setup_global_test_spice();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let x_a = state_koe_to_eci(
            SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        );
        let x_b = state_koe_to_eci(
            SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.2),
            AngleFormat::Degrees,
        );
        register_object("A", FnProvider(move |_| Ok(x_a)), CelestialFrame::GCRF).unwrap();

        // A's RTN frame is Earth-rooted, so routing out of it into the
        // Moon-centered inertial frame combines the object-origin offset
        // with the Earth -> Moon center offset.
        let x_rtn = state_frame_to_frame(CelestialFrame::GCRF, Frame::RTN("A"), epc, x_b).unwrap();
        let via_rtn =
            state_frame_to_frame(Frame::RTN("A"), CelestialFrame::LCI, epc, x_rtn).unwrap();
        let direct =
            state_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::LCI, epc, x_b).unwrap();
        assert_abs_diff_eq!(via_rtn, direct, epsilon = 1e-6);

        let p_b: Vector3<f64> = x_b.fixed_rows::<3>(0).into_owned();
        let p_rtn =
            position_frame_to_frame(CelestialFrame::GCRF, Frame::RTN("A"), epc, p_b).unwrap();
        let p_via =
            position_frame_to_frame(Frame::RTN("A"), CelestialFrame::LCI, epc, p_rtn).unwrap();
        let p_direct =
            position_frame_to_frame(CelestialFrame::GCRF, CelestialFrame::LCI, epc, p_b).unwrap();
        assert_abs_diff_eq!(p_via, p_direct, epsilon = 1e-6);
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_same_frame_short_circuits_without_evaluation() {
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let frame = Frame::CSS("SC", "1");

        // Nothing is registered, so resolving either endpoint would fail;
        // the identity short-circuit returns the input untouched.
        assert_eq!(
            rotation_frame_to_frame(frame.clone(), frame.clone(), epc).unwrap(),
            SMatrix3::identity()
        );
        let p = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(
            position_frame_to_frame(frame.clone(), frame.clone(), epc, p).unwrap(),
            p
        );
        let x = SVector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(
            state_frame_to_frame(frame.clone(), frame, epc, x).unwrap(),
            x
        );
    }

    #[test]
    #[parallel]
    fn test_unbound_frame_has_no_origin() {
        let unbound: Frame = BodyFrame::SCBody(None).into();
        let err = unbound.origin().unwrap_err().to_string();
        assert!(err.contains("not bound to an object"));

        let unbound =
            Frame::orbit_relative(OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating, None)
                .unwrap();
        let err = unbound.origin().unwrap_err().to_string();
        assert!(err.contains("Frame::RTN(object)"));
    }

    #[test]
    #[serial]
    fn test_cross_root_state_transport_round_trips() {
        setup_global_test_eop();
        clear_frame_registry();
        clear_object_registry();
        let epc = Epoch::from_date(2024, 3, 1, TimeSystem::UTC);
        let x_sc = state_koe_to_eci(
            SVector6::new(R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        );
        register_object("SC", FnProvider(move |_| Ok(x_sc)), CelestialFrame::GCRF).unwrap();
        let q = Quaternion::from_euler_axis(EulerAxis::new(
            Vector3::z_axis().into_inner(),
            0.3,
            AngleFormat::Radians,
        ));
        register_frame(Frame::SC_BODY("SC"), CelestialFrame::ITRF.into(), q).unwrap();

        // An ITRF-rooted body frame carries the Earth-rotation transport term,
        // so the round trip exercises both the forward and inverse forms.
        let x_gcrf = state_koe_to_eci(
            SVector6::new(R_EARTH + 501e3, 0.001, 97.8, 15.0, 30.0, 45.1),
            AngleFormat::Degrees,
        );
        let x_body =
            state_frame_to_frame(CelestialFrame::GCRF, Frame::SC_BODY("SC"), epc, x_gcrf).unwrap();
        let back =
            state_frame_to_frame(Frame::SC_BODY("SC"), CelestialFrame::GCRF, epc, x_body).unwrap();
        assert_abs_diff_eq!(back, x_gcrf, epsilon = 1e-6);
        clear_frame_registry();
        clear_object_registry();
    }
}
