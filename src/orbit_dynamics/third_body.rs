/*!
Module for the third body perturbations. Also provides low-precession models for the Sun and Moon
ephemerides.
 */

use nalgebra::Vector3;

use crate::math::traits::IntoPosition;
use crate::orbit_dynamics::ephemerides::{moon_position, sun_position};
use crate::orbit_dynamics::gravity::accel_point_mass_gravity;
use crate::propagators::CentralBody;
use crate::propagators::force_model_config::{EphemerisSource, ThirdBody};
use crate::spice::{
    SPKKernel, jupiter_position_de, load_kernel, mars_position_de, mercury_position_de,
    moon_position_de, neptune_position_de, saturn_position_de, spk_position, sun_position_de,
    uranus_position_de, venus_position_de,
};
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::{
    GM_JUPITER, GM_MARS, GM_MERCURY, GM_MOON, GM_NEPTUNE, GM_SATURN, GM_SUN, GM_URANUS, GM_VENUS,
};

fn de_kernel_from_source(source: EphemerisSource) -> SPKKernel {
    SPKKernel::try_from(source).expect("DE ephemeris source should map to a DE kernel")
}

/// Unified third-body acceleration with source enumeration.
///
/// Calculate gravitational acceleration due to a celestial body using
/// the specified ephemeris source. This function consolidates all
/// body-specific and source-specific acceleration functions.
///
/// # Arguments
///
/// * `body` - Celestial body acting as perturber
/// * `source` - Ephemeris source for body position
/// * `epc` - Epoch for ephemeris lookup
/// * `r_object` - Position of object in GCRF frame (or 6D state, position only used). Units: [m]
///
/// # Returns
///
/// * Acceleration vector in GCRF frame. Units: [m/s²]
///
/// # Panics
///
/// Panics if requesting a planet body with `EphemerisSource::LowPrecision`
/// (only Sun and Moon supported for low-precision).
///
/// # Example
///
/// ```
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::accel_third_body;
/// use brahe::propagators::force_model_config::{ThirdBody, EphemerisSource};
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// // Low-precision Sun
/// let a_sun = accel_third_body(ThirdBody::Sun, EphemerisSource::LowPrecision, epc, r_object);
///
/// // High-precision Mars (requires DE440s/DE440)
/// let a_mars = accel_third_body(ThirdBody::Mars, EphemerisSource::DE440s, epc, r_object);
/// ```
pub fn accel_third_body<P: IntoPosition>(
    body: ThirdBody,
    source: EphemerisSource,
    epc: Epoch,
    r_object: P,
) -> Vector3<f64> {
    let (r_body, gm) = match (body, source) {
        // Low-precision - Sun/Moon only
        (ThirdBody::Sun, EphemerisSource::LowPrecision) => (sun_position(epc), GM_SUN),
        (ThirdBody::Moon, EphemerisSource::LowPrecision) => (moon_position(epc), GM_MOON),

        // SPK-backed ephemerides - all bodies (shared code, differ only in kernel loaded)
        (
            ThirdBody::Sun,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            sun_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Sun position"),
            GM_SUN,
        ),
        (
            ThirdBody::Moon,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            moon_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Moon position"),
            GM_MOON,
        ),
        (
            ThirdBody::Mercury,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            mercury_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Mercury position"),
            GM_MERCURY,
        ),
        (
            ThirdBody::Venus,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            venus_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Venus position"),
            GM_VENUS,
        ),
        (
            ThirdBody::Mars,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            mars_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Mars position"),
            GM_MARS,
        ),
        (
            ThirdBody::Jupiter,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            jupiter_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Jupiter position"),
            GM_JUPITER,
        ),
        (
            ThirdBody::Saturn,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            saturn_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Saturn position"),
            GM_SATURN,
        ),
        (
            ThirdBody::Uranus,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            uranus_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Uranus position"),
            GM_URANUS,
        ),
        (
            ThirdBody::Neptune,
            source @ (EphemerisSource::DE440s | EphemerisSource::DE440 | EphemerisSource::SPK(_)),
        ) => (
            neptune_position_de(epc, de_kernel_from_source(source))
                .expect("Failed to get Neptune position"),
            GM_NEPTUNE,
        ),

        // Invalid: bodies only supported through `accel_third_body_for_body`
        // (Earth, Phobos, Deimos, and Custom only make sense relative to a
        // non-Earth central body, which this Earth-centered function does
        // not model).
        (
            body @ (ThirdBody::Earth
            | ThirdBody::Phobos
            | ThirdBody::Deimos
            | ThirdBody::Custom { .. }),
            _,
        ) => {
            panic!(
                "accel_third_body only supports Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, \
                Uranus, and Neptune. Requested {:?}. Use accel_third_body_for_body for other bodies.",
                body
            )
        }

        // Invalid: planets with low-precision
        (body, EphemerisSource::LowPrecision) => {
            panic!(
                "Low-precision ephemerides only support Sun and Moon. \
                Requested {:?}. Use EphemerisSource::DE440s, DE440, or SPK(...) for planets.",
                body
            )
        }
    };

    accel_point_mass_gravity(r_object, r_body, gm)
}

/// Central-body-aware third-body acceleration, including barycenter handling.
///
/// `accel_third_body` implicitly assumes the frame is centered on Earth,
/// which is not accelerated by the bodies it perturbs (Earth's own gravity
/// on itself is not part of the third-body sum). When the frame is instead
/// centered on some other body, that assumption breaks: the perturbing
/// body's gravity also acts on the central body itself, which moves the
/// origin of the frame the object's position is expressed in. The
/// acceleration relative to that moving origin is therefore a *difference*
/// of two terms:
///
/// - the **direct term**, GM(s - r)/|s - r|³ — the perturber's attraction on
///   the object, where `s` is the perturber's position relative to
///   `central_body` and `r` is the object's position relative to
///   `central_body`;
/// - the **indirect term**, GM·s/|s|³ — the perturber's attraction on
///   `central_body` itself, which must be subtracted because it also
///   accelerates the (non-inertial) frame origin.
///
/// Two cases use the direct term only, because nothing accelerates the
/// frame origin there:
/// - `CentralBody::SSB`: the Solar System Barycenter is not accelerated by
///   any body's gravity (nothing external to the solar system is in the
///   force model), for any perturber.
/// - `CentralBody::EMB`, for `body` = `ThirdBody::Earth` or `ThirdBody::Moon`
///   only: internal Earth-Moon gravitational forces are equal and opposite
///   (Newton's third law), so neither body can accelerate their own mutual
///   barycenter. Other perturbers (Sun, planets) still use the differential
///   form about `EMB`, since they accelerate the Earth-Moon system as a
///   whole.
///
/// `CentralBody::Custom` always uses the differential form: a general body
/// is, in principle, accelerated by every other body in the force model.
///
/// # Kernel selection
///
/// Unlike [`accel_third_body`] (which always queries the process-wide SPICE
/// kernel registry via [`crate::spice::spk_position`], relying on whatever
/// kernel happens to be loaded), this function loads the DE kernel named by
/// `source` immediately before querying. Since the registry resolves
/// cross-kernel queries using a "most recently loaded kernel wins" rule for
/// overlapping body pairs, this makes `source` actually select DE440 vs.
/// DE440s (or an explicit `SPK` kernel) rather than being silently ignored
/// by the registry's auto-initialize-with-DE440s default. Loading is
/// idempotent, and kernels already loaded for bodies the DE kernel itself
/// doesn't cover (e.g. Phobos/Deimos, which require the `mar099s` satellite
/// kernel) remain loaded and still participate in cross-kernel chain
/// resolution.
///
/// # Arguments
///
/// * `central_body` - Body the object's position and the returned
///   acceleration are expressed relative to
/// * `body` - Perturbing celestial body
/// * `source` - Ephemeris source for the perturber's position.
///   `EphemerisSource::LowPrecision` is only valid when `central_body` is
///   `CentralBody::Earth` and `body` is `ThirdBody::Sun` or `ThirdBody::Moon`
///   (the analytic ephemerides are geocentric)
/// * `epc` - Epoch for ephemeris lookup
/// * `r_object` - Position of the object relative to `central_body` (or a 6D
///   state vector, position only used). Units: [m]
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Acceleration in the inertial frame centered on
///   `central_body`. Units: [m/s²]
/// * `Err(BraheError)` - If `source` is `EphemerisSource::LowPrecision` with
///   a non-Earth `central_body` or a `body` other than `Sun`/`Moon`, or if
///   the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```no_run
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::accel_third_body_for_body;
/// use brahe::propagators::CentralBody;
/// use brahe::propagators::force_model_config::{ThirdBody, EphemerisSource};
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a_sun = accel_third_body_for_body(
///     &CentralBody::Earth,
///     &ThirdBody::Sun,
///     EphemerisSource::DE440s,
///     epc,
///     r_object,
/// ).unwrap();
/// ```
pub fn accel_third_body_for_body<P: IntoPosition>(
    central_body: &CentralBody,
    body: &ThirdBody,
    source: EphemerisSource,
    epc: Epoch,
    r_object: P,
) -> Result<Vector3<f64>, BraheError> {
    if matches!(source, EphemerisSource::LowPrecision) && *central_body != CentralBody::Earth {
        return Err(BraheError::Error(
            "LowPrecision ephemerides are geocentric; use a DE/SPK source for non-Earth central bodies"
                .to_string(),
        ));
    }

    let r = r_object.position();

    let s = match (source, body) {
        (EphemerisSource::LowPrecision, ThirdBody::Sun) => sun_position(epc),
        (EphemerisSource::LowPrecision, ThirdBody::Moon) => moon_position(epc),
        (EphemerisSource::LowPrecision, other) => {
            return Err(BraheError::Error(format!(
                "Low-precision ephemerides only support Sun and Moon. Requested {:?}. \
                Use EphemerisSource::DE440s, DE440, or SPK(...) for other bodies.",
                other
            )));
        }
        (source, body) => {
            // Load the requested DE kernel so it takes precedence over any
            // other DE kernel already loaded for this (target, center) pair
            // (see the "Kernel selection" section above).
            let kernel = SPKKernel::try_from(source)?;
            load_kernel(kernel.name())?;
            spk_position(body.naif_id(), central_body.naif_id(), epc)?
        }
    };

    let gm = body.gm();
    let d = s - r;
    let direct = gm * d / d.norm().powi(3);

    let direct_only = match central_body {
        CentralBody::SSB => true,
        CentralBody::EMB => matches!(body, ThirdBody::Earth | ThirdBody::Moon),
        _ => false,
    };

    Ok(if direct_only {
        direct
    } else {
        direct - gm * s / s.norm().powi(3)
    })
}

/// Calculate the acceleration due to the Sun on an object at a given epoch.
/// The calculation is performed using the point-mass gravity model and the
/// low-precision analytical ephemerides for the Sun position implemented in
/// the `ephemerides` module.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_object`.
/// When a state vector is provided, only the position component is used.
///
/// Should a more accurate calculation be required, you can utilize the
/// point-mass gravity model and a higher-precision ephemerides for the Sun.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Sun's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector (position + velocity). Units: [m]
///
/// # Returns
///
/// * `a` - Acceleration due to the Sun. Units: [m/s^2]
///
/// # Example
///
/// ```
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::accel_third_body_sun;
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = accel_third_body_sun(epc, r_object);
/// ```
pub fn accel_third_body_sun<P: IntoPosition>(epc: Epoch, r_object: P) -> Vector3<f64> {
    accel_third_body(ThirdBody::Sun, EphemerisSource::LowPrecision, epc, r_object)
}

/// Calculate the acceleration due to the Moon on an object at a given epoch.
/// The calculation is performed using the point-mass gravity model and the
/// low-precision analytical ephemerides for the Moon position implemented in
/// the `ephemerides` module.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_object`.
/// When a state vector is provided, only the position component is used.
///
/// Should a more accurate calculation be required, you can utilize the
/// point-mass gravity model and a higher-precision ephemerides for the Moon.
///
/// # Arguments
///
/// - `epc` - Epoch at which to calculate the Moon's position
/// - `r_object` - Position of the object in the GCRF frame, or state vector (position + velocity). Units: [m]
///
/// # Returns
///
/// - `a` - Acceleration due to the Moon. Units: [m/s^2]
///
/// # Example
///
/// ```
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::accel_third_body_moon;
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = accel_third_body_moon(epc, r_object);
/// ```
pub fn accel_third_body_moon<P: IntoPosition>(epc: Epoch, r_object: P) -> Vector3<f64> {
    accel_third_body(
        ThirdBody::Moon,
        EphemerisSource::LowPrecision,
        epc,
        r_object,
    )
}

/// Calculate the acceleration due to the Sun on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// This function uses the NAIF SPK kernel (DE440s or DE440) to compute the Sun's position,
/// providing significantly higher accuracy than the analytical ephemerides used
/// by `accel_third_body_sun`.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Sun's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to the Sun. Units: [m/s^2]
///
/// # Example
///
/// ```
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::accel_third_body_sun_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::constants::R_EARTH;
/// use brahe::ephemerides::initialize_ephemeris;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
/// initialize_ephemeris().unwrap();
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = accel_third_body_sun_de(epc, r_object, EphemerisSource::DE440s);
/// ```
pub fn accel_third_body_sun_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Sun, source, epc, r_object)
}

/// Calculate the acceleration due to the Moon on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// This function uses the NAIF SPK kernel (DE440s or DE440) to compute the Moon's position,
/// providing significantly higher accuracy than the analytical ephemerides used
/// by `accel_third_body_moon`.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Moon's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to the Moon. Units: [m/s^2]
///
/// # Example
///
/// ```
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::time::Epoch;
/// use brahe::third_body::accel_third_body_moon_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::constants::R_EARTH;
/// use brahe::ephemerides::initialize_ephemeris;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
/// initialize_ephemeris().unwrap();
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = accel_third_body_moon_de(epc, r_object, EphemerisSource::DE440s);
/// ```
pub fn accel_third_body_moon_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Moon, source, epc, r_object)
}

/// Calculate the acceleration due to Mercury on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Mercury's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Mercury. Units: [m/s^2]
pub fn accel_third_body_mercury_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Mercury, source, epc, r_object)
}

/// Calculate the acceleration due to Venus on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Venus's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Venus. Units: [m/s^2]
pub fn accel_third_body_venus_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Venus, source, epc, r_object)
}

/// Calculate the acceleration due to Mars on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Mars's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Mars. Units: [m/s^2]
pub fn accel_third_body_mars_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Mars, source, epc, r_object)
}

/// Calculate the acceleration due to Jupiter on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Jupiter's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Jupiter. Units: [m/s^2]
pub fn accel_third_body_jupiter_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Jupiter, source, epc, r_object)
}

/// Calculate the acceleration due to Saturn on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Saturn's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Saturn. Units: [m/s^2]
pub fn accel_third_body_saturn_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Saturn, source, epc, r_object)
}

/// Calculate the acceleration due to Uranus on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Uranus's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Uranus. Units: [m/s^2]
pub fn accel_third_body_uranus_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Uranus, source, epc, r_object)
}

/// Calculate the acceleration due to Neptune on an object at a given epoch using
/// the DE high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for `r_object`.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Neptune's position
/// * `r_object` - Position of the object in the GCRF frame, or state vector. Units: [m]
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `a` - Acceleration due to Neptune. Units: [m/s^2]
pub fn accel_third_body_neptune_de<P: IntoPosition>(
    epc: Epoch,
    r_object: P,
    source: EphemerisSource,
) -> Vector3<f64> {
    accel_third_body(ThirdBody::Neptune, source, epc, r_object)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use crate::TimeSystem;

    use super::*;

    #[rstest]
    #[case(60310.0, 4884992.30378986, 4553508.53744864, 1330313.60479734, - 2.83676856237279e-07, 2.42660636226875e-07, 1.32048201247083e-07)]
    #[case(60310.0, 2670937.8974923, 5898362.79515022, 2124959.71017719, - 2.31115657850035e-07, 4.01378977924412e-07, 1.92039921303102e-07)]
    #[case(60310.0, 38796.9774858514, 6320698.88514676, 2587294.93626938, - 1.42403095448685e-07, 4.97330766046125e-07, 2.21999834460446e-07)]
    #[case(60310.0, - 2599961.45855466, 5760720.19357889, 2647597.12683792, - 3.15422631697234e-08, 5.16014363543264e-07, 2.17465940218504e-07)]
    #[case(60310.0, - 4839229.61832879, 4313760.58255103, 2300338.34996557, 8.42078268885445e-08, 4.55276781915684e-07, 1.79457446434353e-07)]
    #[case(60310.0, - 6342536.88656784, 2209712.29939824, 1602811.60820791, 1.87182166643884e-07, 3.25221817468977e-07, 1.14120959837502e-07)]
    #[case(60310.0, - 6891477.8215365, - 227551.810286937, 663813.896586629, 2.62015981019115e-07, 1.46169083244472e-07, 3.1584377565069e-08)]
    #[case(60310.0, - 6412800.79978623, - 2631381.70900648, - 374371.749654303, 2.97795813116627e-07, - 5.47304407469416e-08, - 5.56848123002086e-08)]
    #[case(60310.0, - 4983679.01699774, - 4645498.60891225, - 1357188.62711648, 2.89444343311254e-07, - 2.4755353121528e-07, - 1.34716105636085e-07)]
    #[case(60310.0, - 2817603.71414268, - 5972669.87274763, - 2139313.41892538, 2.38284972069453e-07, - 4.03777237061026e-07, - 1.93828712473536e-07)]
    #[case(60310.0, - 234236.587976406, - 6414628.84861909, - 2604335.85309436, 1.51812336070211e-07, - 5.00147992775384e-07, - 2.24210512692769e-07)]
    #[case(60310.0, 2383524.57084058, - 5900075.61185268, - 2680956.18196418, 4.27172228608481e-08, - 5.21918073677521e-07, - 2.21152074692176e-07)]
    #[case(60310.0, 4641862.77023787, - 4497734.59354263, - 2354086.60315067, - 7.27753469123527e-08, - 4.65126279082538e-07, - 1.84808262415947e-07)]
    #[case(60310.0, 6193136.2430559, - 2411369.56203787, - 1669079.86356028, - 1.77151028990413e-07, - 3.3756567861856e-07, - 1.20350830019883e-07)]
    #[case(60310.0, 6790850.71407875, 45505.4274329756, - 727399.838172203, - 2.54224503688731e-07, - 1.580949129909e-07, - 3.73927783617869e-08)]
    #[case(60310.0, 6333183.86841522, 2494761.03873549, 327102.634966258, - 2.91770539678173e-07, 4.58908225325491e-08, 5.13518087266698e-08)]
    fn test_accel_third_body_sun(
        #[case] mjd_tt: f64,
        #[case] rx: f64,
        #[case] ry: f64,
        #[case] rz: f64,
        #[case] ax: f64,
        #[case] ay: f64,
        #[case] az: f64,
    ) {
        let epc = Epoch::from_mjd(mjd_tt, TimeSystem::TT);
        let r_object = Vector3::new(rx, ry, rz);

        let a = accel_third_body_sun(epc, r_object);

        assert_abs_diff_eq!(a[0], ax, epsilon = 1e-9);
        assert_abs_diff_eq!(a[1], ay, epsilon = 1e-9);
        assert_abs_diff_eq!(a[2], az, epsilon = 1e-9);
    }

    #[rstest]
    #[case(60310.0, 4884992.30378986, 4553508.53744864, 1330313.60479734, 1.62360236246851e-07, - 5.30930401572647e-07, - 2.22022756088401e-07)]
    #[case(60310.0, 2670937.8974923, 5898362.79515022, 2124959.71017719, - 2.10084628821528e-07, - 4.31933921171218e-07, - 1.54339381002608e-07)]
    #[case(60310.0, 38796.9774858514, 6320698.88514676, 2587294.93626938, - 5.58483235850665e-07, - 2.6203733817308e-07, - 6.05903753125981e-08)]
    #[case(60310.0, - 2599961.45855466, 5760720.19357889, 2647597.12683792, - 8.25046337841761e-07, - 4.53028242796273e-08, 4.53066427075969e-08)]
    #[case(60310.0, - 4839229.61832879, 4313760.58255103, 2300338.34996557, - 9.63108738027384e-07, 1.83858250202633e-07, 1.4622908513799e-07)]
    #[case(60310.0, - 6342536.88656784, 2209712.29939824, 1602811.60820791, - 9.48011832170594e-07, 3.86674684929409e-07, 2.25026995803795e-07)]
    #[case(60310.0, - 6891477.8215365, - 227551.810286937, 663813.896586629, - 7.83191277225506e-07, 5.28327949832493e-07, 2.68246894531318e-07)]
    #[case(60310.0, - 6412800.79978623, - 2631381.70900648, - 374371.749654303, - 4.98912678830928e-07, 5.85738566093379e-07, 2.6909714100787e-07)]
    #[case(60310.0, - 4983679.01699774, - 4645498.60891225, - 1357188.62711648, - 1.44380586166042e-07, 5.51955765893565e-07, 2.28583689612585e-07)]
    #[case(60310.0, - 2817603.71414268, - 5972669.87274763, - 2139313.41892538, 2.23328070379479e-07, 4.35988467235581e-07, 1.54614554610566e-07)]
    #[case(60310.0, - 234236.587976406, - 6414628.84861909, - 2604335.85309436, 5.49604398045391e-07, 2.59053532360343e-07, 5.97432211499815e-08)]
    #[case(60310.0, 2383524.57084058, - 5900075.61185268, - 2680956.18196418, 7.89599228288718e-07, 4.93434256460948e-08, - 4.14510695387178e-08)]
    #[case(60310.0, 4641862.77023787, - 4497734.59354263, - 2354086.60315067, 9.12218233923866e-07, - 1.62830886237673e-07, - 1.34595434862506e-07)]
    #[case(60310.0, 6193136.2430559, - 2411369.56203787, - 1669079.86356028, 9.01868211930885e-07, - 3.48656149518958e-07, - 2.07100394322338e-07)]
    #[case(60310.0, 6790850.71407875, 45505.4274329756, - 727399.838172203, 7.59196602766636e-07, - 4.83281433661868e-07, - 2.49203881536061e-07)]
    #[case(60310.0, 6333183.86841522, 2494761.03873549, 327102.634966258, 5.01475600782815e-07, - 5.47736810287354e-07, - 2.54764046632745e-07)]
    fn test_accel_third_body_moon(
        #[case] mjd_tt: f64,
        #[case] rx: f64,
        #[case] ry: f64,
        #[case] rz: f64,
        #[case] ax: f64,
        #[case] ay: f64,
        #[case] az: f64,
    ) {
        let epc = Epoch::from_mjd(mjd_tt, TimeSystem::TT);
        let r_object = Vector3::new(rx, ry, rz);

        let a = accel_third_body_moon(epc, r_object);

        assert_abs_diff_eq!(a[0], ax, epsilon = 1e-9);
        assert_abs_diff_eq!(a[1], ay, epsilon = 1e-9);
        assert_abs_diff_eq!(a[2], az, epsilon = 1e-9);
    }

    use crate::utils::testing::setup_global_test_spice;

    #[test]
    fn test_accel_third_body_sun_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_sun_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-5); // Should be on order of 1e-6 to 1e-7 m/s^2
    }

    #[test]
    fn test_accel_third_body_moon_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_moon_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-5); // Should be on order of 1e-6 to 1e-7 m/s^2
    }

    #[test]
    fn test_accel_third_body_mercury_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_mercury_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector (very small for Mercury)
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-10); // Mercury effect is very small
    }

    #[test]
    fn test_accel_third_body_venus_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_venus_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-9); // Venus effect is small
    }

    #[test]
    fn test_accel_third_body_mars_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_mars_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-10); // Mars effect is very small
    }

    #[test]
    fn test_accel_third_body_jupiter_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_jupiter_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-9); // Jupiter effect is relatively larger but still small
    }

    #[test]
    fn test_accel_third_body_saturn_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_saturn_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-10); // Saturn effect is small
    }

    #[test]
    fn test_accel_third_body_uranus_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_uranus_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-11); // Uranus effect is very small
    }

    #[test]
    fn test_accel_third_body_neptune_de() {
        setup_global_test_spice();

        let epc = Epoch::from_mjd(60310.0, TimeSystem::TT);
        let r_object = Vector3::new(4884992.30378986, 4553508.53744864, 1330313.60479734);

        let a = accel_third_body_neptune_de(epc, r_object, EphemerisSource::DE440s);

        // Should return a valid acceleration vector
        assert!(a.norm() > 0.0);
        assert!(a.norm() < 1e-11); // Neptune effect is very small
    }

    use crate::{GM_EARTH, R_EARTH, R_MARS};
    use serial_test::serial;

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_accel_third_body_for_body_earth_center_matches_legacy() {
        setup_global_test_spice();

        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let legacy = accel_third_body(ThirdBody::Sun, EphemerisSource::DE440s, epc, r);
        let new = accel_third_body_for_body(
            &CentralBody::Earth,
            &ThirdBody::Sun,
            EphemerisSource::DE440s,
            epc,
            r,
        )
        .unwrap();

        for i in 0..3 {
            assert_abs_diff_eq!(new[i], legacy[i], epsilon = 1e-15);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_emb_internal_bodies_use_direct_form() {
        setup_global_test_spice();

        // For EMB center, Earth's contribution at position r is exactly
        // GM_EARTH * (s - r)/|s - r|^3 with NO indirect term, since internal
        // Earth-Moon forces cannot accelerate their own barycenter.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(1e8, 2e8, -5e7);
        let s = crate::spice::spk_position(399, 3, epc).unwrap();
        let d = s - r;
        let expected = GM_EARTH * d / d.norm().powi(3);

        let got = accel_third_body_for_body(
            &CentralBody::EMB,
            &ThirdBody::Earth,
            EphemerisSource::DE440s,
            epc,
            r,
        )
        .unwrap();

        for i in 0..3 {
            assert_abs_diff_eq!(got[i], expected[i], epsilon = 1e-18);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_emb_external_body_uses_differential_form() {
        setup_global_test_spice();

        // Sun about EMB: differential form (direct minus indirect-at-EMB),
        // since the Sun does accelerate the Earth-Moon barycenter.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(1e8, 2e8, -5e7);
        let s = crate::spice::spk_position(10, 3, epc).unwrap();
        let d = s - r;
        let expected = GM_SUN * (d / d.norm().powi(3) - s / s.norm().powi(3));

        let got = accel_third_body_for_body(
            &CentralBody::EMB,
            &ThirdBody::Sun,
            EphemerisSource::DE440s,
            epc,
            r,
        )
        .unwrap();

        for i in 0..3 {
            assert_abs_diff_eq!(got[i], expected[i], epsilon = 1e-18);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_ssb_all_bodies_direct() {
        setup_global_test_spice();

        // Nothing accelerates the Solar System Barycenter, so every
        // perturber (here Jupiter) uses the direct term only.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(1e8, 2e8, -5e7);
        let s = crate::spice::spk_position(5, 0, epc).unwrap();
        let d = s - r;
        let expected = GM_JUPITER * d / d.norm().powi(3);

        let got = accel_third_body_for_body(
            &CentralBody::SSB,
            &ThirdBody::Jupiter,
            EphemerisSource::DE440s,
            epc,
            r,
        )
        .unwrap();

        for i in 0..3 {
            assert_abs_diff_eq!(got[i], expected[i], epsilon = 1e-18);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_phobos_third_body_about_mars() {
        setup_global_test_spice();
        crate::spice::load_kernel("mar099s").unwrap();

        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = Vector3::new(R_MARS + 400e3, 0.0, 0.0);

        let a = accel_third_body_for_body(
            &CentralBody::Mars,
            &ThirdBody::Phobos,
            EphemerisSource::DE440s,
            epc,
            r,
        )
        .unwrap();

        // Phobos is tiny (GM_PHOBOS ~ 7e5 m^3/s^2), so its perturbation on a
        // Mars-orbiting object is small but nonzero.
        assert!(a.norm() > 0.0 && a.norm() < 1e-8);
    }

    #[test]
    fn test_low_precision_rejected_for_non_earth_center() {
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let e = accel_third_body_for_body(
            &CentralBody::Moon,
            &ThirdBody::Sun,
            EphemerisSource::LowPrecision,
            epc,
            Vector3::new(2e6, 0.0, 0.0),
        );
        assert!(e.is_err());
    }
}
