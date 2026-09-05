/*!
 * Equinox-based Earth reference frame transformations between the GCRF,
 * the mean equator and equinox of date (MOD), the true equator and equinox
 * of date (TOD), and the ITRF.
 *
 * The chain is `[ITRF] = W R3(GAST) N P B [GCRF]` (SOFA cookbook Section
 * 2.9 and Appendix p. A4), evaluated on the IAU 2000 precession and IAU
 * 2000B nutation basis of `iauPn00b`, the same basis the CIO-based chain in
 * [`super::gcrf_itrf`] uses through `iauXys00b`. `P B [GCRF]` is MOD and
 * `N P B [GCRF]` is TOD. IERS celestial pole offsets enter the nutation
 * step as dPsi/dEps corrections (cookbook Section 5.4), and Greenwich
 * apparent sidereal time is `ERA - EO` with the equation of the origins
 * evaluated from the corrected NPB matrix, so that `R3(GAST) N P B` equals
 * the CIO form `R3(ERA) C` for the same CIP coordinates.
 */
use nalgebra::Vector3;

use crate::constants;
use crate::constants::MJD_ZERO;
use crate::eop;
use crate::frames::gcrf_itrf::polar_motion;
use crate::math::{SMatrix3, SVector6, matrix3_from_array};
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_map_epochs};

/// Equinox-based precession-nutation products for one epoch, computed once
/// and shared by the pairwise and batch transformations.
pub(crate) struct EquinoxContext {
    /// MOD -> TOD nutation matrix, including the IERS dPsi/dEps corrections.
    pub(crate) rn: SMatrix3,
    /// GCRF -> TOD matrix (`rn * rbp`).
    pub(crate) rnpb: SMatrix3,
    /// Greenwich apparent sidereal time, `ERA - EO`. Units: (*rad*)
    pub(crate) gast: f64,
}

impl EquinoxContext {
    /// Evaluates the IAU 2000/2000B equinox-based chain at `epc`.
    ///
    /// # Arguments
    /// - `epc`: Epoch of the transformation
    ///
    /// # Returns
    /// - `EquinoxContext`: Bias-precession, corrected nutation, combined
    ///   GCRF -> TOD matrix, and Greenwich apparent sidereal time
    ///
    /// # Panics
    /// Panics if Earth orientation data is unavailable for the requested
    /// epoch, matching [`super::gcrf_itrf::bias_precession_nutation`].
    ///
    /// # References
    /// - SOFA cookbook Section 5.4 (correction conversion), 2.7 (equation of
    ///   the origins); SOFA `pn00b` notes 2-9, `numat` note 3, `eors` notes 1-2
    #[allow(non_snake_case)]
    pub(crate) fn new(epc: Epoch) -> Self {
        let tt = epc.mjd_as_time_system(TimeSystem::TT);
        let ut1 = epc.mjd_as_time_system(TimeSystem::UT1);

        let mut dpsi = 0.0;
        let mut deps = 0.0;
        let mut epsa = 0.0;
        let mut rb = [[0.0; 3]; 3];
        let mut rp = [[0.0; 3]; 3];
        let mut rbp = [[0.0; 3]; 3];
        let mut rn_model = [[0.0; 3]; 3];
        let mut rbpn_model = [[0.0; 3]; 3];
        unsafe {
            rsofa::iauPn00b(
                MJD_ZERO,
                tt,
                &mut dpsi,
                &mut deps,
                &mut epsa,
                &mut rb[0],
                &mut rp[0],
                &mut rbp[0],
                &mut rn_model[0],
                &mut rbpn_model[0],
            );
        }

        let (dX, dY) = eop::get_global_dxdy(epc.mjd_as_time_system(TimeSystem::UTC))
            .unwrap_or_else(|e| {
                panic!(
                    "EOP dX/dY corrections unavailable for epoch {} ({}); initialize a global EOP \
                     provider covering this epoch or set extrapolation to Hold or Zero",
                    epc, e
                )
            });

        // Celestial pole offsets from GCRF axes into the of-date equatorial
        // frame, then into nutation-angle corrections.
        let mut v_gcrf = [dX, dY, 0.0];
        let mut v_date = [0.0; 3];
        unsafe {
            rsofa::iauRxp(&mut rbpn_model[0], &mut v_gcrf[0], &mut v_date[0]);
        }
        let ddpsi = v_date[0] / epsa.sin();
        let ddeps = v_date[1];

        let mut rn = [[0.0; 3]; 3];
        let mut rnpb = [[0.0; 3]; 3];
        let mut x = 0.0;
        let mut y = 0.0;
        let gast;
        unsafe {
            rsofa::iauNumat(epsa, dpsi + ddpsi, deps + ddeps, &mut rn[0]);
            rsofa::iauRxr(&mut rn[0], &mut rbp[0], &mut rnpb[0]);
            rsofa::iauBpn2xy(&mut rnpb[0], &mut x, &mut y);
            let s = rsofa::iauS00(MJD_ZERO, tt, x, y);
            let eo = rsofa::iauEors(&mut rnpb[0], s);
            let era = rsofa::iauEra00(MJD_ZERO, ut1);
            gast = rsofa::iauAnp(era - eo);
        }

        Self {
            rn: matrix3_from_array(&rn),
            rnpb: matrix3_from_array(&rnpb),
            gast,
        }
    }

    /// Rotation about the z axis by GAST (TOD -> TIRS).
    pub(crate) fn sidereal_rotation(&self) -> SMatrix3 {
        let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        unsafe {
            rsofa::iauRz(self.gast, &mut r[0]);
        }
        matrix3_from_array(&r)
    }
}

/// Computes the IAU 2000 bias-precession matrix `rp * rb` at `tt` using
/// `iauBp00` directly, without evaluating the nutation corrections that
/// require Earth orientation data.
///
/// This is the same `rbp` that `iauPn00b` returns internally, since
/// `iauPn00b` obtains it by calling `iauBp00`.
///
/// # Arguments
/// - `tt`: TT as a Modified Julian Date. Units: (*days*)
///
/// # Returns
/// - `rbp`: 3x3 rotation matrix transforming GCRF -> MOD
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 (`B`, `P` rows)
fn bias_precession_matrix(tt: f64) -> SMatrix3 {
    let mut rb = [[0.0; 3]; 3];
    let mut rp = [[0.0; 3]; 3];
    let mut rbp = [[0.0; 3]; 3];
    unsafe {
        rsofa::iauBp00(MJD_ZERO, tt, &mut rb[0], &mut rp[0], &mut rbp[0]);
    }
    matrix3_from_array(&rbp)
}

/// Computes the bias-precession matrix transforming the GCRF to the mean
/// equator and equinox of date (MOD) using the IAU 2000 precession model.
///
/// The matrix is `P B` where `B` is the frame bias and `P` the precession
/// from J2000.0 to date, so MOD is rooted at the GCRF with the bias applied
/// explicitly. It does not depend on Earth orientation corrections.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `rbp`: 3x3 rotation matrix transforming GCRF -> MOD
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let rbp = bias_precession(epc);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 (`B`, `P` rows)
pub fn bias_precession(epc: Epoch) -> SMatrix3 {
    bias_precession_matrix(epc.mjd_as_time_system(TimeSystem::TT))
}

/// Computes the nutation matrix transforming the mean equator and equinox
/// of date (MOD) to the true equator and equinox of date (TOD) using the
/// IAU 2000B nutation series with IERS celestial pole offset corrections.
///
/// The global Earth orientation dX/dY corrections are rotated into the
/// of-date equatorial frame and converted to dPsi/dEps (`ddpsi = v_x /
/// sin(eps_A)`, `ddeps = v_y`) before forming `N = R1(-(eps_A + deps))
/// R3(-dpsi) R1(eps_A)`.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `rn`: 3x3 rotation matrix transforming MOD -> TOD
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let rn = nutation(epc);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn nutation(epc: Epoch) -> SMatrix3 {
    EquinoxContext::new(epc).rn
}

/// Computes the Earth rotation matrix `R3(GAST)` transforming the true
/// equator and equinox of date (TOD) to the Terrestrial Intermediate
/// Reference System (TIRS).
///
/// Greenwich apparent sidereal time is `ERA - EO`, with the equation of the
/// origins evaluated from the corrected GCRF -> TOD matrix, so that
/// `R3(GAST) N P B` equals the CIO-based `R3(ERA) C` for the same CIP.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming TOD -> TIRS
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = greenwich_apparent_sidereal_rotation(epc);
/// ```
///
/// # References
/// - SOFA `eors` notes 1-2, `gst06` note 3; SOFA cookbook Sections 2.7 and
///   3.6; Wallace & Capitaine 2006, A&A 459, 981
pub fn greenwich_apparent_sidereal_rotation(epc: Epoch) -> SMatrix3 {
    EquinoxContext::new(epc).sidereal_rotation()
}

/// Computes the rotation matrix transforming the GCRF to the mean equator
/// and equinox of date (MOD) using the IAU 2000 precession model.
///
/// The matrix is `P B` where `B` is the frame bias and `P` the precession
/// from J2000.0 to date. It does not depend on Earth orientation
/// corrections.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming GCRF -> MOD
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_gcrf_to_mod(epc);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn rotation_gcrf_to_mod(epc: Epoch) -> SMatrix3 {
    bias_precession(epc)
}

/// Computes the rotation matrix transforming the mean equator and equinox
/// of date (MOD) to the GCRF: the transpose of [`rotation_gcrf_to_mod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming MOD -> GCRF
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_mod_to_gcrf(epc);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn rotation_mod_to_gcrf(epc: Epoch) -> SMatrix3 {
    rotation_gcrf_to_mod(epc).transpose()
}

/// Computes the rotation matrix transforming the mean equator and equinox
/// of date (MOD) to the true equator and equinox of date (TOD) using the
/// IAU 2000B nutation series with IERS celestial pole offset corrections.
///
/// The global Earth orientation dX/dY corrections are rotated into the
/// of-date equatorial frame and converted to dPsi/dEps (`ddpsi = v_x /
/// sin(eps_A)`, `ddeps = v_y`) before forming `N = R1(-(eps_A + deps))
/// R3(-dpsi) R1(eps_A)`.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming MOD -> TOD
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_mod_to_tod(epc);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn rotation_mod_to_tod(epc: Epoch) -> SMatrix3 {
    nutation(epc)
}

/// Computes the rotation matrix transforming the true equator and equinox
/// of date (TOD) to the mean equator and equinox of date (MOD): the
/// transpose of [`rotation_mod_to_tod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming TOD -> MOD
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_tod_to_mod(epc);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn rotation_tod_to_mod(epc: Epoch) -> SMatrix3 {
    rotation_mod_to_tod(epc).transpose()
}

/// Computes the rotation matrix transforming the GCRF to the true equator
/// and equinox of date (TOD): `N P B`, bias, precession, and corrected
/// nutation applied in that order.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming GCRF -> TOD
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_gcrf_to_tod(epc);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn rotation_gcrf_to_tod(epc: Epoch) -> SMatrix3 {
    EquinoxContext::new(epc).rnpb
}

/// Computes the rotation matrix transforming the true equator and equinox
/// of date (TOD) to the GCRF: the transpose of [`rotation_gcrf_to_tod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming TOD -> GCRF
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_tod_to_gcrf(epc);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn rotation_tod_to_gcrf(epc: Epoch) -> SMatrix3 {
    rotation_gcrf_to_tod(epc).transpose()
}

/// Computes the rotation matrix transforming the true equator and equinox
/// of date (TOD) to the ITRF: polar motion applied after the Greenwich
/// apparent sidereal rotation, `W R3(GAST)`.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming TOD -> ITRF
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_tod_to_itrf(epc);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn rotation_tod_to_itrf(epc: Epoch) -> SMatrix3 {
    polar_motion(epc) * EquinoxContext::new(epc).sidereal_rotation()
}

/// Computes the rotation matrix transforming the ITRF to the true equator
/// and equinox of date (TOD): the transpose of [`rotation_tod_to_itrf`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming ITRF -> TOD
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_itrf_to_tod(epc);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn rotation_itrf_to_tod(epc: Epoch) -> SMatrix3 {
    rotation_tod_to_itrf(epc).transpose()
}

/// Applies a rotation to both the position and velocity halves of a state.
fn rotate_state(r: &SMatrix3, x: &SVector6) -> SVector6 {
    let p: Vector3<f64> = r * x.fixed_rows::<3>(0);
    let v: Vector3<f64> = r * x.fixed_rows::<3>(3);
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Rotation and polar-motion matrices for the TOD <-> ITRF state transforms.
struct TodItrfContext {
    r: SMatrix3,
    pm: SMatrix3,
}

/// Computes the sidereal-rotation and polar-motion matrices for `epc`.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrices
///
/// # Returns
/// - Context holding the `R3(GAST)` and polar-motion matrices
fn tod_itrf_context(epc: Epoch) -> TodItrfContext {
    TodItrfContext {
        r: EquinoxContext::new(epc).sidereal_rotation(),
        pm: polar_motion(epc),
    }
}

/// Apply a precomputed TOD-to-ITRF context to one Cartesian TOD state.
///
/// # Arguments
/// - `c`: Transformation matrices for the epoch
/// - `x_tod`: Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
fn apply_state_tod_to_itrf(c: &TodItrfContext, x_tod: &SVector6) -> SVector6 {
    let omega_vec = Vector3::new(0.0, 0.0, constants::OMEGA_EARTH);
    let r_tod = x_tod.fixed_rows::<3>(0);
    let v_tod = x_tod.fixed_rows::<3>(3);
    let p: Vector3<f64> = c.pm * c.r * r_tod;
    let v: Vector3<f64> = c.pm * (c.r * v_tod - omega_vec.cross(&(c.r * r_tod)));
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Apply a precomputed TOD-to-ITRF context to one Cartesian ITRF state,
/// producing the TOD state.
///
/// # Arguments
/// - `c`: Transformation matrices for the epoch
/// - `x_itrf`: Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
fn apply_state_itrf_to_tod(c: &TodItrfContext, x_itrf: &SVector6) -> SVector6 {
    let omega_vec = Vector3::new(0.0, 0.0, constants::OMEGA_EARTH);
    let r_itrf = x_itrf.fixed_rows::<3>(0);
    let v_itrf = x_itrf.fixed_rows::<3>(3);
    let p: Vector3<f64> = (c.pm * c.r).transpose() * r_itrf;
    let v: Vector3<f64> = c.r.transpose()
        * (c.pm.transpose() * v_itrf + omega_vec.cross(&(c.pm.transpose() * r_itrf)));
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Transforms a Cartesian position in the GCRF to the equivalent position
/// in the mean equator and equinox of date (MOD).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - Cartesian MOD position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_gcrf = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_mod = position_gcrf_to_mod(epc, x_gcrf);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn position_gcrf_to_mod(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_gcrf_to_mod(epc) * x
}

/// Transforms a Cartesian position in the mean equator and equinox of date
/// (MOD) to the equivalent position in the GCRF: the inverse of
/// [`position_gcrf_to_mod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian MOD position. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_mod = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_gcrf = position_mod_to_gcrf(epc, x_mod);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn position_mod_to_gcrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_mod_to_gcrf(epc) * x
}

/// Transforms a Cartesian state in the GCRF to the equivalent state in the
/// mean equator and equinox of date (MOD).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_gcrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_mod = state_gcrf_to_mod(epc, x_gcrf);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn state_gcrf_to_mod(epc: Epoch, x_gcrf: SVector6) -> SVector6 {
    rotate_state(&rotation_gcrf_to_mod(epc), &x_gcrf)
}

/// Transforms a Cartesian state in the mean equator and equinox of date
/// (MOD) to the equivalent state in the GCRF: the inverse of
/// [`state_gcrf_to_mod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mod`: Cartesian MOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_mod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_gcrf = state_mod_to_gcrf(epc, x_mod);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn state_mod_to_gcrf(epc: Epoch, x_mod: SVector6) -> SVector6 {
    rotate_state(&rotation_mod_to_gcrf(epc), &x_mod)
}

/// Computes the GCRF-to-MOD rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gcrf_to_mod`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GCRF -> MOD, one per epoch, in input order
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_gcrf_to_mod(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn rotations_gcrf_to_mod(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_gcrf_to_mod(*epc), epochs)
}

/// Computes the MOD-to-GCRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_mod_to_gcrf`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming MOD -> GCRF, one per epoch, in input order
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_mod_to_gcrf(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn rotations_mod_to_gcrf(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_mod_to_gcrf(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from GCRF to MOD.
///
/// Batch form of [`position_gcrf_to_mod`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian GCRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian MOD positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![
///     vector3_from_array([R_EARTH, 0.0, 0.0]),
///     vector3_from_array([0.0, R_EARTH, 0.0]),
/// ];
///
/// // One epoch, many positions
/// let x_mod = positions_gcrf_to_mod(&[epc], &positions).unwrap();
/// assert_eq!(x_mod.len(), 2);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn positions_gcrf_to_mod(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_gcrf_to_mod, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian positions from MOD to GCRF.
///
/// Batch form of [`position_mod_to_gcrf`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian MOD positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let x = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // One position, many epochs
/// let x_gcrf = positions_mod_to_gcrf(&epochs, &[x]).unwrap();
/// assert_eq!(x_gcrf.len(), 3);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn positions_mod_to_gcrf(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_mod_to_gcrf, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian states from GCRF to MOD.
///
/// Batch form of [`state_gcrf_to_mod`]. `epochs` and `x_gcrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MOD states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_gcrf` do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let states = vec![
///     vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]),
///     vector6_from_array([0.0, R_EARTH + 500e3, 0.0, -v, 0.0, 0.0]),
/// ];
///
/// // One epoch, many states
/// let x_mod = states_gcrf_to_mod(&[epc], &states).unwrap();
/// assert_eq!(x_mod.len(), 2);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn states_gcrf_to_mod(
    epochs: &[Epoch],
    x_gcrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_gcrf_to_mod, rotate_state, epochs, x_gcrf)
}

/// Transforms a batch of Cartesian states from MOD to GCRF.
///
/// Batch form of [`state_mod_to_gcrf`]. `epochs` and `x_mod` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mod`: Cartesian MOD states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_mod` do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let x_mod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]);
///
/// // One state, many epochs
/// let epochs = vec![epc, epc + 60.0];
/// let x_gcrf = states_mod_to_gcrf(&epochs, &[x_mod]).unwrap();
/// assert_eq!(x_gcrf.len(), 2);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 ("GCRF to MOD")
pub fn states_mod_to_gcrf(
    epochs: &[Epoch],
    x_mod: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_mod_to_gcrf, rotate_state, epochs, x_mod)
}

/// Transforms a Cartesian position in the mean equator and equinox of date
/// (MOD) to the equivalent position in the true equator and equinox of date
/// (TOD).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian MOD position. Units: (*m*)
///
/// # Returns
/// - Cartesian TOD position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_mod = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_tod = position_mod_to_tod(epc, x_mod);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn position_mod_to_tod(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_mod_to_tod(epc) * x
}

/// Transforms a Cartesian position in the true equator and equinox of date
/// (TOD) to the equivalent position in the mean equator and equinox of
/// date (MOD): the inverse of [`position_mod_to_tod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian TOD position. Units: (*m*)
///
/// # Returns
/// - Cartesian MOD position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_tod = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_mod = position_tod_to_mod(epc, x_tod);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn position_tod_to_mod(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_tod_to_mod(epc) * x
}

/// Transforms a Cartesian state in the mean equator and equinox of date
/// (MOD) to the equivalent state in the true equator and equinox of date
/// (TOD).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mod`: Cartesian MOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_mod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_tod = state_mod_to_tod(epc, x_mod);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn state_mod_to_tod(epc: Epoch, x_mod: SVector6) -> SVector6 {
    rotate_state(&rotation_mod_to_tod(epc), &x_mod)
}

/// Transforms a Cartesian state in the true equator and equinox of date
/// (TOD) to the equivalent state in the mean equator and equinox of date
/// (MOD): the inverse of [`state_mod_to_tod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_tod`: Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_tod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_mod = state_tod_to_mod(epc, x_tod);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn state_tod_to_mod(epc: Epoch, x_tod: SVector6) -> SVector6 {
    rotate_state(&rotation_tod_to_mod(epc), &x_tod)
}

/// Computes the MOD-to-TOD rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_mod_to_tod`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming MOD -> TOD, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_mod_to_tod(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn rotations_mod_to_tod(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_mod_to_tod(*epc), epochs)
}

/// Computes the TOD-to-MOD rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_tod_to_mod`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming TOD -> MOD, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_tod_to_mod(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn rotations_tod_to_mod(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_tod_to_mod(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from MOD to TOD.
///
/// Batch form of [`position_mod_to_tod`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian MOD positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian TOD positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![
///     vector3_from_array([R_EARTH, 0.0, 0.0]),
///     vector3_from_array([0.0, R_EARTH, 0.0]),
/// ];
///
/// // One epoch, many positions
/// let x_tod = positions_mod_to_tod(&[epc], &positions).unwrap();
/// assert_eq!(x_tod.len(), 2);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn positions_mod_to_tod(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_mod_to_tod, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian positions from TOD to MOD.
///
/// Batch form of [`position_tod_to_mod`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian TOD positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian MOD positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let x = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // One position, many epochs
/// let x_mod = positions_tod_to_mod(&epochs, &[x]).unwrap();
/// assert_eq!(x_mod.len(), 3);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn positions_tod_to_mod(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_tod_to_mod, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian states from MOD to TOD.
///
/// Batch form of [`state_mod_to_tod`]. `epochs` and `x_mod` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mod`: Cartesian MOD states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_mod` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let states = vec![
///     vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]),
///     vector6_from_array([0.0, R_EARTH + 500e3, 0.0, -v, 0.0, 0.0]),
/// ];
///
/// // One epoch, many states
/// let x_tod = states_mod_to_tod(&[epc], &states).unwrap();
/// assert_eq!(x_tod.len(), 2);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn states_mod_to_tod(
    epochs: &[Epoch],
    x_mod: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_mod_to_tod, rotate_state, epochs, x_mod)
}

/// Transforms a batch of Cartesian states from TOD to MOD.
///
/// Batch form of [`state_tod_to_mod`]. `epochs` and `x_tod` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_tod`: Cartesian TOD states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MOD states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_tod` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let x_tod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]);
///
/// // One state, many epochs
/// let epochs = vec![epc, epc + 60.0];
/// let x_mod = states_tod_to_mod(&epochs, &[x_tod]).unwrap();
/// assert_eq!(x_mod.len(), 2);
/// ```
///
/// # References
/// - SOFA cookbook Section 5.4 p. 23 (correction conversion) and Section
///   3.2 (classical nutation); SOFA `numat` note 3, `pn00b` notes 2, 3, 7;
///   Capitaine & Wallace 2006, A&A 450, 855
pub fn states_tod_to_mod(
    epochs: &[Epoch],
    x_tod: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_tod_to_mod, rotate_state, epochs, x_tod)
}

/// Transforms a Cartesian position in the GCRF to the equivalent position
/// in the true equator and equinox of date (TOD).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - Cartesian TOD position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_gcrf = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_tod = position_gcrf_to_tod(epc, x_gcrf);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn position_gcrf_to_tod(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_gcrf_to_tod(epc) * x
}

/// Transforms a Cartesian position in the true equator and equinox of date
/// (TOD) to the equivalent position in the GCRF: the inverse of
/// [`position_gcrf_to_tod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian TOD position. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_tod = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_gcrf = position_tod_to_gcrf(epc, x_tod);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn position_tod_to_gcrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_tod_to_gcrf(epc) * x
}

/// Transforms a Cartesian state in the GCRF to the equivalent state in the
/// true equator and equinox of date (TOD).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_gcrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_tod = state_gcrf_to_tod(epc, x_gcrf);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn state_gcrf_to_tod(epc: Epoch, x_gcrf: SVector6) -> SVector6 {
    rotate_state(&rotation_gcrf_to_tod(epc), &x_gcrf)
}

/// Transforms a Cartesian state in the true equator and equinox of date
/// (TOD) to the equivalent state in the GCRF: the inverse of
/// [`state_gcrf_to_tod`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_tod`: Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_tod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_gcrf = state_tod_to_gcrf(epc, x_tod);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn state_tod_to_gcrf(epc: Epoch, x_tod: SVector6) -> SVector6 {
    rotate_state(&rotation_tod_to_gcrf(epc), &x_tod)
}

/// Computes the GCRF-to-TOD rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gcrf_to_tod`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GCRF -> TOD, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_gcrf_to_tod(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn rotations_gcrf_to_tod(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_gcrf_to_tod(*epc), epochs)
}

/// Computes the TOD-to-GCRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_tod_to_gcrf`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming TOD -> GCRF, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_tod_to_gcrf(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn rotations_tod_to_gcrf(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_tod_to_gcrf(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from GCRF to TOD.
///
/// Batch form of [`position_gcrf_to_tod`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian GCRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian TOD positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![
///     vector3_from_array([R_EARTH, 0.0, 0.0]),
///     vector3_from_array([0.0, R_EARTH, 0.0]),
/// ];
///
/// // One epoch, many positions
/// let x_tod = positions_gcrf_to_tod(&[epc], &positions).unwrap();
/// assert_eq!(x_tod.len(), 2);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn positions_gcrf_to_tod(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_gcrf_to_tod, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian positions from TOD to GCRF.
///
/// Batch form of [`position_tod_to_gcrf`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian TOD positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let x = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // One position, many epochs
/// let x_gcrf = positions_tod_to_gcrf(&epochs, &[x]).unwrap();
/// assert_eq!(x_gcrf.len(), 3);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn positions_tod_to_gcrf(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_tod_to_gcrf, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian states from GCRF to TOD.
///
/// Batch form of [`state_gcrf_to_tod`]. `epochs` and `x_gcrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_gcrf` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let states = vec![
///     vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]),
///     vector6_from_array([0.0, R_EARTH + 500e3, 0.0, -v, 0.0, 0.0]),
/// ];
///
/// // One epoch, many states
/// let x_tod = states_gcrf_to_tod(&[epc], &states).unwrap();
/// assert_eq!(x_tod.len(), 2);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn states_gcrf_to_tod(
    epochs: &[Epoch],
    x_gcrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_gcrf_to_tod, rotate_state, epochs, x_gcrf)
}

/// Transforms a batch of Cartesian states from TOD to GCRF.
///
/// Batch form of [`state_tod_to_gcrf`]. `epochs` and `x_tod` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_tod`: Cartesian TOD states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_tod` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let x_tod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]);
///
/// // One state, many epochs
/// let epochs = vec![epc, epc + 60.0];
/// let x_gcrf = states_tod_to_gcrf(&epochs, &[x_tod]).unwrap();
/// assert_eq!(x_gcrf.len(), 2);
/// ```
///
/// # References
/// - SOFA `pn00b` note 8; SOFA cookbook Section 2.9 and Appendix p. A4
///   ("NPB: GCRS -> true of date")
pub fn states_tod_to_gcrf(
    epochs: &[Epoch],
    x_tod: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_tod_to_gcrf, rotate_state, epochs, x_tod)
}

/// Transforms a Cartesian position in the true equator and equinox of date
/// (TOD) to the equivalent position in the ITRF.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian TOD position. Units: (*m*)
///
/// # Returns
/// - Cartesian ITRF position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_tod = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_itrf = position_tod_to_itrf(epc, x_tod);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn position_tod_to_itrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_tod_to_itrf(epc) * x
}

/// Transforms a Cartesian position in the ITRF to the equivalent position
/// in the true equator and equinox of date (TOD): the inverse of
/// [`position_tod_to_itrf`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian ITRF position. Units: (*m*)
///
/// # Returns
/// - Cartesian TOD position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_itrf = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_tod = position_itrf_to_tod(epc, x_itrf);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn position_itrf_to_tod(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_itrf_to_tod(epc) * x
}

/// Transforms a Cartesian state in the true equator and equinox of date
/// (TOD) to the equivalent state in the ITRF.
///
/// Accounts for the transport term from Earth's rotation, so the ITRF
/// velocity is not simply a rotated TOD velocity.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_tod`: Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_tod = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_itrf = state_tod_to_itrf(epc, x_tod);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn state_tod_to_itrf(epc: Epoch, x_tod: SVector6) -> SVector6 {
    apply_state_tod_to_itrf(&tod_itrf_context(epc), &x_tod)
}

/// Transforms a Cartesian state in the ITRF to the equivalent state in the
/// true equator and equinox of date (TOD): the inverse of
/// [`state_tod_to_itrf`].
///
/// Accounts for the transport term from Earth's rotation, so the TOD
/// velocity is not simply a rotated ITRF velocity.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_itrf`: Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_itrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_tod = state_itrf_to_tod(epc, x_itrf);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn state_itrf_to_tod(epc: Epoch, x_itrf: SVector6) -> SVector6 {
    apply_state_itrf_to_tod(&tod_itrf_context(epc), &x_itrf)
}

/// Computes the TOD-to-ITRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_tod_to_itrf`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming TOD -> ITRF, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_tod_to_itrf(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn rotations_tod_to_itrf(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_tod_to_itrf(*epc), epochs)
}

/// Computes the ITRF-to-TOD rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_itrf_to_tod`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming ITRF -> TOD, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_itrf_to_tod(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn rotations_itrf_to_tod(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_itrf_to_tod(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from TOD to ITRF.
///
/// Batch form of [`position_tod_to_itrf`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian TOD positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian ITRF positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![
///     vector3_from_array([R_EARTH, 0.0, 0.0]),
///     vector3_from_array([0.0, R_EARTH, 0.0]),
/// ];
///
/// // One epoch, many positions
/// let x_itrf = positions_tod_to_itrf(&[epc], &positions).unwrap();
/// assert_eq!(x_itrf.len(), 2);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn positions_tod_to_itrf(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_tod_to_itrf, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian positions from ITRF to TOD.
///
/// Batch form of [`position_itrf_to_tod`]. `epochs` and `x` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x`: Cartesian ITRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian TOD positions in input order. Units: (*m*)
/// - Error if `epochs` and `x` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let x = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // One position, many epochs
/// let x_tod = positions_itrf_to_tod(&epochs, &[x]).unwrap();
/// assert_eq!(x_tod.len(), 3);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn positions_itrf_to_tod(
    epochs: &[Epoch],
    x: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_itrf_to_tod, |r, x| r * x, epochs, x)
}

/// Transforms a batch of Cartesian states from TOD to ITRF.
///
/// Batch form of [`state_tod_to_itrf`]. `epochs` and `x_tod` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the transformation matrices once and applies them to every
/// state; per-element epochs compute them per state. Evaluation runs on the
/// global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_tod`: Cartesian TOD states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ITRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_tod` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let states = vec![
///     vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]),
///     vector6_from_array([0.0, R_EARTH + 500e3, 0.0, -v, 0.0, 0.0]),
/// ];
///
/// // One epoch, many states
/// let x_itrf = states_tod_to_itrf(&[epc], &states).unwrap();
/// assert_eq!(x_itrf.len(), 2);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn states_tod_to_itrf(
    epochs: &[Epoch],
    x_tod: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(tod_itrf_context, apply_state_tod_to_itrf, epochs, x_tod)
}

/// Transforms a batch of Cartesian states from ITRF to TOD.
///
/// Batch form of [`state_itrf_to_tod`]. `epochs` and `x_itrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the transformation matrices once and applies them to every
/// state; per-element epochs compute them per state. Evaluation runs on the
/// global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_itrf`: Cartesian ITRF states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TOD states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_itrf` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch
/// (the shared equinox context also evaluates the nutation corrections).
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let x_itrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]);
///
/// // One state, many epochs
/// let epochs = vec![epc, epc + 60.0];
/// let x_tod = states_itrf_to_tod(&epochs, &[x_itrf]).unwrap();
/// assert_eq!(x_tod.len(), 2);
/// ```
///
/// # References
/// - SOFA `c2teqx` note 2; SOFA cookbook Section 3.5 (polar motion) and
///   Appendix p. A4 (`R3(GAST)`, `W` rows)
pub fn states_itrf_to_tod(
    epochs: &[Epoch],
    x_itrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(tod_itrf_context, apply_state_itrf_to_tod, epochs, x_itrf)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use super::*;
    use crate::constants::AS2RAD;
    use crate::eop::{StaticEOPProvider, set_global_eop_provider};
    use crate::frames::gcrf_itrf::{
        bias_precession_nutation, earth_rotation, rotation_gcrf_to_itrf,
    };
    use crate::math::vector6_from_array;

    /// Cookbook Section 5 starting point (x_p, y_p, UT1-UTC, dX_2006, dY_2006).
    #[allow(non_snake_case)]
    fn set_cookbook_eop() {
        let pm_x = 0.0349282 * AS2RAD;
        let pm_y = 0.4833163 * AS2RAD;
        let ut1_utc = -0.072073685;
        let dX = 0.0001750 * AS2RAD;
        let dY = -0.0002259 * AS2RAD;
        set_global_eop_provider(StaticEOPProvider::from_values((
            pm_x, pm_y, ut1_utc, dX, dY, 0.0,
        )));
    }

    fn cookbook_epoch() -> Epoch {
        Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    fn assert_matrix_eq(a: &SMatrix3, b: &SMatrix3, tol: f64) {
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(a[(i, j)], b[(i, j)], epsilon = tol);
            }
        }
    }

    #[test]
    #[serial]
    fn test_bias_precession_matches_sofa_bp00() {
        set_global_eop_provider(StaticEOPProvider::from_zero());
        let epc = cookbook_epoch();
        let tt = epc.mjd_as_time_system(TimeSystem::TT);

        let mut rb = [[0.0; 3]; 3];
        let mut rp = [[0.0; 3]; 3];
        let mut rbp = [[0.0; 3]; 3];
        unsafe {
            rsofa::iauBp00(MJD_ZERO, tt, &mut rb[0], &mut rp[0], &mut rbp[0]);
        }
        assert_matrix_eq(&bias_precession(epc), &matrix3_from_array(&rbp), 1e-15);
        assert_matrix_eq(&rotation_gcrf_to_mod(epc), &matrix3_from_array(&rbp), 1e-15);
    }

    #[test]
    #[serial]
    fn test_bias_precession_needs_no_eop() {
        // Large dX/dY corrections have no effect on bias-precession: it does
        // not read the nutation-only Earth orientation corrections at all.
        set_global_eop_provider(StaticEOPProvider::from_values((
            0.0,
            0.0,
            0.0,
            1.0 * AS2RAD,
            1.0 * AS2RAD,
            0.0,
        )));
        let epc = cookbook_epoch();
        let tt = epc.mjd_as_time_system(TimeSystem::TT);

        let mut rb = [[0.0; 3]; 3];
        let mut rp = [[0.0; 3]; 3];
        let mut rbp = [[0.0; 3]; 3];
        unsafe {
            rsofa::iauBp00(MJD_ZERO, tt, &mut rb[0], &mut rp[0], &mut rbp[0]);
        }
        let oracle = matrix3_from_array(&rbp);
        let from_fn = bias_precession(epc);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(from_fn[(i, j)], oracle[(i, j)]);
            }
        }
    }

    #[test]
    #[serial]
    fn test_rotation_gcrf_to_tod_matches_sofa_pnm00b_with_zero_eop() {
        set_global_eop_provider(StaticEOPProvider::from_zero());
        let epc = cookbook_epoch();
        let tt = epc.mjd_as_time_system(TimeSystem::TT);

        let mut rbpn = [[0.0; 3]; 3];
        unsafe {
            rsofa::iauPnm00b(MJD_ZERO, tt, &mut rbpn[0]);
        }
        assert_matrix_eq(
            &rotation_gcrf_to_tod(epc),
            &matrix3_from_array(&rbpn),
            1e-15,
        );
        assert_matrix_eq(
            &(nutation(epc) * bias_precession(epc)),
            &matrix3_from_array(&rbpn),
            1e-15,
        );
    }

    #[test]
    #[serial]
    fn test_gcrf_to_itrf_via_tod_matches_sofa_c2t00b_with_zero_eop() {
        set_global_eop_provider(StaticEOPProvider::from_zero());
        let epc = cookbook_epoch();
        let tt = epc.mjd_as_time_system(TimeSystem::TT);
        let ut1 = epc.mjd_as_time_system(TimeSystem::UT1);

        // iauC2t00b omits the TIO locator s'; polar_motion includes it, so
        // the two agree to the size of s' (about 2e-11 rad at this epoch).
        let mut rc2t = [[0.0; 3]; 3];
        unsafe {
            rsofa::iauC2t00b(MJD_ZERO, tt, MJD_ZERO, ut1, 0.0, 0.0, &mut rc2t[0]);
        }
        let via_tod = rotation_tod_to_itrf(epc) * rotation_gcrf_to_tod(epc);
        assert_matrix_eq(&via_tod, &matrix3_from_array(&rc2t), 1e-9);
    }

    #[test]
    #[serial]
    fn test_gast_makes_equinox_chain_equal_cio_chain() {
        // R3(GAST) * NPB == R3(ERA) * C2ixys(X, Y, s) for the X, Y, s of NPB.
        set_cookbook_eop();
        let epc = cookbook_epoch();
        let tt = epc.mjd_as_time_system(TimeSystem::TT);
        let ut1 = epc.mjd_as_time_system(TimeSystem::UT1);

        let rnpb = rotation_gcrf_to_tod(epc);
        let mut rnpb_arr = [
            [rnpb[(0, 0)], rnpb[(0, 1)], rnpb[(0, 2)]],
            [rnpb[(1, 0)], rnpb[(1, 1)], rnpb[(1, 2)]],
            [rnpb[(2, 0)], rnpb[(2, 1)], rnpb[(2, 2)]],
        ];
        let mut x = 0.0;
        let mut y = 0.0;
        let mut rc2i = [[0.0; 3]; 3];
        let mut rc2ti = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        unsafe {
            rsofa::iauBpn2xy(&mut rnpb_arr[0], &mut x, &mut y);
            let s = rsofa::iauS00(MJD_ZERO, tt, x, y);
            rsofa::iauC2ixys(x, y, s, &mut rc2i[0]);
            let era = rsofa::iauEra00(MJD_ZERO, ut1);
            rsofa::iauCr(&mut rc2i[0], &mut rc2ti[0]);
            rsofa::iauRz(era, &mut rc2ti[0]);
        }
        let equinox = greenwich_apparent_sidereal_rotation(epc) * rnpb;
        assert_matrix_eq(&equinox, &matrix3_from_array(&rc2ti), 1e-14);
    }

    #[test]
    #[serial]
    fn test_rotation_gcrf_to_itrf_via_tod_matches_cookbook_5_4() {
        // SOFA cookbook Section 5.4 (IAU 2000A, equinox based) final matrix.
        // The 2000B series differs from 2000A at the milliarcsecond level.
        set_cookbook_eop();
        let epc = cookbook_epoch();

        let r = rotation_tod_to_itrf(epc) * rotation_gcrf_to_tod(epc);

        let tol = 1.0e-8;
        assert_abs_diff_eq!(r[(0, 0)], 0.973104317697618, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 1)], 0.230363826238780, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 2)], -0.000703163482352, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 0)], -0.230363800455689, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 1)], 0.973104570632883, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 2)], 0.000118545366826, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 0)], 0.000711560162864, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 1)], 0.000046626403835, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 2)], 0.999999745754024, epsilon = tol);
    }

    #[test]
    #[serial]
    fn test_equinox_chain_matches_cio_chain_with_corrections() {
        // Same 2000B basis, corrections applied in dPsi/dEps versus X/Y:
        // agreement at the microarcsecond level.
        set_cookbook_eop();
        let epc = cookbook_epoch();
        let via_tod = rotation_tod_to_itrf(epc) * rotation_gcrf_to_tod(epc);
        assert_matrix_eq(&via_tod, &rotation_gcrf_to_itrf(epc), 1e-10);
        let npb_equinox = rotation_gcrf_to_tod(epc);
        let c_cio = bias_precession_nutation(epc);
        // Both share the third row (the CIP direction in the GCRF).
        for j in 0..3 {
            assert_abs_diff_eq!(npb_equinox[(2, j)], c_cio[(2, j)], epsilon = 1e-10);
        }
        // TOD -> TIRS from the equinox chain equals CIRS -> TIRS from ERA.
        let tirs_equinox = greenwich_apparent_sidereal_rotation(epc) * npb_equinox;
        let tirs_cio = earth_rotation(epc) * c_cio;
        assert_matrix_eq(&tirs_equinox, &tirs_cio, 1e-10);
    }

    #[test]
    #[serial]
    fn test_rotation_inverses_are_transposes() {
        set_cookbook_eop();
        let epc = cookbook_epoch();
        assert_matrix_eq(
            &rotation_mod_to_gcrf(epc),
            &rotation_gcrf_to_mod(epc).transpose(),
            0.0,
        );
        assert_matrix_eq(
            &rotation_tod_to_mod(epc),
            &rotation_mod_to_tod(epc).transpose(),
            0.0,
        );
        assert_matrix_eq(
            &rotation_tod_to_gcrf(epc),
            &rotation_gcrf_to_tod(epc).transpose(),
            0.0,
        );
        assert_matrix_eq(
            &rotation_itrf_to_tod(epc),
            &rotation_tod_to_itrf(epc).transpose(),
            0.0,
        );
        assert_matrix_eq(&rotation_mod_to_tod(epc), &nutation(epc), 0.0);
        let identity = rotation_gcrf_to_tod(epc) * rotation_tod_to_gcrf(epc);
        assert_matrix_eq(&identity, &SMatrix3::identity(), 1e-15);
    }

    #[test]
    #[serial]
    fn test_nutation_applies_eop_corrections() {
        // A large synthetic dX shifts the CIP by dX along the GCRF x axis.
        let epc = cookbook_epoch();
        set_global_eop_provider(StaticEOPProvider::from_zero());
        let uncorrected = rotation_gcrf_to_tod(epc);
        let dx = 1.0 * AS2RAD;
        set_global_eop_provider(StaticEOPProvider::from_values((
            0.0, 0.0, 0.0, dx, 0.0, 0.0,
        )));
        let corrected = rotation_gcrf_to_tod(epc);
        // Third row is the CIP unit vector in the GCRF (pn00b note 9).
        assert_abs_diff_eq!(
            corrected[(2, 0)] - uncorrected[(2, 0)],
            dx,
            epsilon = 1e-3 * dx
        );
        assert_abs_diff_eq!(
            corrected[(2, 1)] - uncorrected[(2, 1)],
            0.0,
            epsilon = 1e-3 * dx
        );
        // MOD does not depend on the corrections.
        assert_matrix_eq(&rotation_gcrf_to_mod(epc), &bias_precession(epc), 0.0);
    }

    fn sample_state() -> SVector6 {
        vector6_from_array([constants::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
    }

    #[test]
    #[serial]
    fn test_state_transforms_rotate_position_and_velocity() {
        set_cookbook_eop();
        let epc = cookbook_epoch();
        let x = sample_state();
        for (r, x_out) in [
            (rotation_gcrf_to_mod(epc), state_gcrf_to_mod(epc, x)),
            (rotation_mod_to_tod(epc), state_mod_to_tod(epc, x)),
            (rotation_gcrf_to_tod(epc), state_gcrf_to_tod(epc, x)),
            (rotation_mod_to_gcrf(epc), state_mod_to_gcrf(epc, x)),
            (rotation_tod_to_mod(epc), state_tod_to_mod(epc, x)),
            (rotation_tod_to_gcrf(epc), state_tod_to_gcrf(epc, x)),
        ] {
            let r_exp: Vector3<f64> = r * x.fixed_rows::<3>(0);
            let v_exp: Vector3<f64> = r * x.fixed_rows::<3>(3);
            for i in 0..3 {
                assert_abs_diff_eq!(x_out[i], r_exp[i], epsilon = 1e-6);
                assert_abs_diff_eq!(x_out[i + 3], v_exp[i], epsilon = 1e-9);
            }
        }
    }

    #[test]
    #[allow(clippy::type_complexity)]
    #[serial]
    fn test_position_round_trips() {
        set_cookbook_eop();
        let epc = cookbook_epoch();
        let x = Vector3::new(constants::R_EARTH + 500e3, 1.0e6, -2.0e6);
        let pairs: [(
            fn(Epoch, Vector3<f64>) -> Vector3<f64>,
            fn(Epoch, Vector3<f64>) -> Vector3<f64>,
        ); 4] = [
            (position_gcrf_to_mod, position_mod_to_gcrf),
            (position_mod_to_tod, position_tod_to_mod),
            (position_gcrf_to_tod, position_tod_to_gcrf),
            (position_tod_to_itrf, position_itrf_to_tod),
        ];
        for (fwd, inv) in pairs {
            let back = inv(epc, fwd(epc, x));
            for i in 0..3 {
                assert_abs_diff_eq!(back[i], x[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[serial]
    fn test_state_tod_to_itrf_round_trip_and_transport_term() {
        set_cookbook_eop();
        let epc = cookbook_epoch();
        let x_tod = sample_state();
        let x_itrf = state_tod_to_itrf(epc, x_tod);
        let back = state_itrf_to_tod(epc, x_itrf);
        for i in 0..3 {
            assert_abs_diff_eq!(back[i], x_tod[i], epsilon = 1e-6);
            assert_abs_diff_eq!(back[i + 3], x_tod[i + 3], epsilon = 1e-9);
        }
        // Velocity in ITRF must equal the finite difference of ITRF positions.
        let dt = 0.5;
        let p_minus = position_tod_to_itrf(
            epc - dt,
            Vector3::from(x_tod.fixed_rows::<3>(0)) - dt * Vector3::from(x_tod.fixed_rows::<3>(3)),
        );
        let p_plus = position_tod_to_itrf(
            epc + dt,
            Vector3::from(x_tod.fixed_rows::<3>(0)) + dt * Vector3::from(x_tod.fixed_rows::<3>(3)),
        );
        let v_fd = (p_plus - p_minus) / (2.0 * dt);
        for i in 0..3 {
            assert_abs_diff_eq!(x_itrf[i + 3], v_fd[i], epsilon = 1e-3);
        }
        // Matches the CIO chain's transport handling.
        let x_gcrf = state_tod_to_gcrf(epc, x_tod);
        let x_itrf_cio = crate::frames::gcrf_itrf::state_gcrf_to_itrf(epc, x_gcrf);
        for i in 0..3 {
            assert_abs_diff_eq!(x_itrf[i], x_itrf_cio[i], epsilon = 1e-3);
            assert_abs_diff_eq!(x_itrf[i + 3], x_itrf_cio[i + 3], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_batch_forms_match_scalar_forms() {
        set_cookbook_eop();
        let epochs: Vec<Epoch> = (0..3)
            .map(|i| cookbook_epoch() + 3600.0 * i as f64)
            .collect();
        let positions: Vec<Vector3<f64>> = (0..3)
            .map(|i| Vector3::new(constants::R_EARTH + 500e3 + 1e3 * i as f64, 1.0e6, -2.0e6))
            .collect();
        let states: Vec<SVector6> = (0..3)
            .map(|i| sample_state() + SVector6::repeat(10.0 * i as f64))
            .collect();

        let rots = rotations_gcrf_to_tod(&epochs);
        let rots_tod_gcrf = rotations_tod_to_gcrf(&epochs);
        let rots_gcrf_mod = rotations_gcrf_to_mod(&epochs);
        let rots_mod_gcrf = rotations_mod_to_gcrf(&epochs);
        let rots_mod_tod = rotations_mod_to_tod(&epochs);
        let rots_tod_mod = rotations_tod_to_mod(&epochs);
        let rots_tod_itrf = rotations_tod_to_itrf(&epochs);
        let rots_itrf_tod = rotations_itrf_to_tod(&epochs);
        let poss = positions_tod_to_itrf(&epochs, &positions).unwrap();
        let poss_itrf_tod = positions_itrf_to_tod(&epochs, &positions).unwrap();
        let poss_gcrf_mod_fwd = positions_gcrf_to_mod(&epochs, &positions).unwrap();
        let poss_gcrf_mod = positions_mod_to_gcrf(&epochs, &positions).unwrap();
        let poss_mod_tod = positions_mod_to_tod(&epochs, &positions).unwrap();
        let poss_tod_mod = positions_tod_to_mod(&epochs, &positions).unwrap();
        let poss_gcrf_tod = positions_gcrf_to_tod(&epochs, &positions).unwrap();
        let poss_tod_gcrf = positions_tod_to_gcrf(&epochs, &positions).unwrap();
        let sts = states_gcrf_to_tod(&epochs, &states).unwrap();
        let sts_inv = states_itrf_to_tod(&epochs, &states).unwrap();
        let sts_tod_itrf = states_tod_to_itrf(&epochs, &states).unwrap();
        let sts_gcrf_mod = states_gcrf_to_mod(&epochs, &states).unwrap();
        let sts_mod_gcrf = states_mod_to_gcrf(&epochs, &states).unwrap();
        let sts_tod_mod = states_tod_to_mod(&epochs, &states).unwrap();
        let sts_tod_gcrf = states_tod_to_gcrf(&epochs, &states).unwrap();
        for i in 0..3 {
            assert_matrix_eq(&rots[i], &rotation_gcrf_to_tod(epochs[i]), 0.0);
            assert_matrix_eq(&rots_tod_gcrf[i], &rotation_tod_to_gcrf(epochs[i]), 0.0);
            assert_matrix_eq(&rots_gcrf_mod[i], &rotation_gcrf_to_mod(epochs[i]), 0.0);
            assert_matrix_eq(&rots_mod_gcrf[i], &rotation_mod_to_gcrf(epochs[i]), 0.0);
            assert_matrix_eq(&rots_mod_tod[i], &rotation_mod_to_tod(epochs[i]), 0.0);
            assert_matrix_eq(&rots_tod_mod[i], &rotation_tod_to_mod(epochs[i]), 0.0);
            assert_matrix_eq(&rots_tod_itrf[i], &rotation_tod_to_itrf(epochs[i]), 0.0);
            assert_matrix_eq(&rots_itrf_tod[i], &rotation_itrf_to_tod(epochs[i]), 0.0);
            assert_eq!(poss[i], position_tod_to_itrf(epochs[i], positions[i]));
            assert_eq!(
                poss_itrf_tod[i],
                position_itrf_to_tod(epochs[i], positions[i])
            );
            assert_eq!(
                poss_gcrf_mod_fwd[i],
                position_gcrf_to_mod(epochs[i], positions[i])
            );
            assert_eq!(
                poss_gcrf_mod[i],
                position_mod_to_gcrf(epochs[i], positions[i])
            );
            assert_eq!(
                poss_mod_tod[i],
                position_mod_to_tod(epochs[i], positions[i])
            );
            assert_eq!(
                poss_tod_mod[i],
                position_tod_to_mod(epochs[i], positions[i])
            );
            assert_eq!(
                poss_gcrf_tod[i],
                position_gcrf_to_tod(epochs[i], positions[i])
            );
            assert_eq!(
                poss_tod_gcrf[i],
                position_tod_to_gcrf(epochs[i], positions[i])
            );
            assert_eq!(sts[i], state_gcrf_to_tod(epochs[i], states[i]));
            assert_eq!(sts_inv[i], state_itrf_to_tod(epochs[i], states[i]));
            assert_eq!(sts_tod_itrf[i], state_tod_to_itrf(epochs[i], states[i]));
            assert_eq!(sts_gcrf_mod[i], state_gcrf_to_mod(epochs[i], states[i]));
            assert_eq!(sts_mod_gcrf[i], state_mod_to_gcrf(epochs[i], states[i]));
            assert_eq!(sts_tod_mod[i], state_tod_to_mod(epochs[i], states[i]));
            assert_eq!(sts_tod_gcrf[i], state_tod_to_gcrf(epochs[i], states[i]));
        }
        // Broadcast: one epoch, many states.
        let one = states_mod_to_tod(&epochs[..1], &states).unwrap();
        for i in 0..3 {
            assert_eq!(one[i], state_mod_to_tod(epochs[0], states[i]));
        }
        // Mismatched lengths error.
        assert!(positions_gcrf_to_mod(&epochs[..2], &positions).is_err());
    }
}
