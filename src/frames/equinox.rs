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
use crate::constants::MJD_ZERO;
use crate::eop;
use crate::frames::gcrf_itrf::polar_motion;
use crate::math::{SMatrix3, matrix3_from_array};
use crate::time::{Epoch, TimeSystem};

/// Equinox-based precession-nutation products for one epoch, computed once
/// and shared by the pairwise and batch transformations.
pub(crate) struct EquinoxContext {
    /// GCRF -> MOD bias-precession matrix (`rp * rb`).
    pub(crate) rbp: SMatrix3,
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
            rbp: matrix3_from_array(&rbp),
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
/// let rbp = bias_precession(epc);
/// ```
///
/// # References
/// - SOFA `pn00b` notes 4-6 (`rbp = rp * rb`); SOFA cookbook Section 3.1
///   (classical precession) and Appendix p. A4 (`B`, `P` rows)
pub fn bias_precession(epc: Epoch) -> SMatrix3 {
    EquinoxContext::new(epc).rbp
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::constants::AS2RAD;
    use crate::eop::{StaticEOPProvider, set_global_eop_provider};
    use crate::frames::gcrf_itrf::{
        bias_precession_nutation, earth_rotation, rotation_gcrf_to_itrf,
    };

    /// Cookbook Section 5 starting point (x_p, y_p, UT1-UTC, dX_2006, dY_2006).
    #[allow(non_snake_case)]
    fn set_cookbook_eop() {
        let pm_x = 0.0349282 * AS2RAD;
        let pm_y = 0.4833163 * AS2RAD;
        let ut1_utc = -0.072073685;
        let dX = 0.0001750 * AS2RAD * 1.0e-3;
        let dY = -0.0002259 * AS2RAD * 1.0e-3;
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
        // the two agree to the size of s' (about 1e-10 rad at this epoch).
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
}
