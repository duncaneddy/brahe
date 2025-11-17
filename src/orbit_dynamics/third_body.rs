/*!
Module for the third body perturbations. Also provides low-precession models for the Sun and Moon
ephemerides.
 */

use nalgebra::Vector3;

use crate::ephemerides::{moon_position, sun_position};
use crate::orbit_dynamics::gravity::acceleration_point_mass_gravity;
use crate::time::Epoch;
use crate::{GM_MOON, GM_SUN};

/// Calculate the acceleration due to the Sun on an object at a given epoch.
/// The calculation is performed using the point-mass gravity model and the
/// low-precision analytical ephemerides for the Sun position implemented in
/// the `ephemerides` module.
///
/// Should a more accurate calculation be required, you can utilize the
/// point-mass gravity model and a higher-precision ephemerides for the Sun.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Sun's position
/// * `r_object` - Position of the object in the GCRF frame. Units: [m]
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
/// use brahe::third_body::acceleration_third_body_sun;
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = acceleration_third_body_sun(epc, r_object);
/// ```
pub fn acceleration_third_body_sun(epc: Epoch, r_object: Vector3<f64>) -> Vector3<f64> {
    acceleration_point_mass_gravity(r_object, sun_position(epc), GM_SUN)
}

/// Calculate the acceleration due to the Moon on an object at a given epoch.
/// The calculation is performed using the point-mass gravity model and the
/// low-precision analytical ephemerides for the Moon position implemented in
/// the `ephemerides` module.
///
/// Should a more accurate calculation be required, you can utilize the
/// point-mass gravity model and a higher-precision ephemerides for the Moon.
///
/// # Arguments
///
/// - `epc` - Epoch at which to calculate the Moon's position
/// - `r_object` - Position of the object in the GCRF frame. Units: [m]
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
/// use brahe::third_body::acceleration_third_body_moon;
/// use brahe::constants::R_EARTH;
/// use nalgebra::Vector3;
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_date(2024, 2, 25, brahe::TimeSystem::UTC);
/// let r_object = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
///
/// let a = acceleration_third_body_moon(epc, r_object);
/// ```
pub fn acceleration_third_body_moon(epc: Epoch, r_object: Vector3<f64>) -> Vector3<f64> {
    acceleration_point_mass_gravity(r_object, moon_position(epc), GM_MOON)
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
    fn test_acceleration_third_body_sun(
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

        let a = acceleration_third_body_sun(epc, r_object);

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
    fn test_acceleration_third_body_moon(
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

        let a = acceleration_third_body_moon(epc, r_object);

        assert_abs_diff_eq!(a[0], ax, epsilon = 1e-9);
        assert_abs_diff_eq!(a[1], ay, epsilon = 1e-9);
        assert_abs_diff_eq!(a[2], az, epsilon = 1e-9);
    }
}
