//! WASM bindings for `brahe::constants`.
//!
//! All exports are prefixed with `__` in their JS name to mark them as
//! private/internal — only the TypeScript wrapper in `js/constants.ts`
//! is expected to call them directly.

use wasm_bindgen::prelude::*;

use brahe::constants as c;

// ────────────────────────────────────────────────────────────────
// Math constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __DEG2RAD)]
pub fn deg2rad() -> f64 {
    c::DEG2RAD
}

#[wasm_bindgen(js_name = __RAD2DEG)]
pub fn rad2deg() -> f64 {
    c::RAD2DEG
}

#[wasm_bindgen(js_name = __AS2RAD)]
pub fn as2rad() -> f64 {
    c::AS2RAD
}

#[wasm_bindgen(js_name = __RAD2AS)]
pub fn rad2as() -> f64 {
    c::RAD2AS
}

// ────────────────────────────────────────────────────────────────
// Time constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __MJD_ZERO)]
pub fn mjd_zero() -> f64 { c::MJD_ZERO }

#[wasm_bindgen(js_name = __MJD_J2000)]
pub fn mjd_j2000() -> f64 { c::MJD_J2000 }

#[wasm_bindgen(js_name = __JD_J2000)]
pub fn jd_j2000() -> f64 { c::JD_J2000 }

#[wasm_bindgen(js_name = __GPS_TAI)]
pub fn gps_tai() -> f64 { c::GPS_TAI }

#[wasm_bindgen(js_name = __TAI_GPS)]
pub fn tai_gps() -> f64 { c::TAI_GPS }

#[wasm_bindgen(js_name = __TT_TAI)]
pub fn tt_tai() -> f64 { c::TT_TAI }

#[wasm_bindgen(js_name = __TAI_TT)]
pub fn tai_tt() -> f64 { c::TAI_TT }

#[wasm_bindgen(js_name = __GPS_TT)]
pub fn gps_tt() -> f64 { c::GPS_TT }

#[wasm_bindgen(js_name = __TT_GPS)]
pub fn tt_gps() -> f64 { c::TT_GPS }

#[wasm_bindgen(js_name = __GPS_ZERO)]
pub fn gps_zero() -> f64 { c::GPS_ZERO }

#[wasm_bindgen(js_name = __BDT_TAI)]
pub fn bdt_tai() -> f64 { c::BDT_TAI }

#[wasm_bindgen(js_name = __TAI_BDT)]
pub fn tai_bdt() -> f64 { c::TAI_BDT }

#[wasm_bindgen(js_name = __GST_TAI)]
pub fn gst_tai() -> f64 { c::GST_TAI }

#[wasm_bindgen(js_name = __TAI_GST)]
pub fn tai_gst() -> f64 { c::TAI_GST }

#[wasm_bindgen(js_name = __BDT_ZERO)]
pub fn bdt_zero() -> f64 { c::BDT_ZERO }

#[wasm_bindgen(js_name = __GST_ZERO)]
pub fn gst_zero() -> f64 { c::GST_ZERO }

#[wasm_bindgen(js_name = __UNIX_EPOCH_JD)]
pub fn unix_epoch_jd() -> f64 { c::UNIX_EPOCH_JD }

#[wasm_bindgen(js_name = __UNIX_EPOCH_MJD)]
pub fn unix_epoch_mjd() -> f64 { c::UNIX_EPOCH_MJD }

// ────────────────────────────────────────────────────────────────
// Physical constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __C_LIGHT)]
pub fn c_light() -> f64 { c::C_LIGHT }

#[wasm_bindgen(js_name = __AU)]
pub fn au() -> f64 { c::AU }

// ────────────────────────────────────────────────────────────────
// Earth constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __R_EARTH)]
pub fn r_earth() -> f64 { c::R_EARTH }

#[wasm_bindgen(js_name = __WGS84_A)]
pub fn wgs84_a() -> f64 { c::WGS84_A }

#[wasm_bindgen(js_name = __WGS84_F)]
pub fn wgs84_f() -> f64 { c::WGS84_F }

#[wasm_bindgen(js_name = __GM_EARTH)]
pub fn gm_earth() -> f64 { c::GM_EARTH }

#[wasm_bindgen(js_name = __ECC_EARTH)]
pub fn ecc_earth() -> f64 { c::ECC_EARTH }

#[wasm_bindgen(js_name = __J2_EARTH)]
pub fn j2_earth() -> f64 { c::J2_EARTH }

#[wasm_bindgen(js_name = __J3_EARTH)]
pub fn j3_earth() -> f64 { c::J3_EARTH }

#[wasm_bindgen(js_name = __J4_EARTH)]
pub fn j4_earth() -> f64 { c::J4_EARTH }

#[wasm_bindgen(js_name = __J5_EARTH)]
pub fn j5_earth() -> f64 { c::J5_EARTH }

#[wasm_bindgen(js_name = __J6_EARTH)]
pub fn j6_earth() -> f64 { c::J6_EARTH }

#[wasm_bindgen(js_name = __OMEGA_EARTH)]
pub fn omega_earth() -> f64 { c::OMEGA_EARTH }

// ────────────────────────────────────────────────────────────────
// Solar constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __GM_SUN)]
pub fn gm_sun() -> f64 { c::GM_SUN }

#[wasm_bindgen(js_name = __R_SUN)]
pub fn r_sun() -> f64 { c::R_SUN }

#[wasm_bindgen(js_name = __P_SUN)]
pub fn p_sun() -> f64 { c::P_SUN }

// ────────────────────────────────────────────────────────────────
// Lunar constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __R_MOON)]
pub fn r_moon() -> f64 { c::R_MOON }

#[wasm_bindgen(js_name = __GM_MOON)]
pub fn gm_moon() -> f64 { c::GM_MOON }

// ────────────────────────────────────────────────────────────────
// Planetary constants
// ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = __GM_MERCURY)]
pub fn gm_mercury() -> f64 { c::GM_MERCURY }

#[wasm_bindgen(js_name = __GM_VENUS)]
pub fn gm_venus() -> f64 { c::GM_VENUS }

#[wasm_bindgen(js_name = __GM_MARS)]
pub fn gm_mars() -> f64 { c::GM_MARS }

#[wasm_bindgen(js_name = __GM_JUPITER)]
pub fn gm_jupiter() -> f64 { c::GM_JUPITER }

#[wasm_bindgen(js_name = __GM_SATURN)]
pub fn gm_saturn() -> f64 { c::GM_SATURN }

#[wasm_bindgen(js_name = __GM_URANUS)]
pub fn gm_uranus() -> f64 { c::GM_URANUS }

#[wasm_bindgen(js_name = __GM_NEPTUNE)]
pub fn gm_neptune() -> f64 { c::GM_NEPTUNE }

#[wasm_bindgen(js_name = __GM_PLUTO)]
pub fn gm_pluto() -> f64 { c::GM_PLUTO }
