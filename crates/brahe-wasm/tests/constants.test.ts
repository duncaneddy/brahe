import { describe, expect, test } from 'vitest';
import * as brahe from '../js/index.js';

describe('math constants', () => {
  test('DEG2RAD', () => {
    expect(brahe.DEG2RAD).toBe(Math.PI / 180);
  });

  test('RAD2DEG', () => {
    expect(brahe.RAD2DEG).toBe(180 / Math.PI);
  });

  test('AS2RAD', () => {
    expect(brahe.AS2RAD).toBe(brahe.DEG2RAD / 3600);
  });

  test('RAD2AS', () => {
    expect(brahe.RAD2AS).toBe(brahe.RAD2DEG * 3600);
  });
});

describe('time constants', () => {
  test('MJD_ZERO', () => {
    expect(brahe.MJD_ZERO).toBe(2400000.5);
  });

  test('MJD_J2000', () => {
    expect(brahe.MJD_J2000).toBe(51544.5);
  });

  test('JD_J2000', () => {
    expect(brahe.JD_J2000).toBe(2451545.0);
  });

  test('GPS_TAI', () => {
    expect(brahe.GPS_TAI).toBe(-19.0);
  });

  test('TAI_GPS', () => {
    expect(brahe.TAI_GPS).toBe(-brahe.GPS_TAI);
  });

  test('TT_TAI', () => {
    expect(brahe.TT_TAI).toBe(32.184);
  });

  test('TAI_TT', () => {
    expect(brahe.TAI_TT).toBe(-brahe.TT_TAI);
  });

  test('GPS_TT', () => {
    expect(brahe.GPS_TT).toBe(brahe.GPS_TAI + brahe.TAI_TT);
  });

  test('TT_GPS', () => {
    expect(brahe.TT_GPS).toBe(-brahe.GPS_TT);
  });

  test('GPS_ZERO', () => {
    expect(brahe.GPS_ZERO).toBe(44244.0);
  });

  test('BDT_TAI', () => {
    expect(brahe.BDT_TAI).toBe(-33.0);
  });

  test('TAI_BDT', () => {
    expect(brahe.TAI_BDT).toBe(-brahe.BDT_TAI);
  });

  test('GST_TAI', () => {
    expect(brahe.GST_TAI).toBe(-19.0);
  });

  test('TAI_GST', () => {
    expect(brahe.TAI_GST).toBe(-brahe.GST_TAI);
  });

  test('BDT_ZERO', () => {
    expect(brahe.BDT_ZERO).toBe(53736.0);
  });

  test('GST_ZERO', () => {
    expect(brahe.GST_ZERO).toBe(51412.0);
  });

  test('UNIX_EPOCH_JD', () => {
    expect(brahe.UNIX_EPOCH_JD).toBe(2440587.5);
  });

  test('UNIX_EPOCH_MJD', () => {
    expect(brahe.UNIX_EPOCH_MJD).toBe(40587.0);
  });
});

describe('physical constants', () => {
  test('C_LIGHT', () => {
    expect(brahe.C_LIGHT).toBe(299792458.0);
  });

  test('AU', () => {
    expect(brahe.AU).toBe(1.49597870700e11);
  });
});

describe('Earth constants', () => {
  test('R_EARTH', () => {
    expect(brahe.R_EARTH).toBe(6.378136300e6);
  });

  test('WGS84_A', () => {
    expect(brahe.WGS84_A).toBe(6378137.0);
  });

  test('WGS84_F', () => {
    expect(brahe.WGS84_F).toBe(1.0 / 298.257223563);
  });

  test('GM_EARTH', () => {
    expect(brahe.GM_EARTH).toBe(3.986004415e14);
  });

  test('ECC_EARTH', () => {
    expect(brahe.ECC_EARTH).toBe(8.1819190842622e-2);
  });

  test('OMEGA_EARTH', () => {
    expect(brahe.OMEGA_EARTH).toBe(7.292115146706979e-5);
  });
});

// EGM2008 fully-normalized Stokes coefficients C_n,0 — used to verify the
// J_n constants were derived correctly via J_n = -C_n,0 * sqrt(2n + 1).
// Source: data/gravity_models/EGM2008_360.gfc (mirrors tests/test_constants.py).
const EGM2008_C_2_0 = -0.484165143790815e-03;
const EGM2008_C_3_0 = 0.957161207093473e-06;
const EGM2008_C_4_0 = 0.539965866638991e-06;
const EGM2008_C_5_0 = 0.686702913736681e-07;
const EGM2008_C_6_0 = -0.149953927978527e-06;

function unnormalizeZonal(c_n0: number, n: number): number {
  return -c_n0 * Math.sqrt(2 * n + 1);
}

describe('Earth zonal harmonics (derived from EGM2008)', () => {
  // toBeCloseTo(value, digits) asserts that |actual - expected| < 0.5 * 10^-digits.
  // The Python test uses abs=1e-18 / 1e-21 / 1e-22; we match those tolerances.

  test('J2_EARTH', () => {
    expect(brahe.J2_EARTH).toBeCloseTo(unnormalizeZonal(EGM2008_C_2_0, 2), 18);
  });

  test('J3_EARTH', () => {
    expect(brahe.J3_EARTH).toBeCloseTo(unnormalizeZonal(EGM2008_C_3_0, 3), 21);
  });

  test('J4_EARTH', () => {
    expect(brahe.J4_EARTH).toBeCloseTo(unnormalizeZonal(EGM2008_C_4_0, 4), 21);
  });

  test('J5_EARTH', () => {
    expect(brahe.J5_EARTH).toBeCloseTo(unnormalizeZonal(EGM2008_C_5_0, 5), 22);
  });

  test('J6_EARTH', () => {
    expect(brahe.J6_EARTH).toBeCloseTo(unnormalizeZonal(EGM2008_C_6_0, 6), 21);
  });
});

describe('solar constants', () => {
  test('GM_SUN', () => {
    expect(brahe.GM_SUN).toBe(132712440041.939400 * 1e9);
  });

  test('R_SUN', () => {
    expect(brahe.R_SUN).toBe(6.957 * 1e8);
  });

  test('P_SUN', () => {
    expect(brahe.P_SUN).toBe(4.560e-6);
  });
});

describe('lunar constants', () => {
  test('R_MOON', () => {
    expect(brahe.R_MOON).toBe(1738 * 1e3);
  });

  test('GM_MOON', () => {
    expect(brahe.GM_MOON).toBe(4902.800066 * 1e9);
  });
});

describe('planetary constants', () => {
  test('GM_MERCURY', () => {
    expect(brahe.GM_MERCURY).toBe(22031.780000 * 1e9);
  });

  test('GM_VENUS', () => {
    expect(brahe.GM_VENUS).toBe(324858.592000 * 1e9);
  });

  test('GM_MARS', () => {
    expect(brahe.GM_MARS).toBe(42828.37521 * 1e9);
  });

  test('GM_JUPITER', () => {
    expect(brahe.GM_JUPITER).toBe(126712764.8 * 1e9);
  });

  test('GM_SATURN', () => {
    expect(brahe.GM_SATURN).toBe(37940585.2 * 1e9);
  });

  test('GM_URANUS', () => {
    expect(brahe.GM_URANUS).toBe(5794548.6 * 1e9);
  });

  test('GM_NEPTUNE', () => {
    expect(brahe.GM_NEPTUNE).toBe(6836527.100580 * 1e9);
  });

  test('GM_PLUTO', () => {
    expect(brahe.GM_PLUTO).toBe(977.000000 * 1e9);
  });
});
