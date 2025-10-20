# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Example demonstrating calculation of orbital properties.

This example shows how to calculate key orbital properties including apoapsis distance,
periapsis distance, and velocities at apoapsis and periapsis for different orbit types.
"""

import brahe as bh
import pytest

if __name__ == "__main__":
    # Example 1: Low Earth Orbit (LEO)
    a_leo = bh.R_EARTH + 500e3  # 500 km altitude
    e_leo = 0.01  # Nearly circular

    # Calculate distances
    r_apo_leo = bh.apoapsis_distance(a_leo, e_leo)
    r_peri_leo = bh.periapsis_distance(a_leo, e_leo)

    # Calculate velocities
    v_apo_leo = bh.apoapsis_velocity(a_leo, e_leo, bh.GM_EARTH)
    v_peri_leo = bh.periapsis_velocity(a_leo, e_leo, bh.GM_EARTH)

    # Verify relationship: apoapsis distance > periapsis distance
    assert r_apo_leo > r_peri_leo

    # Verify relationship: periapsis velocity > apoapsis velocity
    assert v_peri_leo > v_apo_leo

    # For nearly circular orbit, distances should be nearly equal
    assert r_apo_leo == pytest.approx(r_peri_leo, rel=2.5e-2)

    # Example 2: Geostationary Transfer Orbit (GTO)
    a_gto = (bh.R_EARTH + 250e3 + bh.R_EARTH + 35786e3) / 2.0  # Average of LEO and GEO
    e_gto = (bh.R_EARTH + 35786e3 - bh.R_EARTH - 250e3) / (
        bh.R_EARTH + 35786e3 + bh.R_EARTH + 250e3
    )

    # Calculate distances
    r_apo_gto = bh.apoapsis_distance(a_gto, e_gto)
    r_peri_gto = bh.periapsis_distance(a_gto, e_gto)

    # Calculate velocities
    v_apo_gto = bh.apoapsis_velocity(a_gto, e_gto, bh.GM_EARTH)
    v_peri_gto = bh.periapsis_velocity(a_gto, e_gto, bh.GM_EARTH)

    # Verify apoapsis is at GEO altitude
    assert r_apo_gto == pytest.approx(bh.R_EARTH + 35786e3, rel=1e-3)

    # Verify periapsis is at LEO altitude
    assert r_peri_gto == pytest.approx(bh.R_EARTH + 250e3, rel=1e-3)

    # Verify energy conservation (specific mechanical energy should be constant)
    # E = v²/2 - GM/r
    E_peri = v_peri_gto**2 / 2.0 - bh.GM_EARTH / r_peri_gto
    E_apo = v_apo_gto**2 / 2.0 - bh.GM_EARTH / r_apo_gto
    assert E_peri == pytest.approx(E_apo, rel=1e-10)

    # Example 3: Highly Elliptical Orbit (HEO)
    a_heo = bh.R_EARTH + 30000e3  # Very high semi-major axis
    e_heo = 0.7  # High eccentricity

    # Calculate distances
    r_apo_heo = bh.apoapsis_distance(a_heo, e_heo)
    r_peri_heo = bh.periapsis_distance(a_heo, e_heo)

    # Calculate velocities
    v_apo_heo = bh.apoapsis_velocity(a_heo, e_heo, bh.GM_EARTH)
    v_peri_heo = bh.periapsis_velocity(a_heo, e_heo, bh.GM_EARTH)

    # Verify formulas: r_apo = a(1+e), r_peri = a(1-e)
    assert r_apo_heo == pytest.approx(a_heo * (1.0 + e_heo), rel=1e-10)
    assert r_peri_heo == pytest.approx(a_heo * (1.0 - e_heo), rel=1e-10)

    # Verify angular momentum conservation: r_peri * v_peri = r_apo * v_apo
    h_peri = r_peri_heo * v_peri_heo
    h_apo = r_apo_heo * v_apo_heo
    assert h_peri == pytest.approx(h_apo, rel=1e-10)

    print("✓ Orbital properties calculations validated successfully!")
