#!/usr/bin/env python3
# /// script
# dependencies = ["mpmath"]
# ///
"""Generate high-precision spherical-harmonic gravity reference accelerations.

Produces the mpmath (40-digit) reference values pinned in
`test_clenshaw_high_precision_reference` in `src/orbit_dynamics/gravity.rs`.
The evaluation is independent of the library's Clenshaw and Cunningham
kernels: it sums the truncated EGM2008_120 geopotential directly using a
forward-column recurrence for the fully-normalized associated Legendre
functions and the analytic theta-derivative identity, entirely in mpmath
arbitrary-precision arithmetic.

Run from the repository root:

    uv run scripts/generate_clenshaw_gravity_reference.py

The output lists, for each (position, n_max, m_max) case, the body-fixed
acceleration components to 20 significant digits. Copy these into the test
if the case list changes (the raw model file never changes, so existing
values should always reproduce bit-for-bit at f64 precision).
"""

from pathlib import Path

from mpmath import mp, mpf, sqrt, cos, sin, atan2

mp.dps = 40

GFC_PATH = (
    Path(__file__).resolve().parent.parent / "data/gravity_models/EGM2008_120.gfc"
)

# (position [m], n_max, m_max) — must match the cases in
# `test_clenshaw_high_precision_reference`.
MID_LAT_LEO = ("6.5e6", "1.2e6", "3.1e6")
EQUATORIAL_LEO = ("6878136.3", "0.0", "0.0")  # R_EARTH + 500e3 with R_EARTH = 6378136.3
CASES = [
    (MID_LAT_LEO, 80, 80),
    (MID_LAT_LEO, 120, 120),
    (EQUATORIAL_LEO, 60, 60),
    (EQUATORIAL_LEO, 90, 90),
    (EQUATORIAL_LEO, 120, 60),
    (EQUATORIAL_LEO, 120, 120),
]


def parse_gfc(path, n_max):
    """Parse a fully-normalized ICGEM .gfc file up to degree n_max."""
    gm = radius = None
    c = {}
    s = {}
    in_head = True
    for line in path.read_text().splitlines():
        parts = line.split()
        if in_head:
            if parts and parts[0] == "earth_gravity_constant":
                gm = mpf(parts[1].replace("d", "e").replace("D", "e"))
            elif parts and parts[0] == "radius":
                radius = mpf(parts[1].replace("d", "e").replace("D", "e"))
            elif parts and parts[0] == "end_of_head":
                in_head = False
            continue
        if not parts or parts[0] != "gfc":
            continue
        n, m = int(parts[1]), int(parts[2])
        if n > n_max:
            continue
        c[(n, m)] = mpf(parts[3].replace("d", "e").replace("D", "e"))
        s[(n, m)] = mpf(parts[4].replace("d", "e").replace("D", "e"))
    return gm, radius, c, s


def legendre_column(m, n_max, t, u, pbar_prev_sectoral):
    """Fully-normalized ALF column P̄(n, m) for n = m..n_max via the standard
    forward-column recurrence, seeded from the sectoral P̄(m, m).

    Returns (column dict n -> P̄(n, m), sectoral value P̄(m, m))."""
    col = {}
    if m == 0:
        pmm = mpf(1)
    elif m == 1:
        pmm = sqrt(3) * u
    else:
        pmm = u * sqrt(mpf(2 * m + 1) / mpf(2 * m)) * pbar_prev_sectoral
    col[m] = pmm
    if m + 1 <= n_max:
        col[m + 1] = sqrt(mpf(2 * m + 3)) * t * pmm
    for n in range(m + 2, n_max + 1):
        a = sqrt(mpf((2 * n - 1) * (2 * n + 1)) / mpf((n - m) * (n + m)))
        b = sqrt(
            mpf((2 * n + 1) * (n + m - 1) * (n - m - 1))
            / mpf((n - m) * (n + m) * (2 * n - 3))
        )
        col[n] = a * t * col[n - 1] - b * col[n - 2]
    return col, pmm


def acceleration(pos, n_max, m_max, gm, radius, cnm, snm):
    """Body-fixed acceleration of the truncated geopotential at `pos`.

    Sums the spherical-coordinate gradient of
        V = (GM/r) Σ_n (R/r)^n Σ_m P̄nm(cosθ) (C̄nm cos mλ + S̄nm sin mλ)
    with the theta derivative from the analytic identity
        dP̄nm/dθ = (n t P̄nm − sqrt((n²−m²)(2n+1)/(2n−1)) P̄(n−1)m) / u,
    then rotates (a_r, a_θ, a_λ) into Cartesian axes."""
    x, y, z = (mpf(v) for v in pos)
    r = sqrt(x * x + y * y + z * z)
    t = z / r  # cos(theta)
    u = sqrt(x * x + y * y) / r  # sin(theta), >= 0
    lam = atan2(y, x)

    q = radius / r

    # Accumulate the three gradient sums.
    sum_r = mpf(0)  # Σ (n+1) (R/r)^n P̄nm (C cos + S sin)
    sum_t = mpf(0)  # Σ (R/r)^n dP̄nm/dθ (C cos + S sin)
    sum_l = mpf(0)  # Σ (R/r)^n m P̄nm (−C sin + S cos)

    qn = [q**n for n in range(n_max + 1)]

    pbar_sectoral = mpf(1)
    for m in range(0, min(m_max, n_max) + 1):
        col, pbar_sectoral = legendre_column(m, n_max, t, u, pbar_sectoral)
        cml, sml = cos(m * lam), sin(m * lam)
        for n in range(m, n_max + 1):
            cn = cnm.get((n, m), mpf(0))
            sn = snm.get((n, m), mpf(0))
            trig = cn * cml + sn * sml
            p = col[n]
            sum_r += mpf(n + 1) * qn[n] * p * trig
            # dP̄nm/dθ via the analytic identity (u > 0 for all test cases).
            f = (
                sqrt(mpf((n * n - m * m) * (2 * n + 1)) / mpf(2 * n - 1))
                if n > m
                else mpf(0)
            )
            dp = (mpf(n) * t * p - f * col.get(n - 1, mpf(0))) / u
            sum_t += qn[n] * dp * trig
            sum_l += qn[n] * mpf(m) * p * (-cn * sml + sn * cml)

    gm_r2 = gm / (r * r)
    a_r = -gm_r2 * sum_r  # ∂V/∂r
    a_t = gm_r2 * sum_t  # (1/r) ∂V/∂θ
    a_l = gm_r2 * sum_l / u  # (1/(r sinθ)) ∂V/∂λ

    # Spherical → Cartesian: e_r, e_θ, e_λ at (θ, λ).
    cl, sl = cos(lam), sin(lam)
    ax = a_r * u * cl + a_t * t * cl - a_l * sl
    ay = a_r * u * sl + a_t * t * sl + a_l * cl
    az = a_r * t - a_t * u
    return ax, ay, az


def main():
    n_needed = max(n for _, n, _ in CASES)
    gm, radius, cnm, snm = parse_gfc(GFC_PATH, n_needed)
    print(f"# EGM2008_120: GM = {gm}, R = {radius}, mp.dps = {mp.dps}")
    for pos, n_max, m_max in CASES:
        ax, ay, az = acceleration(pos, n_max, m_max, gm, radius, cnm, snm)
        print(f"pos = {pos}, n_max = {n_max}, m_max = {m_max}")
        for name, v in (("ax", ax), ("ay", ay), ("az", az)):
            print(f"  {name} = {mp.nstr(v, 20)}")


if __name__ == "__main__":
    main()
