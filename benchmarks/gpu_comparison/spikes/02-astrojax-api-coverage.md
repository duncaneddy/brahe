# Spike 02 — astrojax API Coverage for Benchmark Task Families

## Verdict

Every task family in the GPU benchmark spec has a usable astrojax callable
**except one**: explicit time-system conversions between UTC/TAI/GPS/TT.
astrojax does not expose `utc_to_tai`, `tai_to_gps`, `tt_to_utc`, etc. as
public functions — only the underlying building blocks
(`leap_seconds_tai_utc`, `TT_TAI`, `get_ut1_utc`) and high-level frame
transforms (`state_gcrf_to_itrf` etc.) that consume an `Epoch` and an
`EOPData` directly.

All 35 callables tested below are JIT-able and vmap-able under
`JAX_ENABLE_X64=1`. EOP-backed frame transforms work with brahe's bundled
`finals.all.iau2000.txt` (relying on Spike 1).

## Coverage Table

| Spec task family | astrojax module | Public callable(s) | JIT-able? | vmap-able? | Notes |
|---|---|---|---|---|---|
| Time conversions (UTC↔TAI↔GPS↔TT) | `astrojax.time`, `astrojax.eop` | `leap_seconds_tai_utc`, constant `TT_TAI`, `get_ut1_utc(eop, mjd_utc)` — **no combined converters** | YES (parts) | YES (parts) | **GAP**: no public `utc_to_tai` / `tai_to_gps` / `tt_to_utc` / etc. See Gap §1. |
| Time conversions (caldate ↔ MJD ↔ JD) | `astrojax.time` | `caldate_to_mjd`, `mjd_to_caldate`, `caldate_to_jd`, `jd_to_caldate`, `mjd_to_jd`, `jd_to_mjd` | YES | YES | Date-component conversions are fine; only system-offset conversions are missing. |
| Coordinates: geodetic↔ECEF | `astrojax.coordinates.geodetic` | `position_geodetic_to_ecef(x_geod, use_degrees=False)`, `position_ecef_to_geodetic(x_ecef, use_degrees=False)` | YES | YES (axis 0 over Nx3) | Bowring iteration uses `jax.lax.while_loop`; jits fine. |
| Coordinates: Keplerian↔Cartesian | `astrojax.coordinates.keplerian` | `state_koe_to_eci(x_oe, use_degrees=False)`, `state_eci_to_koe(x_cart, use_degrees=False)` | YES | YES (axis 0 over Nx6) | Element order `[a, e, i, RAAN, omega, M]` (Brahe-compatible). |
| Coordinates: topocentric (ENZ→AzEl) | `astrojax.coordinates.topocentric` | `position_enz_to_azel(x_enz, use_degrees=False)`, `rotation_ellipsoid_to_enz`, `relative_position_ecef_to_enz`, `relative_position_enz_to_ecef`, `rotation_enz_to_ellipsoid` | YES | YES (`in_axes=(0, 0)` for relative variants) | `relative_position_ecef_to_enz` takes `(location_ecef, r_ecef, use_geodetic=True)`; `use_geodetic` is static — pass it as a Python kwarg, not a traced value. |
| Frames: GCRF↔ITRF | `astrojax.frames.gcrf_itrf` (re-exported from `astrojax.frames`) | `rotation_gcrf_to_itrf(eop, epc)`, `rotation_itrf_to_gcrf`, `state_gcrf_to_itrf(eop, epc, x_gcrf)`, `state_itrf_to_gcrf`, plus ECI/ECEF aliases | YES | YES (`in_axes=(None, 0, 0)` — keep `eop` static, vmap over `Epoch` pytree + state) | Needs `JAX_ENABLE_X64=1` and an `EOPData` instance (`load_eop_from_file(...)` per Spike 1). |
| Frames: TEME↔GCRF | `astrojax.frames.teme` | `state_teme_to_gcrf(eop, epc, x_teme)`, `state_gcrf_to_teme`, `state_teme_to_itrf`, `state_itrf_to_teme`, `rotation_teme_to_pef`, `rotation_pef_to_teme`, `rotation_teme_to_itrf`, `rotation_itrf_to_teme` | YES | YES (`in_axes=(None, 0, 0)`) | Not re-exported from top-level `astrojax`; import from `astrojax.frames.teme`. |
| SGP4 propagation | `astrojax.sgp4` | `create_sgp4_propagator(line1, line2)` (factory; `parse_tle` + `sgp4_init`), `sgp4_propagate_unified(params, tsince)`, `sgp4_propagate_unified_unbounded` | YES | YES | Existing benchmarks already use this pattern; confirmed jit/vmap in `benchmarks/comparative`. |
| Numerical two-body / J2 | `astrojax.integrators` + `astrojax.orbit_dynamics.factory` | `create_orbit_dynamics(eop, epoch_0, config=None)` → `dynamics(t, x)` closure; `rk4_step(dynamics, t, state, dt)`, `rkf45_step`, `dp54_step`, `rkn1210_step` | YES | YES (`vmap` over state) | Factory pattern: build `dynamics` once, then `vmap(lambda x: rk4_step(dyn, 0.0, x, dt).state)`. J2 not a separate model — use `ForceModelConfig(gravity_type="spherical_harmonics", gravity_degree=2, gravity_order=0, gravity_model=GravityModel.from_type("JGM3"))`. |
| Force model: 5×5 / 20×20 / 20×20+drag+SRP+third-body | `astrojax.orbit_dynamics.{gravity,drag,srp,third_body,factory}` | `accel_point_mass(r_obj, r_body, gm)`, `accel_gravity(r_obj)`, `accel_gravity_spherical_harmonics(r_eci, R_eci_to_ecef, gravity_model, n_max, m_max)`, `accel_drag(x, density, mass, area, cd, T)`, `accel_srp(r_obj, r_sun, mass, cr, area, p0)`, `accel_third_body_sun(epc, r_obj)`, `accel_third_body_moon(epc, r_obj)`, `create_orbit_dynamics(...)` | YES | YES (selectively — see notes) | Easiest: drive everything through `create_orbit_dynamics` + `ForceModelConfig(...)`. The factory closure resolves `gravity_degree`, `m_max`, drag/SRP toggles at trace time, so each (degree, order, toggle) combination yields a separately-jitted graph. `gravity_model` and `n_max`/`m_max` are *not* traceable — they must be captured as Python values (so degree×order sweeps run as separate compilations). `harris_priester` density model is in-tree; `nrlmsise00` needs `SpaceWeatherData`. |

### Smoke-test summary (35 / 35 pass)

Run from the spike script:

```
PASS=35  FAIL=0  INFO=2
```

INFO rows:
- `time/system_conv_search`: searching `dir(astrojax)` for `utc|tai|gps|tt`
  matches only `get_ut1_utc` — confirms the gap.
- `sgp4/api`: enumerated public callables for reference.

## Gaps

### Gap §1 — UTC↔TAI↔GPS↔TT explicit converters (Plan Task 15/16)

Astrojax does **not** currently expose `utc_to_tai`, `tai_to_utc`,
`utc_to_gps`, `gps_to_utc`, `utc_to_tt`, `tt_to_utc`, `tai_to_tt`, or
`tt_to_tai` as public callables. It only exposes the building blocks:

- `astrojax.time.TT_TAI = 32.184` (Python constant)
- `astrojax.time.leap_seconds_tai_utc(mjd_utc) -> jax.Array` (table lookup)
- `astrojax.eop.get_ut1_utc(eop, mjd_utc) -> jax.Array` (EOP-based UT1
  offset, not a relevant pair for the GPS/TAI/TT triangle)
- GPS-time offset (TAI−GPS = 19 s) does not appear in the codebase at all
  (confirmed via grep).

The frame-transform code calls `leap_seconds_tai_utc` + adds `TT_TAI`
inline (`gcrf_itrf.py` `_mjd_tt`). It is a couple-of-lines change for
astrojax to expose `utc_to_tai`/`utc_to_tt`/etc. as public functions, but
**as of this spike they are not exposed**.

> **Downstream impact (Plan Task 15/16):** the explicit time-system
> conversion benchmark must declare
> `configs=["brahe-rust-rayon"]` (or whichever brahe-only config tag the
> harness uses) and skip the astrojax row with
> `config_not_supported_by_task`. *Alternatively*, the harness can
> implement a thin wrapper in `data_alignment.py` that calls
> `leap_seconds_tai_utc` + adds `TT_TAI` / `19.0` for the GPS/TT
> conversions. If the project decides to ship the wrapper, treat it as a
> non-benchmarked utility, not as an astrojax public API parity claim.

**No other rows are gaps.** All other task families have working
public callables that pass jit + vmap.

## Working import + call patterns (kernel-builder ready)

These are the exact import + call patterns to copy into the kernel
builders in Plan Tasks 15–21.

### Time (caldate ↔ MJD ↔ JD — Plan Task 15)

```python
from astrojax.time import (
    caldate_to_mjd, mjd_to_caldate,
    caldate_to_jd, jd_to_caldate,
    mjd_to_jd, jd_to_mjd,
    leap_seconds_tai_utc,
)
# JIT
mjd = jax.jit(caldate_to_mjd)(2024, 6, 15, 12, 0, 0.0)
# vmap a (N,) array of MJDs
jds = jax.jit(jax.vmap(mjd_to_jd))(mjds)
leap = jax.jit(jax.vmap(leap_seconds_tai_utc))(mjds)
```

### Coordinates: geodetic ↔ ECEF (Plan Task 17)

```python
from astrojax.coordinates import (
    position_geodetic_to_ecef, position_ecef_to_geodetic,
)
# Single (3,) -> (3,)
xyz = jax.jit(position_geodetic_to_ecef)(jnp.array([lon, lat, alt]))
# Batched (N,3) -> (N,3)
xyzs = jax.jit(jax.vmap(position_geodetic_to_ecef))(geods_N3)
# use_degrees is a static Python kwarg; bake it via functools.partial if
# you want to sweep on it.
```

### Coordinates: Keplerian ↔ Cartesian (Plan Task 17)

```python
from astrojax.coordinates import state_koe_to_eci, state_eci_to_koe
# Element order: [a, e, i, RAAN, omega, M]
cart = jax.jit(state_koe_to_eci)(jnp.array([a, e, i, raan, omega, M]))
carts = jax.jit(jax.vmap(state_koe_to_eci))(oes_N6)  # (N,6) -> (N,6)
```

### Coordinates: topocentric (ENZ → AzEl) (Plan Task 17)

```python
from astrojax.coordinates import (
    position_enz_to_azel,
    rotation_ellipsoid_to_enz, rotation_enz_to_ellipsoid,
    relative_position_ecef_to_enz, relative_position_enz_to_ecef,
)
azel = jax.jit(position_enz_to_azel)(jnp.array([e, n, z]))  # (3,)
azels = jax.jit(jax.vmap(position_enz_to_azel))(enzs_N3)    # (N,3) -> (N,3)
# Station-to-target ENZ — vmap over both batched args
enzs = jax.jit(
    jax.vmap(relative_position_ecef_to_enz, in_axes=(0, 0))
)(stations_N3, targets_N3)
```

### Frames: GCRF ↔ ITRF (Plan Task 18)

```python
from astrojax import Epoch
from astrojax.eop import load_eop_from_file
from astrojax.frames import (
    rotation_gcrf_to_itrf, rotation_itrf_to_gcrf,
    state_gcrf_to_itrf, state_itrf_to_gcrf,
)
# JAX_ENABLE_X64=1 strongly recommended for EOP precision
eop = load_eop_from_file("data/eop/finals.all.iau2000.txt")
epc = Epoch(2024, 6, 15, 12, 0, 0.0)

# Single rotation
R = jax.jit(rotation_gcrf_to_itrf)(eop, epc)
# Single state (6,) -> (6,)
x_itrf = jax.jit(state_gcrf_to_itrf)(eop, epc, x_gcrf)

# Batched: stack Epoch pytrees, vmap over (epoch, state), keep eop static
epochs = [Epoch(2024, 6, 15, h, 0, 0.0) for h in range(0, N)]
batched_epoch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *epochs)
xs_itrf = jax.jit(
    jax.vmap(state_gcrf_to_itrf, in_axes=(None, 0, 0))
)(eop, batched_epoch, xs_gcrf_N6)
```

### Frames: TEME ↔ GCRF (Plan Task 18)

```python
from astrojax import Epoch
from astrojax.eop import load_eop_from_file
from astrojax.frames.teme import (  # NOT re-exported at top level
    state_teme_to_gcrf, state_gcrf_to_teme,
    state_teme_to_itrf, state_itrf_to_teme,
)
eop = load_eop_from_file("data/eop/finals.all.iau2000.txt")
epc = Epoch(2024, 6, 15, 12, 0, 0.0)
x_gcrf = jax.jit(state_teme_to_gcrf)(eop, epc, x_teme)
# Batched (as above for GCRF↔ITRF)
xs_gcrf = jax.jit(
    jax.vmap(state_teme_to_gcrf, in_axes=(None, 0, 0))
)(eop, batched_epoch, xs_teme_N6)
```

### SGP4 propagation (Plan Task 19)

```python
from astrojax.sgp4 import (
    create_sgp4_propagator,
    sgp4_propagate_unified,
)
# Factory parses a TLE line pair and returns a jit-friendly closure
prop = create_sgp4_propagator(line1, line2)
# prop(tsince_minutes) -> (r, v) in TEME, km / km/s by sgp4 convention
r, v = jax.jit(prop)(0.0)
rs, vs = jax.jit(jax.vmap(prop))(tsinces_N)
```

(Already exercised by `benchmarks/comparative` — see existing harness for
batched-over-TLE patterns using `parse_tle` + `sgp4_init_jax`.)

### Numerical two-body / J2 (Plan Task 20)

```python
from astrojax import Epoch
from astrojax.eop import zero_eop
from astrojax.integrators import rk4_step, rkf45_step, dp54_step
from astrojax.orbit_dynamics.config import ForceModelConfig
from astrojax.orbit_dynamics.factory import create_orbit_dynamics
from astrojax.orbit_dynamics.gravity import GravityModel

epoch_0 = Epoch(2024, 6, 15, 12, 0, 0.0)

# Two-body (point mass)
dyn_2body = create_orbit_dynamics(zero_eop(), epoch_0)  # default config

# J2: SH gravity truncated to (2, 0). Need a gravity model loaded.
grav = GravityModel.from_type("JGM3")
dyn_j2 = create_orbit_dynamics(
    zero_eop(), epoch_0,
    ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=grav,
        gravity_degree=2, gravity_order=0,
    ),
)

# Single step
x_next = jax.jit(lambda t, x: rk4_step(dyn_2body, t, x, dt).state)(0.0, x0)
# Batched over initial states (vmap over state, keep dt scalar)
xs_next = jax.jit(
    jax.vmap(lambda x: rk4_step(dyn_2body, 0.0, x, dt).state)
)(xs_N6)
```

### Force model (5×5 / 20×20 / 20×20+drag+SRP+3-body) (Plan Task 21)

```python
from astrojax import Epoch
from astrojax.eop import load_eop_from_file, zero_eop
from astrojax.integrators import rk4_step
from astrojax.orbit_dynamics.config import ForceModelConfig, SpacecraftParams
from astrojax.orbit_dynamics.factory import create_orbit_dynamics
from astrojax.orbit_dynamics.gravity import GravityModel

eop = load_eop_from_file("data/eop/finals.all.iau2000.txt")
grav = GravityModel.from_type("JGM3")   # or "EGM2008_360", "GGM05S"
epoch_0 = Epoch(2024, 6, 15, 12, 0, 0.0)

def make_step(degree, order, *, drag=False, srp=False, third_body=False):
    cfg = ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=grav,
        gravity_degree=degree, gravity_order=order,
        drag=drag, density_model="harris_priester",
        srp=srp,
        third_body_sun=third_body, third_body_moon=third_body,
        spacecraft=SpacecraftParams(),  # defaults: 1000 kg, 10 m^2, Cd=2.2, Cr=1.3
    )
    dyn = create_orbit_dynamics(eop, epoch_0, cfg)
    return jax.jit(lambda t, x: rk4_step(dyn, t, x, dt).state)

step_5x5     = make_step(5, 5)
step_20x20   = make_step(20, 20)
step_20_full = make_step(20, 20, drag=True, srp=True, third_body=True)

# vmap any of them over a batch of initial states
xs_next = jax.jit(
    jax.vmap(lambda x: rk4_step(
        create_orbit_dynamics(eop, epoch_0, cfg_full), 0.0, x, dt
    ).state)
)(xs_N6)
```

**Important static-arg note for the force-model row.** Each
`(gravity_degree, gravity_order, drag, srp, third_body)` combination must
be a separate Python call that builds its own closure — the factory uses
Python `if` on the config booleans, which is resolved at trace time. The
spec calls for three force-model configs (5×5, 20×20, 20×20+full); this
maps cleanly to three separate jit compilations.

For raw acceleration kernels (without the integrator):

```python
from astrojax.orbit_dynamics import (
    accel_point_mass, accel_gravity, accel_gravity_spherical_harmonics,
    accel_drag, accel_srp,
    accel_third_body_sun, accel_third_body_moon,
)
from functools import partial

# accel_gravity_spherical_harmonics — gravity_model + n_max + m_max
# are static (Python). Bake via partial, then jit/vmap over r and R only.
sh_20 = partial(accel_gravity_spherical_harmonics,
                gravity_model=grav, n_max=20, m_max=20)
a = jax.jit(sh_20)(r_eci, R_eci_to_ecef)
as_batched = jax.jit(jax.vmap(sh_20, in_axes=(0, None)))(rs_N3, R_eci_to_ecef)

# accel_drag(x, density, mass, area, cd, T)
a_drag = jax.jit(accel_drag)(x6, density, mass, area, cd, T_3x3)
# accel_srp(r_obj, r_sun, mass, cr, area, p0)
a_srp = jax.jit(accel_srp)(r3, r_sun3, mass, cr, area, P_SUN)
# accel_third_body_{sun,moon}(epc, r_obj)
a_tbs = jax.jit(accel_third_body_sun)(epoch, r3)
```

## Caveats for downstream implementers

1. **`use_degrees`, `use_geodetic`, `n_max`, `m_max`, `gravity_model`, the
   `ForceModelConfig` booleans, the density / eclipse model strings**:
   all **static** (Python) arguments. They cannot be passed as traced
   JAX values inside a `vmap`. Bake them with `functools.partial` (for
   accel functions) or by building a fresh `ForceModelConfig` +
   `create_orbit_dynamics` closure per combination (for force-model
   sweeps).
2. **`Epoch` is a registered pytree** with three leaves
   (`_jd:int32`, `_seconds:float`, `_kahan_c:float`).  Build a batched
   epoch by stacking each leaf:
   ```python
   batched_epoch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *list_of_epochs)
   ```
   Then `vmap(..., in_axes=(None, 0, 0))(eop, batched_epoch, states)`.
3. **`EOPData` is large** (~20k entries). Keep it static
   (`in_axes=(None, ...)`) — never broadcast it across the batch axis.
4. **`JAX_ENABLE_X64=1`** is required for any EOP-backed transform to
   match brahe to interesting precision. The smoke-test harness sets it
   via `os.environ.setdefault("JAX_ENABLE_X64", "1")` at the top of the
   module, *before* importing JAX.
5. **TEME functions are not re-exported** from `astrojax` or
   `astrojax.frames`; import from `astrojax.frames.teme`.
6. **SGP4 unit convention**: `sgp4_propagate_unified` returns r/v in
   km / (km/min) per the original Vallado/Brandon SGP4 convention.
   Confirm with the existing `benchmarks/comparative` harness before
   converting to m / (m/s) for state-vector comparison.
7. **J2-only** is not a built-in preset; build it as
   `ForceModelConfig(gravity_type="spherical_harmonics", gravity_degree=2, gravity_order=0, gravity_model=...)`.
   The factory still requires *some* loaded `GravityModel`
   (`JGM3`/`EGM2008_360`/`GGM05S` are packaged).

## Reproduction

The smoke-test driver lives at `/tmp/claude/spike_02_smoke.py` (workspace
scratch; not committed). Run with:

```bash
JAX_ENABLE_X64=1 uv run --with astrojax python /tmp/claude/spike_02_smoke.py
```

Final output:

```
=== SUMMARY ===
PASS=35  FAIL=0  INFO=2
```

The two INFO rows are not failures — one enumerates the public SGP4 API
surface for downstream reference, the other documents the time-system
converter gap (`get_ut1_utc` is the only `utc|tai|tt|gps`-matching
public name in the `astrojax` namespace).
