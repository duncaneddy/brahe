#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "numpy", "plotly"]
# ///
"""
Compares different numerical integrators on two-body orbital dynamics.

This example demonstrates how to use RK4, RKF45, DP54, and RKN1210 integrators
to propagate a satellite orbit over 7 days and compare their accuracy and efficiency.
Angular momentum conservation is used as a measure of integration quality.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import pathlib
import os
import sys
import brahe as bh
import numpy as np
import plotly.graph_objects as go
# --8<-- [end:preamble]

bh.initialize_eop()

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================================
# SETUP: Define the problem
# ============================================================================


# Define dynamics function for two-body orbital mechanics
# State vector: [x, y, z, vx, vy, vz] in meters and meters/second
# --8<-- [start:setup]
def dynamics(t, state):
    """Two-body gravitational dynamics: acceleration = -mu/r^3 * r"""
    r = state[0:3]  # Position vector (m)
    v = state[3:6]  # Velocity vector (m/s)

    r_mag = np.linalg.norm(r)  # Distance from Earth center (m)
    a = -bh.GM_EARTH / (r_mag**3) * r  # Gravitational acceleration (m/s^2)

    return np.concatenate([v, a])  # Return [velocity, acceleration]


# --8<-- [end:setup]


# --8<-- [start:helpers]
def calculate_specific_energy(state):
    """Calculate specific orbital energy: E = v^2/2 - mu/r (J/kg)"""
    r_mag = np.linalg.norm(state[0:3])
    v_mag = np.linalg.norm(state[3:6])
    return 0.5 * v_mag**2 - bh.GM_EARTH / r_mag


def calculate_angular_momentum(state):
    """Calculate specific angular momentum: h = r × v (m^2/s)"""
    r = state[0:3]
    v = state[3:6]
    return np.cross(r, v)


# --8<-- [end:helpers]


# ============================================================================
# INITIAL CONDITIONS: LEO satellite orbit
# ============================================================================

# --8<-- [start:ics]
# Define orbital elements
a = bh.R_EARTH + 500e3  # Semi-major axis: 500 km altitude (m)
e = 0.001  # Eccentricity: nearly circular
i = 97.8  # Inclination: sun-synchronous (degrees)
raan = 0.0  # Right ascension of ascending node (degrees)
argp = 0.0  # Argument of periapsis (degrees)
M = 0.0  # Mean anomaly (degrees)

# Convert orbital elements to Cartesian state vector
oe = np.array([a, e, i, raan, argp, M])
state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Calculate reference values
orbital_period = bh.orbital_period(a)
initial_energy = calculate_specific_energy(state0)
initial_h = calculate_angular_momentum(state0)
h_magnitude_initial = np.linalg.norm(initial_h)

# Set integration time to 7 days
t_end = 7 * 24 * 3600.0  # 7 days in seconds
n_orbits = t_end / orbital_period
# --8<-- [end:ics]

# Print problem setup
print("=" * 70)
print("COMPARING INTEGRATORS ON TWO-BODY ORBITAL DYNAMICS")
print("=" * 70)
print(f"Orbit altitude: {(a - bh.R_EARTH) / 1e3:.1f} km")
print(f"Inclination: {i:.1f}°")
print(f"Orbital period: {orbital_period / 60:.2f} minutes")
print(f"Initial energy: {initial_energy:.6e} J/kg")
print(f"Initial |h|: {h_magnitude_initial:.6e} m²/s")
print(f"Integration duration: 7 days ({n_orbits:.1f} orbits)")
print("=" * 70)
print()

# Store results for comparison
results = []


# ============================================================================
# INTEGRATOR 1: RK4 (Fixed-step)
# ============================================================================

print("1. RK4 - Fourth-order Runge-Kutta (Fixed-step)")
print("-" * 70)

# --8<-- [start:brk]
dt = 10.0  # 10 second steps
config_rk4 = bh.IntegratorConfig.fixed_step(step_size=dt)
integrator_rk4 = bh.RK4Integrator(6, dynamics, config=config_rk4)

# Storage for trajectory analysis
times_rk4 = []
h_errors_rk4 = []

t = 0.0
state = state0.copy()
steps = 0
while t < t_end:
    state = integrator_rk4.step(t, state, dt)
    t += dt
    steps += 1

    # Store data every 10 steps for plotting
    if steps % 10 == 0:
        h = calculate_angular_momentum(state)
        h_error = abs(np.linalg.norm(h) - h_magnitude_initial)
        times_rk4.append(t / 86400.0)  # Convert to days
        h_errors_rk4.append(h_error)

# Final results
r_mag = np.linalg.norm(state[0:3])
final_energy = calculate_specific_energy(state)
energy_error = abs(final_energy - initial_energy)
h_final = calculate_angular_momentum(state)
h_error_final = abs(np.linalg.norm(h_final) - h_magnitude_initial)
# --8<-- [end:brk]

print(f"Configuration: Fixed step size = {dt} seconds")
print(f"Steps taken: {steps}")
print(f"Final altitude: {(r_mag - bh.R_EARTH) / 1e3:.3f} km")
print(f"Energy error: {energy_error:.3e} J/kg")
print(f"Angular momentum error: {h_error_final:.3e} m²/s")
print()

results.append(
    {
        "integrator": "RK4",
        "type": "Fixed-step",
        "steps": steps,
        "energy_error": energy_error,
        "h_error": h_error_final,
        "times": times_rk4,
        "h_errors": h_errors_rk4,
    }
)


# ============================================================================
# INTEGRATOR 2: RKF45 (Adaptive)
# ============================================================================

print("2. RKF45 - Runge-Kutta-Fehlberg (Adaptive)")
print("-" * 70)

# --8<-- [start:rkf]
abs_tol = 1e-10
rel_tol = 1e-9
config_adaptive = bh.IntegratorConfig.adaptive(abs_tol=abs_tol, rel_tol=rel_tol)
integrator_rkf45 = bh.RKF45Integrator(6, dynamics, config=config_adaptive)

times_rkf45 = []
h_errors_rkf45 = []

t = 0.0
state = state0.copy()
dt_current = 10.0
steps = 0
sample_count = 0
while t < t_end:
    result = integrator_rkf45.step(t, state, min(dt_current, t_end - t))
    t += result.dt_used
    state = result.state
    dt_current = result.dt_next
    steps += 1

    # Sample at approximately same rate as RK4
    sample_count += 1
    if sample_count % 10 == 0:
        h = calculate_angular_momentum(state)
        h_error = abs(np.linalg.norm(h) - h_magnitude_initial)
        times_rkf45.append(t / 86400.0)  # Convert to days
        h_errors_rkf45.append(h_error)

r_mag = np.linalg.norm(state[0:3])
final_energy = calculate_specific_energy(state)
energy_error = abs(final_energy - initial_energy)
h_final = calculate_angular_momentum(state)
h_error_final = abs(np.linalg.norm(h_final) - h_magnitude_initial)
# --8<-- [end:rkf]

print(f"Configuration: abs_tol={abs_tol}, rel_tol={rel_tol}")
print(f"Steps taken: {steps}")
print(f"Final altitude: {(r_mag - bh.R_EARTH) / 1e3:.3f} km")
print(f"Energy error: {energy_error:.3e} J/kg")
print(f"Angular momentum error: {h_error_final:.3e} m²/s")
print()

results.append(
    {
        "integrator": "RKF45",
        "type": "Adaptive",
        "steps": steps,
        "energy_error": energy_error,
        "h_error": h_error_final,
        "times": times_rkf45,
        "h_errors": h_errors_rkf45,
    }
)


# ============================================================================
# INTEGRATOR 3: DP54 (Adaptive)
# ============================================================================

print("3. DP54 - Dormand-Prince 5(4) (Adaptive)")
print("-" * 70)

# --8<-- [start:dp]
integrator_dp54 = bh.DP54Integrator(6, dynamics, config=config_adaptive)

times_dp54 = []
h_errors_dp54 = []

t = 0.0
state = state0.copy()
dt_current = 10.0
steps = 0
sample_count = 0
while t < t_end:
    result = integrator_dp54.step(t, state, min(dt_current, t_end - t))
    t += result.dt_used
    state = result.state
    dt_current = result.dt_next
    steps += 1

    sample_count += 1
    if sample_count % 10 == 0:
        h = calculate_angular_momentum(state)
        h_error = abs(np.linalg.norm(h) - h_magnitude_initial)
        times_dp54.append(t / 86400.0)  # Convert to days
        h_errors_dp54.append(h_error)

r_mag = np.linalg.norm(state[0:3])
final_energy = calculate_specific_energy(state)
energy_error = abs(final_energy - initial_energy)
h_final = calculate_angular_momentum(state)
h_error_final = abs(np.linalg.norm(h_final) - h_magnitude_initial)
# --8<-- [end:dp]

print(f"Configuration: abs_tol={abs_tol}, rel_tol={rel_tol}")
print(f"Steps taken: {steps}")
print(f"Final altitude: {(r_mag - bh.R_EARTH) / 1e3:.3f} km")
print(f"Energy error: {energy_error:.3e} J/kg")
print(f"Angular momentum error: {h_error_final:.3e} m²/s")
print()

results.append(
    {
        "integrator": "DP54",
        "type": "Adaptive",
        "steps": steps,
        "energy_error": energy_error,
        "h_error": h_error_final,
        "times": times_dp54,
        "h_errors": h_errors_dp54,
    }
)


# ============================================================================
# INTEGRATOR 4: RKN1210 (High-precision adaptive)
# ============================================================================

print("4. RKN1210 - Runge-Kutta-Nyström 12(10) (High-precision)")
print("-" * 70)

# --8<-- [start:rkn]
abs_tol_hp = 1e-12
rel_tol_hp = 1e-11
config_high_precision = bh.IntegratorConfig.adaptive(
    abs_tol=abs_tol_hp, rel_tol=rel_tol_hp
)
integrator_rkn1210 = bh.RKN1210Integrator(6, dynamics, config=config_high_precision)

times_rkn1210 = []
h_errors_rkn1210 = []

t = 0.0
state = state0.copy()
dt_current = 10.0
steps = 0
sample_count = 0
while t < t_end:
    result = integrator_rkn1210.step(t, state, min(dt_current, t_end - t))
    t += result.dt_used
    state = result.state
    dt_current = result.dt_next
    steps += 1

    sample_count += 1
    if sample_count % 10 == 0:
        h = calculate_angular_momentum(state)
        h_error = abs(np.linalg.norm(h) - h_magnitude_initial)
        times_rkn1210.append(t / 86400.0)  # Convert to days
        h_errors_rkn1210.append(h_error)

r_mag = np.linalg.norm(state[0:3])
final_energy = calculate_specific_energy(state)
energy_error = abs(final_energy - initial_energy)
h_final = calculate_angular_momentum(state)
h_error_final = abs(np.linalg.norm(h_final) - h_magnitude_initial)
# --8<-- [end:rkn]

print(f"Configuration: abs_tol={abs_tol_hp}, rel_tol={rel_tol_hp}")
print(f"Steps taken: {steps}")
print(f"Final altitude: {(r_mag - bh.R_EARTH) / 1e3:.3f} km")
print(f"Energy error: {energy_error:.3e} J/kg")
print(f"Angular momentum error: {h_error_final:.3e} m²/s")
print()

results.append(
    {
        "integrator": "RKN1210",
        "type": "High-precision",
        "steps": steps,
        "energy_error": energy_error,
        "h_error": h_error_final,
        "times": times_rkn1210,
        "h_errors": h_errors_rkn1210,
    }
)


# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("=" * 70)
print("COMPARISON SUMMARY (7 days / {:.1f} orbits)".format(n_orbits))
print("=" * 70)
print()
print(
    f"{'Integrator':<12} {'Type':<15} {'Steps':<8} {'Energy Error':<15} {'|h| Error':<15}"
)
print("-" * 70)
for r in results:
    print(
        f"{r['integrator']:<12} {r['type']:<15} {r['steps']:<8} "
        f"{r['energy_error']:<15.3e} {r['h_error']:<15.3e}"
    )
print()
print("Key Observations:")
print("• RK4 requires many steps with accumulated drift in conserved quantities")
print("• RKF45 and DP54 adapt step size but show energy/momentum drift")
print("• RKN1210 maintains best conservation with far fewer steps")
print("• Angular momentum conservation indicates integration quality")
print("=" * 70)


# ============================================================================
# VISUALIZATION: Angular Momentum Conservation
# ============================================================================

print("\nGenerating angular momentum conservation plot...")

# Create plotly figure
fig = go.Figure()

# Add traces for each integrator
fig.add_trace(
    go.Scatter(
        x=results[0]["times"],
        y=results[0]["h_errors"],
        mode="lines",
        name="RK4 (Fixed-step)",
        line=dict(color="steelblue", width=2),
        hovertemplate="Day: %{x:.2f}<br>|Δh|: %{y:.3e} m²/s<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=results[1]["times"],
        y=results[1]["h_errors"],
        mode="lines",
        name="RKF45 (Adaptive)",
        line=dict(color="coral", width=2),
        hovertemplate="Day: %{x:.2f}<br>|Δh|: %{y:.3e} m²/s<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=results[2]["times"],
        y=results[2]["h_errors"],
        mode="lines",
        name="DP54 (Adaptive)",
        line=dict(color="green", width=2),
        hovertemplate="Day: %{x:.2f}<br>|Δh|: %{y:.3e} m²/s<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=results[3]["times"],
        y=results[3]["h_errors"],
        mode="lines",
        name="RKN1210 (High-precision)",
        line=dict(color="grey", width=2),
        hovertemplate="Day: %{x:.2f}<br>|Δh|: %{y:.3e} m²/s<extra></extra>",
    )
)

# Configure axes
axis_config = {
    "title_font": {"size": 11},
    "tickfont": {"size": 10},
}

fig.update_xaxes(title_text="Time (days)", **axis_config)
fig.update_yaxes(
    title_text="Angular Momentum Error |Δh| (m²/s)", type="log", **axis_config
)

fig.update_layout(
    title="Angular Momentum Conservation Over 7 Days",
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=10)),
)

print("Comparison complete!")
# --8<-- [end:all]

# Expected output:
# ======================================================================
# COMPARING INTEGRATORS ON TWO-BODY ORBITAL DYNAMICS
# ======================================================================
# Orbit altitude: 500.0 km
# Inclination: 97.8°
# Orbital period: 94.62 minutes
# Initial energy: -2.895826e+07 J/kg
# Initial |h|: 5.390639e+10 m²/s
# Integration duration: 7 days (106.5 orbits)
# ======================================================================
#
# 1. RK4 - Fourth-order Runge-Kutta (Fixed-step)
# ----------------------------------------------------------------------
# Configuration: Fixed step size = 10.0 seconds
# Steps taken: 60480
# Final altitude: 499.991 km
# Energy error: 1.063e+02 J/kg
# Angular momentum error: 2.891e+06 m²/s
#
# 2. RKF45 - Runge-Kutta-Fehlberg (Adaptive)
# ----------------------------------------------------------------------
# Configuration: abs_tol=1e-10, rel_tol=1e-09
# Steps taken: 15784
# Final altitude: 499.997 km
# Energy error: 1.983e+04 J/kg
# Angular momentum error: 5.396e+08 m²/s
#
# 3. DP54 - Dormand-Prince 5(4) (Adaptive)
# ----------------------------------------------------------------------
# Configuration: abs_tol=1e-10, rel_tol=1e-09
# Steps taken: 14391
# Final altitude: 499.998 km
# Energy error: 3.949e+03 J/kg
# Angular momentum error: 1.075e+08 m²/s
#
# 4. RKN1210 - Runge-Kutta-Nyström 12(10) (High-precision)
# ----------------------------------------------------------------------
# Configuration: abs_tol=1e-12, rel_tol=1e-11
# Steps taken: 1279
# Final altitude: 500.000 km
# Energy error: 4.058e-02 J/kg
# Angular momentum error: 1.104e+04 m²/s
#
# ======================================================================
# COMPARISON SUMMARY (7 days / 106.5 orbits)
# ======================================================================
#
# Integrator   Type            Steps    Energy Error    |h| Error
# ----------------------------------------------------------------------
# RK4          Fixed-step      60480    1.063e+02       2.891e+06
# RKF45        Adaptive        15784    1.983e+04       5.396e+08
# DP54         Adaptive        14391    3.949e+03       1.075e+08
# RKN1210      High-precision  1279     4.058e-02       1.104e+04
#
# Key Observations:
# • RK4 requires many steps with accumulated drift in conserved quantities
# • RKF45 and DP54 adapt step size but show energy/momentum drift
# • RKN1210 maintains best conservation with far fewer steps
# • Angular momentum conservation indicates integration quality
# ======================================================================
#
# Generating angular momentum conservation plot...
# Comparison complete!

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

light_path, dark_path = save_themed_html(
    fig, OUTDIR / f"{SCRIPT_NAME}_angular_momentum"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
