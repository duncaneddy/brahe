#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Low-Thrust Orbit Raising with the Orbion Aurora Hall Thruster

This example demonstrates how to:
1. Model a commercial electric propulsion system (Orbion Aurora)
2. Use extended state dynamics to track propellant mass depletion
3. Compare orbit raising performance at 100W vs 300W power levels
4. Visualize altitude gain, mass consumption, and delta-v accumulation

The Orbion Aurora is a Hall-effect thruster designed for small satellites,
offering continuous throttling from 100W to 300W with xenon propellant.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import os
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import brahe as bh

bh.initialize_eop()
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:thruster_specs]
# Orbion Aurora Hall Thruster Specifications
# Source 1: https://orbionspace.com/product/
# Source 2: https://orbionspace.com/wp-content/uploads/2021/08/Orbion_Aurora_Datasheet_2021.pdf

# 100W Configuration
THRUST_100W = 5.7e-3  # N (5.7 mN)
ISP_100W = 950.0  # seconds
MASS_FLOW_100W = 0.53e-6  # kg/s (0.53 mg/s)

# 300W Configuration
THRUST_300W = 19.0e-3  # N (19 mN)
ISP_300W = 1370.0  # seconds
MASS_FLOW_300W = 1.3e-6  # kg/s (1.3 mg/s)

# Standard gravity for Isp calculations
G0 = 9.80665  # m/s^2

# Verify mass flow rates match thrust equation: mdot = F / (Isp * g0)
mdot_check_100w = THRUST_100W / (ISP_100W * G0)
mdot_check_300w = THRUST_300W / (ISP_300W * G0)

print("Orbion Aurora Hall Thruster Specifications:")
print("\n100W Configuration:")
print(f"  Thrust:     {THRUST_100W * 1e3:.1f} mN")
print(f"  Isp:        {ISP_100W:.0f} s")
print(f"  Mass flow:  {MASS_FLOW_100W * 1e6:.2f} mg/s (datasheet)")
print(f"              {mdot_check_100w * 1e6:.2f} mg/s (computed from F/Isp*g0)")

print("\n300W Configuration:")
print(f"  Thrust:     {THRUST_300W * 1e3:.1f} mN")
print(f"  Isp:        {ISP_300W:.0f} s")
print(f"  Mass flow:  {MASS_FLOW_300W * 1e6:.2f} mg/s (datasheet)")
print(f"              {mdot_check_300w * 1e6:.2f} mg/s (computed from F/Isp*g0)")
# --8<-- [end:thruster_specs]

# --8<-- [start:spacecraft_config]
# Spacecraft configuration
# Aurora system specs from datasheet:
#   Dry mass: 8.3 kg (thruster + PPU + PMA + harness)
#   Wet mass: 14.3 kg (with max propellant)
#   Max propellant: 6.0 kg xenon
SYSTEM_DRY_MASS = 8.3  # kg - Aurora system dry mass
MAX_PROPELLANT = 6.0  # kg - maximum xenon capacity (14.3 - 8.3)

SPACECRAFT_DRY_MASS = 50.0  # kg - small satellite bus
TOTAL_DRY_MASS = SPACECRAFT_DRY_MASS + SYSTEM_DRY_MASS  # kg
PROPELLANT_MASS = 4.0  # kg xenon (partial load for this mission)

INITIAL_WET_MASS = TOTAL_DRY_MASS + PROPELLANT_MASS

print("\nSpacecraft Configuration:")
print(f"  Spacecraft bus:      {SPACECRAFT_DRY_MASS:.0f} kg")
print(f"  Aurora system:       {SYSTEM_DRY_MASS:.1f} kg")
print(f"  Propellant (Xe):     {PROPELLANT_MASS:.1f} kg (max: {MAX_PROPELLANT:.1f} kg)")
print(f"  Total dry mass:      {TOTAL_DRY_MASS:.1f} kg")
print(f"  Total wet mass:      {INITIAL_WET_MASS:.1f} kg")
# --8<-- [end:spacecraft_config]

# --8<-- [start:initial_orbit]
# Create initial epoch and low Earth orbit
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Initial circular orbit at 400 km altitude
initial_altitude = 400e3  # meters
oe_initial = np.array([bh.R_EARTH + initial_altitude, 0.001, 51.6, 0.0, 0.0, 0.0])
state_eci = bh.state_koe_to_eci(oe_initial, bh.AngleFormat.DEGREES)

# Orbital period for reference
orbital_period = bh.orbital_period(oe_initial[0])

print("\nInitial Orbit:")
print(f"  Altitude:       {initial_altitude / 1e3:.0f} km")
print(f"  Semi-major axis: {oe_initial[0] / 1e3:.1f} km")
print(f"  Inclination:    {oe_initial[2]:.1f} deg (ISS-like)")
print(f"  Orbital period: {orbital_period / 60:.1f} minutes")
# --8<-- [end:initial_orbit]


# --8<-- [start:dynamics_functions]
def create_thrust_dynamics(thrust, mass_flow_rate):
    """Create control and additional dynamics functions for a given thrust configuration.

    The state vector is extended to 7 elements: [x, y, z, vx, vy, vz, mass]
    - control_input: Returns thrust acceleration in velocity direction
    - additional_dynamics: Returns mass flow rate

    Args:
        thrust: Thrust force in Newtons
        mass_flow_rate: Mass flow rate in kg/s

    Returns:
        Tuple of (control_input, additional_dynamics) functions
    """

    def control_input(t, state, params):
        """Apply thrust acceleration in the prograde (velocity) direction."""
        dx = np.zeros(len(state))

        # Get current mass from extended state
        mass = state[6]

        # Compute thrust acceleration
        velocity = state[3:6]
        v_mag = np.linalg.norm(velocity)
        if v_mag > 1e-10:
            v_hat = velocity / v_mag
            acceleration = (thrust / mass) * v_hat
            dx[3:6] = acceleration

        return dx

    def additional_dynamics(t, state, params):
        """Model mass depletion during thrusting."""
        dx = np.zeros(len(state))

        # Mass decreases at the mass flow rate
        # dm/dt = -mdot (negative because mass is decreasing)
        dx[6] = -mass_flow_rate

        return dx

    return control_input, additional_dynamics


# --8<-- [end:dynamics_functions]

# --8<-- [start:propagation]
# Simulation duration: 24 hours of continuous thrusting
SIMULATION_DURATION = 24 * 3600.0  # seconds

print("\nSimulation:")
print(f"  Duration: {SIMULATION_DURATION / 3600:.0f} hours")
print(f"  Orbits:   ~{SIMULATION_DURATION / orbital_period:.0f}")

# Extended initial state: [x, y, z, vx, vy, vz, mass]
initial_state = np.concatenate([state_eci, [INITIAL_WET_MASS]])

# Two-body force model for clean demonstration
force_config = bh.ForceModelConfig.two_body()

# Create dynamics for 100W configuration
control_100w, dynamics_100w = create_thrust_dynamics(THRUST_100W, MASS_FLOW_100W)

# Create dynamics for 300W configuration
control_300w, dynamics_300w = create_thrust_dynamics(THRUST_300W, MASS_FLOW_300W)

# Create propagators with extended state
print("\nPropagating 100W configuration...")
prop_100w = bh.NumericalOrbitPropagator(
    epoch,
    initial_state.copy(),
    bh.NumericalPropagationConfig.default(),
    force_config,
    additional_dynamics=dynamics_100w,
    control_input=control_100w,
)
prop_100w.propagate_to(epoch + SIMULATION_DURATION)
print("  Complete!")

print("Propagating 300W configuration...")
prop_300w = bh.NumericalOrbitPropagator(
    epoch,
    initial_state.copy(),
    bh.NumericalPropagationConfig.default(),
    force_config,
    additional_dynamics=dynamics_300w,
    control_input=control_300w,
)
prop_300w.propagate_to(epoch + SIMULATION_DURATION)
print("  Complete!")
# --8<-- [end:propagation]

# --8<-- [start:analysis]
print("\nAnalyzing results...")

# Get trajectories (stores full extended state)
traj_100w = prop_100w.trajectory
traj_300w = prop_300w.trajectory

# Sample trajectories at regular intervals
dt_sample = 600.0  # 10-minute sampling
times_hours = []
alt_100w = []
alt_300w = []
mass_100w = []
mass_300w = []
dv_100w = []
dv_300w = []

# Track cumulative delta-v
cumulative_dv_100w = 0.0
cumulative_dv_300w = 0.0
last_t = 0.0
last_m_100 = INITIAL_WET_MASS
last_m_300 = INITIAL_WET_MASS

t = 0.0
while t <= SIMULATION_DURATION:
    current_epoch = epoch + t

    # Get full state from trajectory (includes mass as 7th element)
    state_100 = traj_100w.state(current_epoch)
    state_300 = traj_300w.state(current_epoch)

    # Extract position and mass
    r_100 = np.linalg.norm(state_100[:3])
    r_300 = np.linalg.norm(state_300[:3])
    m_100 = state_100[6]
    m_300 = state_300[6]

    # Compute instantaneous delta-v rate and accumulate
    if t > 0:
        dt = t - last_t
        # Use average mass for better accuracy
        avg_m_100 = (last_m_100 + m_100) / 2
        avg_m_300 = (last_m_300 + m_300) / 2
        cumulative_dv_100w += (THRUST_100W / avg_m_100) * dt
        cumulative_dv_300w += (THRUST_300W / avg_m_300) * dt

    times_hours.append(t / 3600.0)
    alt_100w.append((r_100 - bh.R_EARTH) / 1e3)
    alt_300w.append((r_300 - bh.R_EARTH) / 1e3)
    mass_100w.append(m_100)
    mass_300w.append(m_300)
    dv_100w.append(cumulative_dv_100w)
    dv_300w.append(cumulative_dv_300w)

    last_t = t
    last_m_100 = m_100
    last_m_300 = m_300
    t += dt_sample

# Final results
final_alt_100w = alt_100w[-1]
final_alt_300w = alt_300w[-1]
final_mass_100w = mass_100w[-1]
final_mass_300w = mass_300w[-1]
propellant_used_100w = INITIAL_WET_MASS - final_mass_100w
propellant_used_300w = INITIAL_WET_MASS - final_mass_300w

# Theoretical delta-v from Tsiolkovsky
theoretical_dv_100w = ISP_100W * G0 * np.log(INITIAL_WET_MASS / final_mass_100w)
theoretical_dv_300w = ISP_300W * G0 * np.log(INITIAL_WET_MASS / final_mass_300w)

print("\n" + "=" * 60)
print("Results Summary (24 hours continuous thrust)")
print("=" * 60)
print(f"\n{'Parameter':<30} {'100W':>12} {'300W':>12}")
print("-" * 60)
print(
    f"{'Initial altitude (km)':<30} {initial_altitude / 1e3:>12.1f} {initial_altitude / 1e3:>12.1f}"
)
print(f"{'Final altitude (km)':<30} {final_alt_100w:>12.1f} {final_alt_300w:>12.1f}")
print(
    f"{'Altitude gain (km)':<30} {final_alt_100w - initial_altitude / 1e3:>12.1f} {final_alt_300w - initial_altitude / 1e3:>12.1f}"
)
print(
    f"{'Propellant used (g)':<30} {propellant_used_100w * 1e3:>12.1f} {propellant_used_300w * 1e3:>12.1f}"
)
print(f"{'Final mass (kg)':<30} {final_mass_100w:>12.2f} {final_mass_300w:>12.2f}")
print(f"{'Delta-v applied (m/s)':<30} {dv_100w[-1]:>12.1f} {dv_300w[-1]:>12.1f}")
print(
    f"{'Theoretical delta-v (m/s)':<30} {theoretical_dv_100w:>12.1f} {theoretical_dv_300w:>12.1f}"
)
print(
    f"{'Thrust-to-weight (mN/kg)':<30} {THRUST_100W * 1e3 / INITIAL_WET_MASS:>12.3f} {THRUST_300W * 1e3 / INITIAL_WET_MASS:>12.3f}"
)
# --8<-- [end:analysis]

# --8<-- [start:visualization_comparison]
# Create comparison plots
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Altitude Over Time",
        "Fuel Mass Remaining",
        "Cumulative Delta-V",
        "Altitude vs Propellant Used",
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)

# Color scheme
color_100w = "blue"
color_300w = "red"

# Altitude over time
fig.add_trace(
    go.Scatter(
        x=times_hours,
        y=alt_100w,
        mode="lines",
        line=dict(color=color_100w, width=2),
        name="100W (5.7 mN)",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=times_hours,
        y=alt_300w,
        mode="lines",
        line=dict(color=color_300w, width=2),
        name="300W (19 mN)",
    ),
    row=1,
    col=1,
)

# Fuel mass (propellant remaining)
fuel_100w = [m - TOTAL_DRY_MASS for m in mass_100w]
fuel_300w = [m - TOTAL_DRY_MASS for m in mass_300w]

fig.add_trace(
    go.Scatter(
        x=times_hours,
        y=fuel_100w,
        mode="lines",
        line=dict(color=color_100w, width=2),
        name="100W",
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=times_hours,
        y=fuel_300w,
        mode="lines",
        line=dict(color=color_300w, width=2),
        name="300W",
        showlegend=False,
    ),
    row=1,
    col=2,
)

# Cumulative delta-v
fig.add_trace(
    go.Scatter(
        x=times_hours,
        y=dv_100w,
        mode="lines",
        line=dict(color=color_100w, width=2),
        name="100W",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=times_hours,
        y=dv_300w,
        mode="lines",
        line=dict(color=color_300w, width=2),
        name="300W",
        showlegend=False,
    ),
    row=2,
    col=1,
)

# Altitude vs propellant used (efficiency comparison)
prop_used_100w = [INITIAL_WET_MASS - m for m in mass_100w]
prop_used_300w = [INITIAL_WET_MASS - m for m in mass_300w]

fig.add_trace(
    go.Scatter(
        x=[p * 1e3 for p in prop_used_100w],  # Convert to grams
        y=alt_100w,
        mode="lines",
        line=dict(color=color_100w, width=2),
        name="100W",
        showlegend=False,
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=[p * 1e3 for p in prop_used_300w],  # Convert to grams
        y=alt_300w,
        mode="lines",
        line=dict(color=color_300w, width=2),
        name="300W",
        showlegend=False,
    ),
    row=2,
    col=2,
)

# Update axes labels
fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
fig.update_yaxes(title_text="Altitude (km)", row=1, col=1)

fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
fig.update_yaxes(title_text="Fuel Mass (kg)", row=1, col=2)

fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
fig.update_yaxes(title_text="Delta-V (m/s)", row=2, col=1)

fig.update_xaxes(title_text="Propellant Used (g)", row=2, col=2)
fig.update_yaxes(title_text="Altitude (km)", row=2, col=2)

fig.update_layout(
    title="Orbion Aurora Performance Comparison: 100W vs 300W",
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
    height=800,
    margin=dict(l=60, r=40, t=80, b=100),
)
# --8<-- [end:visualization_comparison]

# Validation
assert final_alt_300w > final_alt_100w, "300W should raise orbit faster than 100W"
assert propellant_used_300w > propellant_used_100w, "300W should use more propellant"
assert final_mass_100w > TOTAL_DRY_MASS, "Should not exhaust propellant at 100W"
assert final_mass_300w > TOTAL_DRY_MASS, "Should not exhaust propellant at 300W"

print("\nExample validated successfully!")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section
# ============================================================================

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

light_path, dark_path = save_themed_html(fig, OUTDIR / f"{SCRIPT_NAME}_comparison")
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nOrbion Aurora Orbit Raising Example Complete!")
