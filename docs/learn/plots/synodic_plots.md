# Synodic Frame Trajectory Plots

`plot_synodic_3d` renders 3D trajectories in a synodic (two-body rotating) frame: EMR (Earth-Moon Rotating), SER (Sun-Earth Rotating), GSE (Geocentric Solar Ecliptic), or a generic `ReferenceFrame.Synodic(origin, primary, secondary)`. Each input trajectory is converted to ECI and then transformed per-epoch into the requested frame via `state_frame_to_frame`. The frame's primary and secondary bodies are drawn as textured spheres at a single reference epoch, since a synodic frame keeps both bodies on the x-axis at all times.

`plot_earth_moon_rotating_3d` is an alias for `plot_synodic_3d(trajectories, frame="EMR", ...)`; it accepts the same keyword arguments.

## Input Requirement

`plot_synodic_3d` only accepts `OrbitTrajectory` objects, unlike `plot_trajectory_3d` and brahe's other plotting functions, which also accept propagators or raw arrays. Pass a `KeplerianPropagator`'s or numerical propagator's `.trajectory` attribute directly.

## Frame Selection

The `frame` parameter accepts:

- A frame alias string: `'EMR'`, `'SER'`, or `'GSE'`
- Any other `ReferenceFrame` name string accepted by `ReferenceFrame.from_string`
- A `ReferenceFrame.Synodic(origin, primary, secondary)` instance for a custom two-body pair

See [Synodic Reference Frames](../frames/synodic_frames.md) for the axis construction and physical definition of each frame. Passing a non-synodic frame raises `ValueError`.

## Reference Epoch

`reference_epoch` sets the epoch at which the primary and secondary spheres are placed. It defaults to the first epoch of the first trajectory in `trajectories`, and must be supplied explicitly if `trajectories` is empty.

## Example

The example below propagates a LEO trajectory with a `KeplerianPropagator`, then renders it in the Earth-Moon Rotating frame with `plot_earth_moon_rotating_3d` - the `plot_synodic_3d(..., frame="EMR", ...)` alias described above.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/synodic_plot_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/synodic_plot_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="synodic_plot_plotly.py"
    --8<-- "./plots/learn/plots/synodic_plot_plotly.py"
    ```

## Additional Bodies

The `bodies` parameter adds extra textured spheres beyond the frame's primary/secondary, each a dict with `position` (meters, in the synodic frame), `radius` (meters), `texture`, and `name` - the same shape as `additional_bodies` in [`plot_trajectory_3d`](3d_trajectory.md).

## Shared Parameters

`units`, `view_azimuth`, `view_elevation`, `view_distance`, `sphere_resolution_lon`, `sphere_resolution_lat`, `backend`, `width`, and `height` behave identically to [`plot_trajectory_3d`](3d_trajectory.md).

Primary/secondary body textures are downloaded on first use from Solar System Scope:

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.*

---

## See Also

- [3D Trajectory Plots](3d_trajectory.md) - Central-body-centered inertial trajectory plots
- [Synodic Reference Frames](../frames/synodic_frames.md) - EMR, SER, GSE frame definitions
- [Coordinate Systems](../coordinates/index.md) - Understanding reference frames
