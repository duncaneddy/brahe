# Comparing Integrator Performance

In this example, we'll compare the performance of different numerical integrators (RK4, RKF45, DP54, and RKN1210) by propagating a satellite orbit over 7 days and analyzing their accuracy and efficiency. We'll measure integration quality by tracking how well each method conserves orbital energy and angular momentum—quantities that should remain constant in two-body orbital dynamics.

---

## Setup

First, we import the necessary libraries:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:preamble"
    ```

Then define the two-body gravitational dynamics function:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:setup"
    ```

We also need helper functions to calculate orbital energy and angular momentum:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:helpers"
    ```

## Initial Conditions

We set up a sun-synchronous LEO satellite orbit at 500 km altitude and calculate reference values for comparison:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:ics"
    ```

This gives us a baseline: the orbital period (~95 minutes) and the initial conserved quantities (energy and angular momentum magnitude). Over 7 days, the satellite will complete approximately 106.5 orbits.

## Running Each Integrator

Now we'll propagate the orbit using each of the four integrators, tracking the angular momentum error at regular intervals.

### RK4 - Fixed-Step Integrator

RK4 uses a fixed time step throughout the integration. We'll use 10-second steps:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:brk"
    ```

With a fixed 10-second step size, RK4 takes **60,480 steps** over 7 days. While this is reliable and predictable, it doesn't adapt to the problem dynamics.

### RKF45 - Adaptive Integrator

RKF45 (Runge-Kutta-Fehlberg) automatically adjusts its step size based on local error estimates:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:rkf"
    ```

RKF45 completes the same 7-day propagation in only **15,687 steps** by taking larger steps when the error is small and smaller steps when more accuracy is needed.

### DP54 - Dormand-Prince Integrator

DP54 is another adaptive method, often more efficient than RKF45:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:dp"
    ```

DP54 is slightly more efficient, requiring **14,264 steps** with the same tolerance settings.

### RKN1210 - High-Precision Integrator

RKN1210 is a high-order Runge-Kutta-Nyström method designed for second-order differential equations. We configure it with tighter tolerances:

=== "Python"
    ``` python
    --8<-- "./examples/examples/comparing_methods.py:rkn"
    ```

RKN1210 achieves much higher accuracy with only **945 steps**-over 50× fewer than RK4!

## Comparison Summary

Here's the summary table showing each integrator's performance:

**Output:**
```
======================================================================
COMPARISON SUMMARY (7 days / 106.5 orbits)
======================================================================

Integrator   Type            Steps    Energy Error    |h| Error
----------------------------------------------------------------------
RK4          Fixed-step      60480    8.951e-02       8.085e+01      
RKF45        Adaptive        15687    1.665e+01       1.504e+04      
DP54         Adaptive        14268    3.319e+00       2.999e+03      
RKN1210      High-precision  945      4.869e-05       4.404e-02  
```

Key observations:

- **RK4** requires many steps (60,480) with moderate accumulated error
- **RKF45** and **DP54** reduce steps by ~4× but show more energy/momentum drift
- **RKN1210** achieves the best accuracy with far fewer steps (~64× fewer than RK4)

The energy and angular momentum errors reveal an important pattern: adaptive low-order methods (RKF45, DP54) can accumulate more error over long integrations despite taking fewer steps, while the high-order RKN1210 maintains excellent conservation properties.

## Angular Momentum Conservation

To visualize how each integrator performs over time, we plot the angular momentum magnitude error:


<div class="plotly-embed">
  <iframe class="only-light" src="../figures/comparing_methods_angular_momentum_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/comparing_methods_angular_momentum_dark.html"  loading="lazy"></iframe>
</div>

## Full Code Example

```python title="comparing_methods.py"
--8<-- "./examples/examples/comparing_methods.py:all"
```

## See Also

- [Numerical Integration](../learn/integrators/index.md) - Understanding how integrators work
- [Integrator API Reference](../library_api/integrators/index.md) - Complete API documentation
- [Orbit Propagation Examples](../examples/index.md) - More orbital mechanics examples
