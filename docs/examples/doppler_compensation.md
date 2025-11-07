# Calculating Doppler Compensation

In this example we'll compute the Doppler compensation required for a ground station to maintain communications with a satellite in low Earth orbit (LEO). We'll simulate the International Space Station (ISS) passing over the Cape Canaveral ground station from the NASA Near Earth Network (NEN) and calculate the frequency shift due to the relative motion between the satellite and ground station during the access window.

To accomplish this we'll define a custom access property computer that calculates the Doppler shift in Hz for S-band uplink communications (2.2 GHz) and X-Band downlink communications (8.4 GHz) based on the relative velocity between the satellite and ground station along the line of sight. We'll assume that the satellite uses a fixed carrier frequency for both uplink and downlink, and the ground station must adjust its local oscillator frequency to compensate for the Doppler shift in order to maintain a stable communication link.

---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, download the TLE for the ISS (25544) from CelesTrak, and load the NASA Near Earth Network ground station network, and select the Cape Canaveral ground station.

``` python
--8<-- "./examples/examples/doppler_compensation.py:preamble"
```

We download the ISS TLE directly by NORAD ID and load all NASA NEN ground stations:

``` python
--8<-- "./examples/examples/doppler_compensation.py:download_iss"
```

Then select Cape Canaveral from the loaded stations:

``` python
--8<-- "./examples/examples/doppler_compensation.py:load_nen_stations"
```

## Custom Doppler Shift Property Computer

Next, we'll define a custom access property computer that calculates the Doppler shift compensation required for maintaining stable communication links.

### Doppler Shift Physics

For electromagnetic signals traveling between a moving satellite and a stationary ground station, the observed frequency differs from the transmitted frequency due to relative motion along the line-of-sight from the transmitter to the receiver. Using the non-relativistic Doppler approximation (valid when $|v_{los}| \ll c$), the received frequency $f_r$ given a transmitted frequency $f_x$ is:

$$
f_r = f_x \left( 1 - \frac{v_{los}}{c} \right)
$$

where $f_x$ is the transmitted carrier frequency, $v_{los}$ is the velocity along the line of sight _from_ the station _to_ the spacecraft (negative when approaching, positive when receding), and $c$ is the speed of light.

**Sign Convention:**

- $v_{los} < 0$: satellite approaching → frequency increases and wavelength decreases (blueshift)
- $v_{los} > 0$: satellite receeding → frequency decreases and wavelength increases (redshift)

### Compensation Strategy

To maintain a stable communication link, the ground station must compensate for the Doppler shift that arises from the relative line-of-sight velocity $v_{los}$ between the satellite and the station. Both the satellite and ground station have design frequencies that they expect to operate at for uplink. The challenge is determining how much the ground station must adjust its transmit and receive frequencies to account for the Doppler effect.

#### Downlink Compensation (Satellite → Ground)

The spacecraft transmits at a fixed design frequency $f_x^d$ (e.g., 8.4 GHz for X-band). The ground station's initial receiver frequency $f_r^0$ is initially tuned to $f_r^0 = f_x^d$, expecting to receive at the design frequency. The challenge then is to determine the required frequency adjustment $\Delta f_r = f_r - f_r^0$ that is added to base the receiver frequency to correctly receive the Doppler-shifted signal.

Starting from the Doppler relation:

$$
f_r = f_x \left( 1 - \frac{v_{los}}{c} \right)
$$

where $f_r$ is the actual received frequency, $f_x$ is the transmitted frequency, and $v_{los}$ is the line-of-sight velocity (negative when approaching, positive when receding).

Since the spacecraft transmits at the fixed design frequency $f_x = f_x^d$:

$$
f_r = f_x^d \left( 1 - \frac{v_{los}}{c} \right)
$$

The receiver compensation term is the difference between the actual received frequency and the initial tuning:

$$
\Delta f_r = f_r - f_r^0 = f_x^d \left( 1 - \frac{v_{los}}{c} \right) - f_r^0
$$

Since $f_r^0 = f_x^d$:

$$
\begin{align}
\Delta f_r &= f_x^d \left( 1 - \frac{v_{los}}{c} \right) - f_r^0 \\
&= f_x^d \left( 1 - \frac{v_{los}}{c} \right) - f_x^d \\
&= f_x^d \left( 1 - \frac{v_{los}}{c}  - 1 \right) \\
&= -f_x^d \frac{v_{los}}{c}
\end{align}
$$

**Therefore, the downlink compensation is:**

$$
\boxed{\Delta f_r = -f_x^d \frac{v_{los}}{c}}
$$

**Sign interpretation:**
- When approaching ($v_{los} < 0$): $\Delta f_r > 0$ → tune receiver higher (blueshift)
- When receding ($v_{los} > 0$): $\Delta f_r < 0$ → tune receiver lower (redshift)

#### Uplink Compensation (Ground → Satellite)

For uplink, the spacecraft expects to receive at its design frequency $f_r^d$ (e.g., 2.2 GHz for S-band). The ground station initially transmits at $f_x^0 = f_r^d$, but must pre-compensate so that after Doppler shift, the spacecraft receives exactly $f_r^d$. We must determine the required frequency adjustment $\Delta f_x = f_x - f_x^0$ that the ground station must apply to its transmit frequency.

We again start from the Doppler relation:

$$
\begin{align}
f_r^d &= (f_x + \Delta f_x) \left( 1 - \frac{v_{los}}{c} \right)
&= f_x^0 \left( 1 - \frac{v_{los}}{c} \right) + \Delta f_x \left( 1 - \frac{v_{los}}{c} \right)
\end{align}
$$

Solving for $\Delta f_x$:

$$
\Delta f_x \left( 1 - \frac{v_{los}}{c} \right) = f_r^d - f_x^0 \left( 1 - \frac{v_{los}}{c} \right)
$$

$$
\Delta f_x = \frac{f_r^d - f_x^0 \left( 1 - \frac{v_{los}}{c} \right)}{1 - \frac{v_{los}}{c}}
$$

Since we know that $f_x^0 = f_r^d$, we can substitute and simplify:

$$
\begin{align}
\Delta f_x &= \frac{f_r^d - f_r^d \left( 1 - \frac{v_{los}}{c} \right)}{1 - \frac{v_{los}}{c}} \\
&= \frac{f_r^d \left( 1 - \left( 1 - \frac{v_{los}}{c} \right) \right)}{1 - \frac{v_{los}}{c}} \\
&= f_r^d \frac{\frac{v_{los}}{c}}{1 - \frac{v_{los}}{c}} \\
&= f_x^0 \frac{\frac{v_{los}}{c}}{\frac{c - v_{los}}{c}} \\
&= f_x^0 \frac{v_{los}}{c - v_{los}}
\end{align}
$$

**And so we find the uplink compensation is:**

$$
\boxed{\Delta f_x = f_x^0 \frac{v_{los}}{c - v_{los}}}
$$

!!! note "Frequency Scaling"
    The magnitude of Doppler compensation scales with carrier frequency. X-band (8.4 GHz) experiences about $8.4/2.2 \approx 3.8\times$ the frequency shift of S-band (2.2 GHz) for the same line-of-sight velocity.

### Implementation

We create a custom property computer that calculates the line-of-sight velocity from the satellite state and computes Doppler compensation for both S-band uplink (2.2 GHz) and X-band downlink (8.4 GHz):

``` python
--8<-- "./examples/examples/doppler_compensation.py:custom_doppler_computer"
```

The property computer extracts the satellite velocity from the provided ECEF state, projects it onto the line-of-sight unit vector to get the line-of-sight velocity, and applies the Doppler formula for both frequency bands.



## Access Computation

Next, we'll compute the access windows between the ISS and the Cape Canaveral ground station over a 72-hour period with our custom Doppler shift property computer to calculate the Doppler compensation required to establish S-band and X-band communications during each access.

``` python
--8<-- "./examples/examples/doppler_compensation.py:access_computation"
```

This computes all access windows where the ISS rises above 5° elevation as viewed from Cape Canaveral, and automatically calculates the Doppler compensation frequencies at the midpoint of each window using our custom property computer.

## Ground Track Visualization

We first visualize the ISS ground track over one orbital period, showing Cape Canaveral's location and its communication cone based on the 5° minimum elevation constraint:

``` python
--8<-- "./examples/examples/doppler_compensation.py:ground_track"
```

The resulting plot shows the ISS ground track in red and Cape Canaveral with its communication cone in blue:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/doppler_compensation_groundtrack_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/doppler_compensation_groundtrack_dark.html"  loading="lazy"></iframe>
</div>

## Doppler Compensation Analysis

For detailed analysis, we select one access window extract the computed doppler compensation value. This provides a high-resolution profile showing how the Doppler compensation varies as the satellite approaches, reaches closest approach, and recedes:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/doppler_compensation_doppler_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/doppler_compensation_doppler_dark.html"  loading="lazy"></iframe>
</div>

The top panel shows $v_{los}$ (line-of-sight velocity from ground station to satellite), confirming our sign convention: **negative when approaching** (start of pass), crossing through zero at closest approach, then **positive when receding** (end of pass).

The middle and bottom panels show the Doppler compensation frequencies with **opposite signs** as required:

- **S-band uplink** (middle): Uses $\Delta f_x = f_x^0 v_{los}/(c - v_{los})$, so it's negative when approaching (transmit lower) and positive when receding (transmit higher)
- **X-band downlink** (bottom): Uses $\Delta f_r = -f_x^d v_{los}/c$, so it's positive when approaching (receive higher) and negative when receding (receive lower)

Note that X-band requires approximately 3.8× more compensation than S-band due to its higher carrier frequency (8.4 GHz vs 2.2 GHz). The uplink compensation includes the $(c - v_{los})$ denominator term for proper pre-compensation.

``` python
--8<-- "./examples/examples/doppler_compensation.py:doppler_visualization"
```

## Doppler Compensation Data

Below is a table of sampled Doppler compensation values during the access window. The compensation frequency indicates the adjustment needed for the ground station equipment:

<div class="center-table" markdown="1">

{{ read_csv('figures/doppler_compensation_data.csv') }}

</div> 


## Full Code Example

```python title="doppler_compensation.py"
--8<-- "./examples/examples/doppler_compensation.py:all"
```

---

## See Also

- [Access Properties](../learn/access_computation/properties.md) - Custom property computers and property types
- [Access Computation](../learn/access_computation/computation.md) - Access search algorithms and configuration
- [Predicting Ground Contacts](ground_contacts.md) - Similar example with statistical analysis
- [CelesTrak Dataset](../learn/datasets/celestrak.md) - Downloading TLE data
- [Ground Station Datasets](../library_api/datasets/groundstations.md) - Loading ground station networks
