# True, Eccentric, and Mean Anomaly

This section deals with the conversion between true, eccentric, and mean 
anomaly. 

True anomaly, frequently denoted $\nu$, is the angular parameter that defines 
the position of an object moving along a Keplerian orbit. It is the angle 
between the eccentricity vector (vector pointing from the main pericenter to 
the periapsis) and the current position of the body in the orbital plane itself.

The eccentric anomaly, $E$, is another angular parameter that defines the position 
of an object moving along a Keplerian orbit if viewed from the center of the 
ellipse. 

Finally, the mean anomaly, $M$, defines the fraction of an orbital period that has 
elapsed since the orbiting object has passed its periapsis. It is the angle 
from the pericenter an object moving on a fictitious circular orbit with the 
same semi-major axis would have progressed through in the same time as the 
body on the true elliptical orbit.

Conversion between all types of angular anomaly is possible. However, there is 
no known direct conversion between true and mean anomaly. Conversion between the two is 
accomplished by transformation through eccentric anomaly.

## True and Eccentric Anomaly Conversions

To convert from true anomaly to eccentric anomaly, you can use the function 
`anomaly_eccentric_to_true`. To perform the reverse conversion use 
`anomaly_true_to_eccentric`.

Eccentric anomaly can be converted to true anomaly by using equations derived using equations 
from Vallado[^1]. Starting from Equation (2-12)
$$
\sin{\nu} = \frac{\sin{E}\sqrt{1-e^2}}{1 - e\cos{E}}
$$
can be divided by
$$
\cos{\nu} =  \frac{\cos{E}-e}{1 - e\cos{E}}
$$
and rearrange to get
$$
\nu = \arctan{\frac{\sin{E}\sqrt{1-e^2}}{\cos{E}-e}}
$$

This conversion is what is implemented by `anomaly_eccentric_to_true`. Similarly, we can derive
$$
E = \arctan{\frac{\sin{\nu}\sqrt{1-e^2}}{\cos{\nu}+e}}
$$
which allows for conversion from true anomaly to eccentric anomaly and is implemented in 
`anomaly_true_to_eccentric`.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/anomaly_true_and_eccentric.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/anomaly_true_and_eccentric.rs:8"
    ```

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_anomaly_true_eccentric_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_anomaly_true_eccentric_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_anomaly_true_eccentric.py"
    --8<-- "./plots/fig_anomaly_true_eccentric.py:8"
    ```

## Eccentric and Mean Anomaly Conversions

To convert from true anomaly to eccentric anomaly, you can use the function
`anomaly_eccentric_to_mean`. To perform the reverse conversion use
`anomaly_mean_to_eccentric`. 

Conversion from eccentric anomaly to mean anomaly is accomplished by application of Kepler's 
equation
$$
M = E - e\sin{E}
$$
which is implemented in `anomaly_eccentric_to_mean`.

Converting back from mean anomaly to eccentric anomaly is more challenging.
There is no known closed-form solution to convert from mean anomaly to eccentric anomaly. 
Instead, we introduce the auxiliary equation
$$
f(E) = E - e\sin(E) - M
$$
And treat the problem as numerically solving for the root of $f$ for a given $M$. This iteration 
can be accomplished using Newton's method. Starting from an initial guess $E_0$ the value of 
$E_*$ can be iteratively updated using
$$
E_{i+1} = \frac{f(E_i)}{f^\prime(E_i)}= E_i - \frac{E_i - e\sin{E_i} - M}{1 - e\cos{E_i}}
$$
This update is performed until a coverage value of
$$
|E_{i+1} - E_i| \leq \Delta_{\text{tol}}
$$
is reached. The value set as 100 times floating-point machine precision `100 * f64::epsilon`.
This conversion is provided by `anomaly_mean_to_eccentric`.

!!! warning

    Because this is a numerical method, convergence is not guaranteed. There is an upper 
    limit of 10 iterations to reach convergence. Since convergence may not occur the output of 
    the function is a `Result`, forcing the user to explicitly handle the case where the algorithm 
    does not converage.

    Since Python lacks Rust's same error handling mechanisms, non-convergence will result in a 
    runtime error.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/anomaly_eccentric_and_mean.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/anomaly_eccentric_and_mean.rs:8"
    ```

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_anomaly_eccentric_mean_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_anomaly_eccentric_mean_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_anomaly_eccentric_mean.py"
    --8<-- "./plots/fig_anomaly_eccentric_mean.py:8"
    ```

## True and Mean Anomaly Conversions

Methods to convert from true anomaly to mean anomaly are 
provided for convenience. These methods simply wrap successive calls to two 
`anomaly_true_to_mean`. To perform the reverse conversion use
`anomaly_mean_to_true`.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/anomaly_true_and_mean.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/anomaly_true_and_mean.rs:8"
    ```

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_anomaly_true_mean_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_anomaly_true_mean_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_anomaly_true_mean.py"
    --8<-- "./plots/fig_anomaly_true_mean.py:8"
    ```

[^1]: D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010  
[https://celestrak.com/software/vallado-sw.php](https://celestrak.com/software/vallado-sw.php)