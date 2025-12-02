# Versioning

While Brahe generally tries to adhere to [Semantic Versioning](https://semver.org/) to manage its version numbers. In pratice the versioning strategy is closer to that of Python and Pythongs's SciPy library, where breaking changes may occasionally be introduced in minor releases to facilitate rapid development and improvement. This is for two reasons. First, we want to avoid the forever "0.x" versioning trap that many Rust and scientific software projects fall into, which can deter users from adopting the software for use. Our "1.x" are intended to signal that Brahe is stable and ready for general use. Second, as we again adoption we are actively seeking user feedback and may need to make breaking changes to improve usability, performance, or correctness based on that feedback.

!!! danger "Breaking Changes in Minor Releases"
    If you need guaranteed stability you should pin your project to a specific major.minor version (e.g., `1.2.x`) rather than using a floating version specifier (e.g., `^1.2.0` or `>=1.2.0`).

!!! warning "Experimental Features"
    Some features in Brahe are currently marked as experimental. These features are functional but more likely to undergo changes in minor releases as we refine their design and implementation. Experimental features are indicated in the documentation with a warning box.

## Current Experimental Features

Some features in Brahe are currently marked as experimental. These features are functional but more likely to undergo changes in minor releases as we refine their design and implementation. Experimental features are indicated in the documentation with a warning box.

Current experimental features include:

1. Numerical Integration Module
2. Space Weather Module
3. Atmospheric Density Models (NRLMSISE-00)
4. Numerical Propagation
5. Space Track Client