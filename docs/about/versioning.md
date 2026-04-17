# Versioning

Brahe follows a versioning scheme modeled on [NumPy's policy](https://numpy.org/doc/stable/dev/depending_on_numpy.html) rather than strict [Semantic Versioning](https://semver.org/). Version numbers are [PEP 440](https://peps.python.org/pep-0440/) compliant and take the form `major.minor.bugfix`:

- **Major** releases (`X.0.0`) are rare. They signal significant API changes or ABI breaks between Rust or Python versions.
- **Minor** releases (`1.Y.0`) are the primary release cadence. They contain new features, deprecations, and removals of code that has already been deprecated in a prior release.
- **Bugfix** releases (`1.2.Z`) contain only bug fixes. They do not contain new features, deprecations, or removals.

This scheme intentionally avoids the forever "0.x" versioning trap that many Rust and scientific software projects fall into, which can deter users from adopting the software. The `1.x` series is intended to signal that Brahe is ready for general use. At the same time, because Brahe is actively growing its user base and incorporating feedback, minor releases may refine the public API in ways that strict SemVer would forbid.

## Deprecation Policy

The long-term target — matching NumPy — is that backwards-incompatible API changes emit a `DeprecationWarning` for **at least two minor releases** before the deprecated code is removed. This gives downstream packages a full release cycle to pick up the warning, migrate, and ship a fixed release before the deprecated path disappears.

!!! danger "Transitional Deprecation Window"
    **Currently, a deprecation may occur and be removed within a single minor release.** As Brahe's adoption grows and its public API stabilizes, this window will expand to multiple minor releases with deprecation warnings, converging on the NumPy policy of at least two minor releases between deprecation and removal.

    During this transitional period, review the [changelog](../changelog.md) of each minor release before upgrading, and consider the pinning guidance below if your project requires stronger stability guarantees.

We recommend configuring your CI to treat `DeprecationWarning` and `FutureWarning` as errors so that upcoming breakage in Brahe is surfaced before it lands in a release that removes the deprecated API. In Python, this can be done with:

```bash
python -W error::DeprecationWarning -W error::FutureWarning -m pytest
```

## Experimental Features

Some features in Brahe are currently marked as experimental. These features are functional but are likely to undergo breaking changes in minor releases as we refine their design and implementation. Experimental features are indicated in the documentation with a warning box, and they are explicitly excluded from the backwards-compatibility expectations described above.

Current experimental features include:

1. Numerical Integration Module
2. Space Weather Module
3. Atmospheric Density Models (NRLMSISE-00)
4. Numerical Propagation
5. Estimation & Filtering (e.g., batch least-squares, Kalman filters)
