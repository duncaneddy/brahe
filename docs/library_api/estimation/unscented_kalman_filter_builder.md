# Unscented Kalman Filter Builder

Builder for `UnscentedKalmanFilter`. `builder()` takes the five required inputs (`epoch`, `state`, `initial_covariance`, `force_config`, `config`) directly as arguments. Optional inputs, including measurement models, are set through chained setters.

---

::: brahe.UnscentedKalmanFilterBuilder
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [UnscentedKalmanFilter](unscented_kalman_filter.md) - Sequential state estimator using sigma points
- [UKF Guide](../../learn/estimation/unscented_kalman_filter.md) - Setup, sigma points, and EKF comparison
- [Common Types](common_types.md) - Observation, FilterRecord, configuration types
