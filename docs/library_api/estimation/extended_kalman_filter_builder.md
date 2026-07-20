# Extended Kalman Filter Builder

Builder for `ExtendedKalmanFilter`. `builder()` takes the five required inputs (`epoch`, `state`, `initial_covariance`, `force_config`, `config`) directly as arguments. Optional inputs, including measurement models, are set through chained setters.

---

::: brahe.ExtendedKalmanFilterBuilder
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [ExtendedKalmanFilter](extended_kalman_filter.md) - Sequential state estimator using linearized dynamics and measurement models
- [EKF Guide](../../learn/estimation/extended_kalman_filter.md) - Setup, processing, and diagnostics
- [Common Types](common_types.md) - Observation, FilterRecord, configuration types
