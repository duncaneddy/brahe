# Batch Least Squares Builder

Builder for `BatchLeastSquares`. `builder()` takes the five required inputs (`epoch`, `state`, `apriori_covariance`, `force_config`, `config`) directly as arguments. Optional inputs, including measurement models, are set through chained setters.

::: brahe.BatchLeastSquaresBuilder
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [BatchLeastSquares](batch_least_squares.md) - Iterative batch least squares estimator
- [Batch Least Squares (Learn)](../../learn/estimation/batch_least_squares.md) -- conceptual guide
- [Measurement Models](measurement_models.md) -- built-in and custom models
