# Change Log

<!------------------------------------------------------------------------------------------------->
> ## v0.0.10
> *(under development)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `compat.numba` --> initial support for transparently handling typed Dict, typed List

<!------------------------------------------------------------------------------------------------->
> ## v0.0.9
> *(2025-10-27)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `misc.argument_handling` --> `all_are_none`, `all_are_not_none`, `count_none`, `count_not_none`

- **improved**:
  - apply best practices for coverage reporting of abstract methods (use ellipsis + exclude in coverage config for good measure)
  - `ordered_weighted_mean` & `ordered_weighted_geo_mean`
    - improve numerical stability of exponential weight computation
    - add option to specify target `q` (quantile or orness) value based on which appropriate `c` parameter will be auto-computed

<!------------------------------------------------------------------------------------------------->
> ## v0.0.8
> *(2025-10-26)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `compat` --> `numba`, `is_numba_installed`
  - `caching` --> `per_instance_lru_cache`, `per_instance_cache`

- **improved**:
  - `plotting.canvas` --> `LineStyle`: improved test coverage
  - `math.aggregation` --> move to exponential weighting scheme in `ordered_weighted_mean` & `ordered_weighted_geo_mean`

<!------------------------------------------------------------------------------------------------->
> ## v0.0.7
> *(2025-10-19)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `plotting.canvas` --> `Canvas`, `CanvasRange`, `RangeSpecs`
  - `plotting.utils` --> `TransformLinLog`

<!------------------------------------------------------------------------------------------------->
> ## v0.0.6
> *(2025-10-19)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `plotting.canvas` --> `LineStyle`
  - `plotting.utils` --> `Transform`, `TransformLinear`, `TransformLog`

<!------------------------------------------------------------------------------------------------->
> ## v0.0.5
> *(2025-10-18)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `math.sampling` --> `linspace`, `logspace`
  - `math.aggregation` 
    - `mean`, `weighted_mean`, `ordered_weighted_mean`
    - `geo_mean`, `weighted_geo_mean`, `ordered_weighted_geo_mean`

<!------------------------------------------------------------------------------------------------->
> ## v0.0.4
> *(2025-10-17)*
<!------------------------------------------------------------------------------------------------->

- **new**:
  - `benchmarking` --> `benchmark`
  - `math.utils` --> `clip`
- **updated**:` 
  - `formatting` --> `format_short_time_duration` use more appropriate character for μsec

<!------------------------------------------------------------------------------------------------->
> ## v0.0.3
> *(2025-10-16)*
<!------------------------------------------------------------------------------------------------->

- Fixes to CI/CD pipeline
- **new**: 
  - `benchmarking` --> `Timer`
  - `formatting` --> `format_time_duration`, `format_short_time_duration`, `format_long_time_duration`
  - `math.utils` --> `HALF_EPS`

<!------------------------------------------------------------------------------------------------->
> ## v0.0.2
> *(2025-10-14)*
<!------------------------------------------------------------------------------------------------->

- Improvements to CI/CD pipeline
- **new**: 
  - `collections` --> `zip_random`
  - `math.root_finding` --> `bisection`
  - `math.utils` --> `EPS`, `same_sign`, `sign` 
  

<!------------------------------------------------------------------------------------------------->
> ## v0.0.1
> *(2025-10-11)*
<!------------------------------------------------------------------------------------------------->

- Initial project framework
- Initial CI/CD pipeline