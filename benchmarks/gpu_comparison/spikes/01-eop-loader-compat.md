# Spike 01 — EOP and Space-Weather Loader Compatibility

## Verdict

**`direct-load`** — astrojax's public loader functions accept brahe's bundled
data files as-is, with no format conversion required. Both files parsed
successfully and yielded populated `EOPData` / `SpaceWeatherData` objects.

This means Plan Task 14 (`data_alignment.py`) can be a thin wrapper: it just
points astrojax at the absolute paths under `brahe/data/`. No interchange
file, no one-time conversion step.

## Loader function signatures

EOP (`astrojax.eop`, defined in `astrojax/eop/providers.py`):

- `load_eop_from_file(filepath: str | Path) -> EOPData` — parses an IERS
  standard format file (e.g. `finals.all.iau2000.txt`).
- `load_eop_from_standard_file = load_eop_from_file` — alias.
- `load_default_eop() -> EOPData` — loads astrojax's own bundled
  `finals.all.iau2000.txt` via `importlib.resources`.
- `load_cached_eop(filepath=None, *, max_age_days=7.0) -> EOPData` — cache +
  refresh from IERS, falls back to bundled file on failure.

Space weather (`astrojax.space_weather`, defined in
`astrojax/space_weather/providers.py`):

- `load_sw_from_file(filepath: str | Path) -> SpaceWeatherData` — parses a
  CSSI-format file (e.g. `sw19571001.txt`).
- `load_default_sw() -> SpaceWeatherData` — loads astrojax's own bundled
  `sw19571001.txt`.
- `load_cached_sw(filepath=None, *, max_age_days=7.0) -> SpaceWeatherData` —
  cache + refresh from CelesTrak, falls back to bundled file on failure.

Both `load_eop_from_file` and `load_sw_from_file` are exported from the
package `__init__.py` files and are part of the documented public API.

## Reproduction commands

Step 1 — discovery:

```bash
ls /home/deddy/repos/astrojax/src/astrojax/eop/ \
   /home/deddy/repos/astrojax/src/astrojax/space_weather/
grep -rEn "def (load|from_file|parse|read|from_path)" \
   /home/deddy/repos/astrojax/src/astrojax/eop/ \
   /home/deddy/repos/astrojax/src/astrojax/space_weather/
```

Step 2 — EOP load test (brahe's bundled `finals.all.iau2000.txt`):

```bash
uv run --with astrojax python -c "
from astrojax.eop import load_eop_from_file
eop = load_eop_from_file('/home/deddy/repos/brahe/data/eop/finals.all.iau2000.txt')
print('EOP OK', type(eop).__name__,
      'mjd_min=', float(eop.mjd_min),
      'mjd_max=', float(eop.mjd_max),
      'n=', int(eop.mjd.shape[0]))
"
```

Step 3 — space-weather load test (brahe's bundled `sw19571001.txt`):

```bash
uv run --with astrojax python -c "
from astrojax.space_weather import load_sw_from_file
sw = load_sw_from_file('/home/deddy/repos/brahe/data/space_weather/sw19571001.txt')
print('SW OK', type(sw).__name__,
      'mjd_min=', float(sw.mjd_min),
      'mjd_max=', float(sw.mjd_max),
      'n=', int(sw.mjd.shape[0]))
"
```

## Output / error messages

### EOP

```
EOP OK EOPData mjd_min= 41684.0 mjd_max= 61554.0 n= 19871
```

Plus several non-fatal warnings (one per array) because JAX x64 was not
enabled in the spike environment:

```
UserWarning: Explicitly requested dtype float64 requested in array is not
available, and will be truncated to dtype float32. To enable more dtypes,
set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell
environment variable.
```

Also: `An NVIDIA GPU may be present on this machine, but a CUDA-enabled
jaxlib is not installed. Falling back to cpu.` This is expected — the
`uv run --with astrojax` invocation pulls the CPU-only jaxlib from PyPI;
the GPU benchmark harness will use a CUDA-enabled jaxlib and enable x64.

### Space weather

```
SW OK SpaceWeatherData mjd_min= 36112.0 mjd_max= 66793.0 n= 25298
```

No warnings other than the same CPU-fallback notice.

## Downstream impact

**Plan Task 14 (`data_alignment.py`) is a thin wrapper.** Recommended shape:

```python
# benchmarks/gpu_comparison/data_alignment.py
from pathlib import Path
from astrojax.eop import load_eop_from_file
from astrojax.space_weather import load_sw_from_file

BRAHE_DATA = Path(__file__).resolve().parents[2] / "data"
EOP_FILE   = BRAHE_DATA / "eop" / "finals.all.iau2000.txt"
SW_FILE    = BRAHE_DATA / "space_weather" / "sw19571001.txt"

def load_astrojax_eop():
    return load_eop_from_file(EOP_FILE)

def load_astrojax_sw():
    return load_sw_from_file(SW_FILE)
```

For brahe-side loading the equivalent calls already exist in
`brahe::eop::FileEOPProvider::from_standard_file` and the space-weather
loader; the benchmark harness should call both sides on the **same** brahe
file path so that the two stacks are guaranteed bit-identical on input.

Notes for downstream tasks:

- Make sure the benchmark harness sets `JAX_ENABLE_X64=1` (or
  `jax.config.update("jax_enable_x64", True)`) before constructing
  `EOPData` / `SpaceWeatherData`. Otherwise the warnings above become real
  precision loss in the comparison.
- Only the IERS standard file (`finals.all.iau2000.txt`) was tested.
  brahe also ships `EOP_20_C04_one_file_1962-now.txt` (IERS C04 format) —
  astrojax has no loader for the C04 format, so the harness must use the
  standard file. Same picture on the space-weather side:
  `sw19571001.txt` works, `fluxtable.txt` (a different format) is not
  expected to and was not tested.
- The `load_cached_*` variants will silently fall back to astrojax's
  bundled file if the supplied path can't be parsed; for benchmark
  determinism prefer the bare `load_*_from_file` calls so any breakage
  surfaces as an error rather than a silent data swap.
