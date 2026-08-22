#!/usr/bin/env python3
"""Pre-populate ~/.cache/brahe with the kernels/models CI depends on.

Reads `.github/brahe-data-manifest.txt` (one artifact per line) and downloads
each entry via brahe's own caching downloaders, so a warm run is idempotent:
already-cached artifacts fast-path without hitting the network.

Three manifest line forms are supported:
    <kernel-name>              -> brahe.load_spice_kernel(name), which downloads and
                                   caches known DE/PCK/satellite kernel names
                                   via `download_spice_kernel`'s name
                                   resolution.
    icgem:<body>:<model-name>  -> brahe.datasets.icgem.download_model(body, model-name)
    horizons:<body>:<start>:<stop>
                               -> brahe.datasets.sbdb resolves <body> to its SPK ID,
                                   then brahe.datasets.horizons generates and caches
                                   an SPK spanning the two `YYYY-MM-DD` TDB dates.
                                   Warms both the SBDB and Horizons caches.

An entry carrying a prefix this script does not recognize is a hard error
rather than being passed to `load_spice_kernel` as a kernel name, so a new
manifest form added without a matching handler here fails with a message
naming the prefix.

Every entry is attempted even when an earlier one fails, so a single evicted
or unreachable artifact does not prevent the rest of the cache from being
refreshed. Downloads are retried with a linear backoff, since the upstream
mirrors (NAIF, ICGEM, SBDB, Horizons) fail transiently more often than they
fail permanently. The script still exits non-zero when anything failed.

This is the download-only counterpart to `scripts/warm_cartopy.py`: CI
workflows restore/save `~/.cache/brahe` via `actions/cache`, keyed on the
manifest's hash, and either run this script directly (the weekly keep-warm
workflow) or rely on the test suite's own fixtures to populate the same
cache (regular test/integration runs).
"""

import sys
import time
from pathlib import Path

import brahe as bh
from brahe import datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / ".github" / "brahe-data-manifest.txt"

# Download attempts per entry, and the base delay between them (seconds). The
# delay scales with the attempt number, matching the `curl --retry 3
# --retry-delay 5` used for the star catalogs in warm_data_cache.yml.
ATTEMPTS = 3
RETRY_DELAY = 5.0


def _read_manifest() -> list[str]:
    """Read non-blank, non-comment manifest entries.

    Lines are stripped before the comment test so an indented `#` comment is
    ignored rather than being read as a kernel name.
    """
    lines = (line.strip() for line in MANIFEST_PATH.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _tdb_midnight(date: str) -> bh.Epoch:
    """Parse a `YYYY-MM-DD` manifest date as midnight TDB."""
    year, month, day = (int(part) for part in date.split("-"))
    return bh.Epoch.from_datetime(year, month, day, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)


def _warm_kernel(entry: str) -> str:
    """Warm a NAIF kernel referenced by name."""
    bh.load_spice_kernel(entry)
    assert entry in bh.loaded_spice_kernels(), (
        f"{entry!r} not in loaded_spice_kernels() after load_spice_kernel()"
    )
    return "loaded"


def _warm_icgem(spec: str) -> str:
    """Warm an ICGEM gravity model from a `<body>:<model-name>` spec."""
    body, model_name = spec.split(":", 1)
    return datasets.icgem.download_model(body, model_name)


def _warm_horizons(spec: str) -> str:
    """Warm an SBDB lookup and Horizons SPK from a `<body>:<start>:<stop>` spec.

    The request is built exactly as `examples/examples/dawn_ceres_orbit.py`
    builds it - SPK ID from SBDB, midnight-TDB bounds - because the Horizons
    cache key hashes the command, span, and center, so any difference here
    would cache a second kernel the example never reads. The SBDB cache key
    hashes the search string verbatim, so `<body>` must also match the
    example's spelling for the lookup itself to be warmed.
    """
    body, start, stop = spec.split(":", 2)
    spkid = datasets.sbdb.SBDBClient().lookup(body).naif_id()
    request = datasets.horizons.HorizonsSPKRequest.for_spkid(
        spkid, _tdb_midnight(start), _tdb_midnight(stop)
    )
    return datasets.horizons.HorizonsClient().get_spk(request).path


_PREFIX_HANDLERS = {
    "icgem": _warm_icgem,
    "horizons": _warm_horizons,
}


def _warm_entry(entry: str) -> str:
    """Download one manifest entry, returning a description of what was cached."""
    prefix, separator, spec = entry.partition(":")
    if not separator:
        return _warm_kernel(entry)

    if prefix not in _PREFIX_HANDLERS:
        raise ValueError(
            f"Manifest entry {entry!r} uses unknown prefix {prefix + ':'!r}; "
            f"known prefixes are {sorted(p + ':' for p in _PREFIX_HANDLERS)}. "
            f"Add a handler in {Path(__file__).name} for new manifest forms."
        )

    return _PREFIX_HANDLERS[prefix](spec)


def _warm_with_retries(entry: str) -> str:
    """Warm one entry, retrying transient download failures.

    A `ValueError` is a manifest-format error raised by `_warm_entry` itself,
    so it is reported immediately rather than retried.
    """
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return _warm_entry(entry)
        except ValueError:
            raise
        except Exception as exc:  # retry any download-side failure
            if attempt == ATTEMPTS:
                raise
            print(
                f"  {entry:<28} -> attempt {attempt}/{ATTEMPTS} failed ({exc}); retrying",
                flush=True,
            )
            time.sleep(RETRY_DELAY * attempt)

    raise AssertionError("unreachable: loop either returns or raises")


def main() -> None:
    entries = _read_manifest()
    print(f"Warming brahe data cache from {MANIFEST_PATH.relative_to(REPO_ROOT)}...")

    failures: list[tuple[str, Exception]] = []
    for entry in entries:
        try:
            result = _warm_with_retries(entry)
        except Exception as exc:  # noqa: BLE001 - one bad artifact must not stop the rest
            failures.append((entry, exc))
            print(f"  {entry:<28} -> FAILED: {exc}", flush=True)
            continue
        print(f"  {entry:<28} -> {result}", flush=True)

    if failures:
        print(
            f"\nBrahe data cache partially warm: "
            f"{len(entries) - len(failures)} of {len(entries)} artifact(s) verified, "
            f"{len(failures)} failed:",
            file=sys.stderr,
        )
        for entry, exc in failures:
            print(f"  {entry}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Brahe data cache warm: {len(entries)} artifact(s) verified.")


if __name__ == "__main__":
    main()
