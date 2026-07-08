"""Shared helper for downloading and extracting cached zip resources.

Several plotting features fetch Natural Earth data from naciscdn.org. This
module centralizes that logic so every caller is concurrency-safe and resilient
to truncated upstream responses:

- existence fast-path (no work if the resource is already cached)
- a retry loop with zip-magic validation (catches truncated/HTML error bodies)
- download to a temp file and extract to a temp directory, then an atomic
  rename into place so parallel workers never observe a partial result.
"""

import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import httpx

# Zip magic bytes (PK\x03\x04)
_ZIP_MAGIC = b"PK\x03\x04"

# Some CDNs (naciscdn.org) return 406 Not Acceptable for the default httpx
# User-Agent; present a browser-like UA and a permissive Accept header.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; brahe; +https://github.com/duncaneddy/brahe)"
    ),
    "Accept": "*/*",
}


def _is_zip(path: Path) -> bool:
    """Check that a file starts with the zip magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _ZIP_MAGIC
    except OSError:
        return False


def download_and_extract_zip(
    url: str,
    extract_dir: Path,
    sentinel: Path,
    *,
    description: str,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay: int = 5,
    relocate: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Download a zip and extract it into ``extract_dir``, atomically.

    Args:
        url: URL of the zip archive to download.
        extract_dir: Final directory the extracted contents are published to.
        sentinel: A file whose existence means the resource is fully cached.
        description: Human-readable name used in progress messages.
        timeout: Per-request timeout in seconds.
        max_retries: Number of download/extract attempts before giving up.
        retry_delay: Seconds to wait between attempts.
        relocate: Optional callback invoked on the temporary extraction
            directory before publishing, used to normalize the internal layout
            so that ``sentinel``'s path resolves inside the published directory.

    Returns:
        Path: ``sentinel`` once the resource is cached.

    Raises:
        RuntimeError: If the download or extraction fails after all retries.
    """
    # Fast path: already cached.
    if sentinel.exists():
        return sentinel

    parent = extract_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        # Another worker may have finished while we were retrying.
        if sentinel.exists():
            return sentinel

        print(f"Downloading {description} (attempt {attempt}/{max_retries})...")
        tmp_zip: Optional[Path] = None
        tmp_extract: Optional[Path] = None
        try:
            response = httpx.get(
                url, timeout=timeout, follow_redirects=True, headers=_HEADERS
            )
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                dir=parent, suffix=".zip", delete=False
            ) as tmp:
                # Record the path before writing so a failed write is still
                # cleaned up by the finally block (no orphaned temp zip).
                tmp_zip = Path(tmp.name)
                tmp.write(response.content)

            # Validate the download is actually a zip (catches HTML error pages
            # and truncated bodies from a flaky CDN).
            if not _is_zip(tmp_zip):
                last_error = RuntimeError(
                    f"Downloaded {description} is not a valid zip "
                    f"(got {len(response.content)} bytes, content-type: "
                    f"{response.headers.get('content-type', 'unknown')})"
                )
                if attempt < max_retries:
                    print(f"  Invalid zip, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                continue

            tmp_extract = Path(tempfile.mkdtemp(dir=parent, prefix=".extract-"))
            with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
                zip_ref.extractall(tmp_extract)

            if relocate is not None:
                relocate(tmp_extract)

            # Atomic publish: rename our staged dir into the final location.
            try:
                tmp_extract.rename(extract_dir)
                tmp_extract = None  # ownership transferred
            except OSError:
                # Another worker published first — that's fine, use theirs.
                pass

        except httpx.HTTPError as e:
            last_error = RuntimeError(f"Failed to download {description}: {e}")
            if attempt < max_retries:
                print(f"  Download failed: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            continue
        except (zipfile.BadZipFile, OSError) as e:
            # BadZipFile: corrupt/partial archive. OSError: transient filesystem
            # error during extract or the relocate callback. Both are worth a
            # retry; the finally block cleans up the temp artifacts either way.
            last_error = RuntimeError(f"Failed to extract {description}: {e}")
            if attempt < max_retries:
                print(f"  Extraction failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            continue
        finally:
            if tmp_zip is not None:
                tmp_zip.unlink(missing_ok=True)
            if tmp_extract is not None:
                shutil.rmtree(tmp_extract, ignore_errors=True)

        if sentinel.exists():
            return sentinel

        last_error = RuntimeError(
            f"{description} not found after extraction: {sentinel}"
        )

    raise (
        last_error if last_error else RuntimeError(f"Failed to download {description}")
    )
