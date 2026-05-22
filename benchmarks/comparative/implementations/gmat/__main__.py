"""GMAT subprocess entry point.

Reads a single ``BenchmarkInput`` JSON document from stdin, dispatches it
to the appropriate GMAT adapter, and writes the result JSON to stdout —
matching the protocol used by the Java and Rust subprocess backends.

Why this exists: when the accuracy harness drives GMAT in-process, GMAT's
C++ library accumulates state across many sequential task calls and
eventually segfaults (``Pure virtual function called!``) — most reliably
when ``force_model.accel_point_mass_gravity`` runs after a long chain of
other tasks. Running each GMAT task in a fresh subprocess guarantees the
library starts from a clean state every time, at the cost of one GMAT
initialization per task. That cost is amortized for perf runs (one init
covers N iterations of the same task) and trivial for accuracy runs
(comparable to the Java/Rust subprocess startup we already accept).

Errors:
- If GMAT is not configured (``GMAT_ROOT_PATH`` unset or invalid),
  ``_ensure_gmat()`` raises ImportError; we exit 2 with the message on
  stderr so the parent can render the usual "GMAT not ready" skip line.
- Any other exception during dispatch exits 1 with the message on stderr;
  the parent treats this as a per-task failure (the rest of the sweep
  continues).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from benchmarks.comparative.implementations.gmat import dispatch


def main() -> int:
    try:
        input_data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"GMAT subprocess: invalid input JSON: {exc}", file=sys.stderr)
        return 1

    try:
        task_result = dispatch(input_data)
    except ImportError as exc:
        # _ensure_gmat() raises ImportError when GMAT isn't installed or
        # GMAT_ROOT_PATH isn't set. Use exit code 2 so the parent can
        # disambiguate "not configured" from "ran but failed".
        print(f"GMAT not ready: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"GMAT subprocess error: {exc}", file=sys.stderr)
        return 1

    output = asdict(task_result)
    # Match the subprocess protocol used by the Java and Rust backends:
    # top-level keys task, iterations, times_seconds, results, metadata.
    payload = {
        "task": output["task_name"],
        "iterations": output["iterations"],
        "times_seconds": output["times_seconds"],
        "results": output["results"],
        "metadata": {
            **output.get("metadata", {}),
            "library": output["library"],
            "language": output["language"],
        },
    }
    json.dump(payload, sys.stdout, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
