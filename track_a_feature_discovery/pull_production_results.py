"""
Pulls each REDUCED_CONCEPTS concept's canonical Track A output
(schema.feature_artifact_filename) from the HF hub into artifacts/features/,
where Track B (entanglement_metrics.py) and Track C (run_erasure_eval.py)
actually look for it. Production runs happen on Modal/GCP, not this
machine -- discover.py's --push-to-hub gets the parquet onto the hub, but
nothing automatically brings it back down to the local artifacts/ directory
those two tracks read from. This closes that gap.

Skips (doesn't error on) any concept whose file isn't on the hub yet -- lets
this be re-run as more production runs finish, rather than requiring all 6
to be done first.

Usage:
    python pull_production_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import feature_artifact_filename  # noqa: E402
from hub_storage import pull_run_output, list_run_outputs  # noqa: E402
from reductions import REDUCED_CONCEPTS  # noqa: E402

FEATURES_DIR = ROOT / "artifacts" / "features"


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    available = set(list_run_outputs())

    for concept in REDUCED_CONCEPTS:
        filename = feature_artifact_filename(concept)
        if filename not in available:
            print(f"[{concept}] SKIPPED: {filename} not yet on the hub")
            continue
        path = pull_run_output(filename, FEATURES_DIR)
        print(f"[{concept}] pulled to {path}")


if __name__ == "__main__":
    main()
