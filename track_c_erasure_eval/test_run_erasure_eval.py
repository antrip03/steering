"""
Integration-style test for Track C's output-shape logic that doesn't need a
model/GPU: load_selected_features reading a real (fixture) parquet file and
converting selected=True rows into pisces_ref.editor.Feature objects.
evaluate_concept itself needs a live model + Gemini API access and is out of
scope here.

Note: importing this module (and run_erasure_eval.py) transitively pulls in
pisces_ref/evals.py, which imports both `transformers` and `datasets`. In some
environments that combination triggers a native segfault on import (an
ABI/dependency-version conflict between those two packages' compiled
extensions, unrelated to anything in this project's own code) -- if this
module fails to even collect, that's almost certainly what's happening; see
REPO_DIAGNOSTIC.md for the specific repro.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_erasure_eval as rre  # noqa: E402
from fixtures import FAKE_CONCEPTS, write_fake_features_dir  # noqa: E402


def test_load_selected_features_against_fixture(tmp_path, monkeypatch):
    write_fake_features_dir(tmp_path)
    monkeypatch.setattr(rre, "FEATURES_DIR", tmp_path)

    # Fake Concept Alpha's fixture pattern is (True, False, True) -- 3 candidates, 2 selected.
    features = rre.load_selected_features(FAKE_CONCEPTS[0])

    assert len(features) == 2
    assert all(f.neg for f in features)  # both selected rows in that pattern use neg=True (i even)
    assert {f.layer for f in features} == {1, 7}  # layer = 1 + i*3 for i in (0, 2)


def test_load_selected_features_missing_concept_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rre, "FEATURES_DIR", tmp_path)

    try:
        rre.load_selected_features("Not A Real Concept")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for missing Track A output")
