"""
Integration-style test: exercises entanglement_metrics.compute_all() (real
code, not mocked) against the shared synthetic fixture (tests/fixtures.py),
via a monkeypatched FEATURES_DIR/NEAR_DOMAIN_PAIRS. No model/GPU/HF access
needed -- this only reads the fixture's parquet files.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import entanglement_metrics as em  # noqa: E402
from fixtures import FAKE_CONCEPTS, write_fake_features_dir  # noqa: E402


def test_compute_all_against_fixture(tmp_path, monkeypatch):
    write_fake_features_dir(tmp_path)
    monkeypatch.setattr(em, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(
        em,
        "NEAR_DOMAIN_PAIRS",
        {
            FAKE_CONCEPTS[0]: FAKE_CONCEPTS[1],
            FAKE_CONCEPTS[1]: FAKE_CONCEPTS[0],
            FAKE_CONCEPTS[2]: FAKE_CONCEPTS[0],
        },
    )

    result = em.compute_all(FAKE_CONCEPTS)

    assert list(result.columns) == [
        "concept",
        "entanglement_cosine",
        "entanglement_pullin_rate",
        "entanglement_token_overlap",
    ]
    assert set(result["concept"]) == set(FAKE_CONCEPTS)
    assert result["entanglement_cosine"].notna().all()
    assert result["entanglement_pullin_rate"].notna().all()
    assert result["entanglement_token_overlap"].notna().all()

    # fixture: deliberately different candidate/selected mixes per concept
    # (see tests/fixtures.py::_SELECTED_PATTERNS) -> pull-in rates 1/3, 3/4, 0
    expected = dict(zip(FAKE_CONCEPTS, [1 / 3, 3 / 4, 0.0]))
    for _, row in result.iterrows():
        assert math.isclose(row["entanglement_pullin_rate"], expected[row["concept"]])


def test_compute_all_skips_concepts_with_no_track_a_output(tmp_path, monkeypatch):
    write_fake_features_dir(tmp_path)
    monkeypatch.setattr(em, "FEATURES_DIR", tmp_path)

    result = em.compute_all(FAKE_CONCEPTS + ["No Track A Output For This One"])

    assert set(result["concept"]) == set(FAKE_CONCEPTS)
