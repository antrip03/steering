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


def test_token_overlap_returns_nan_not_crash_when_near_domain_missing(tmp_path, monkeypatch):
    """Real risk, not hypothetical: half of REDUCED_CONCEPTS' own
    NEAR_DOMAIN_PAIRS entries point outside that 6-concept production set
    (Golf -> Gambling, Poison -> Opioid, Homo Sapiens -> Ancient Rome) -- a
    real run scoped to just those 6 concepts hit exactly this: token_overlap
    raised FileNotFoundError instead of returning NaN for the concept whose
    pairing isn't in scope, crashing compute_all() entirely rather than
    reporting NaN for the metrics that aren't computable with what's
    actually been run."""
    write_fake_features_dir(tmp_path)
    monkeypatch.setattr(em, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(em, "NEAR_DOMAIN_PAIRS", {FAKE_CONCEPTS[0]: "Some Concept Not In This Run"})

    assert math.isnan(em.token_overlap(FAKE_CONCEPTS[0]))

    # And compute_all() itself must not crash on this either.
    result = em.compute_all([FAKE_CONCEPTS[0]])
    assert math.isnan(result.iloc[0]["entanglement_token_overlap"])
