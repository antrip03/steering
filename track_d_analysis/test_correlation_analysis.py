"""
Integration-style test: exercises build_combined_results() + compute_correlations()
(real code, not mocked) against the shared synthetic fixture (tests/fixtures.py) --
both Track B's parquet output shape and Track C's, together, through the actual
merge. This is exactly the test that would have caught Blocking #3 from
REPO_DIAGNOSTIC.md (the silent all-NaN merge bug): it asserts the entanglement_*
columns survive the merge as real values, not NaN. No model/GPU/HF access needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests", ROOT / "track_b_entanglement"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import correlation_analysis as ca  # noqa: E402
import entanglement_metrics as em  # noqa: E402
from fixtures import FAKE_CONCEPTS, write_fake_eval_results, write_fake_features_dir  # noqa: E402


def test_build_combined_results_preserves_real_entanglement_values(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    eval_results_path = tmp_path / "erasure_eval_results.parquet"
    write_fake_features_dir(features_dir)
    write_fake_eval_results(eval_results_path)

    monkeypatch.setattr(em, "FEATURES_DIR", features_dir)
    # build_combined_results() calls compute_entanglement() with no args, which
    # defaults to the real NATURAL_CONCEPTS -- patch that too, or compute_all()
    # silently looks for the real 15 concepts' parquets (none of which exist
    # here) and returns an empty, columnless DataFrame instead of the fixture.
    monkeypatch.setattr(em, "NATURAL_CONCEPTS", FAKE_CONCEPTS)
    monkeypatch.setattr(
        em,
        "NEAR_DOMAIN_PAIRS",
        {
            FAKE_CONCEPTS[0]: FAKE_CONCEPTS[1],
            FAKE_CONCEPTS[1]: FAKE_CONCEPTS[0],
            FAKE_CONCEPTS[2]: FAKE_CONCEPTS[0],
        },
    )
    monkeypatch.setattr(ca, "EVAL_RESULTS_PATH", eval_results_path)

    combined = ca.build_combined_results()

    assert set(combined["concept"]) == set(FAKE_CONCEPTS)

    # The actual regression this test guards against: entanglement_* columns
    # must survive the merge as real values, not be silently NaN'd out by a
    # pd.merge column-name collision (see REPO_DIAGNOSTIC.md Blocking #3).
    for col in ("entanglement_cosine", "entanglement_pullin_rate", "entanglement_token_overlap"):
        assert combined[col].notna().all(), f"{col} came back all-NaN -- merge bug regression"

    # efficacy/specificity_* from Track C's fixture must also survive untouched.
    for col in ("efficacy", "specificity_simdomain", "specificity_mmlu"):
        assert combined[col].notna().all(), f"{col} came back all-NaN -- Track C side of the merge broke"

    row = combined[combined["concept"] == FAKE_CONCEPTS[0]].iloc[0]
    assert row["efficacy"] == 0.05
    assert row["specificity_simdomain"] == 0.82
    assert row["specificity_mmlu"] == 0.60


def test_compute_correlations_produces_non_trivial_rho(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    eval_results_path = tmp_path / "erasure_eval_results.parquet"
    write_fake_features_dir(features_dir)
    write_fake_eval_results(eval_results_path)

    monkeypatch.setattr(em, "FEATURES_DIR", features_dir)
    # build_combined_results() calls compute_entanglement() with no args, which
    # defaults to the real NATURAL_CONCEPTS -- patch that too, or compute_all()
    # silently looks for the real 15 concepts' parquets (none of which exist
    # here) and returns an empty, columnless DataFrame instead of the fixture.
    monkeypatch.setattr(em, "NATURAL_CONCEPTS", FAKE_CONCEPTS)
    monkeypatch.setattr(
        em,
        "NEAR_DOMAIN_PAIRS",
        {
            FAKE_CONCEPTS[0]: FAKE_CONCEPTS[1],
            FAKE_CONCEPTS[1]: FAKE_CONCEPTS[0],
            FAKE_CONCEPTS[2]: FAKE_CONCEPTS[0],
        },
    )
    monkeypatch.setattr(ca, "EVAL_RESULTS_PATH", eval_results_path)

    combined = ca.build_combined_results()
    correlations = ca.compute_correlations(combined)

    # Only 3 concepts in the fixture -- compute_correlations requires >=3 non-null
    # pairs to compute a rho at all, so n should be exactly 3 for every pair, not
    # the "always nan, always n=0" failure mode of the merge bug.
    assert (correlations["n"] == 3).all()
    assert correlations["spearman_rho"].notna().all()
