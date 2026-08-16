"""
Unit tests for compare_fc.py's diff logic against small synthetic parquet
files -- no model/GPU needed, just pandas + tmp_path.
"""
from __future__ import annotations

import pandas as pd

from compare_fc import compare, load_candidates

COLUMNS = ["concept", "layer", "feature_id", "neg", "decoder_direction", "top_tokens", "bottom_tokens", "mass_ratio_or_effect_score", "selected"]


def make_row(layer, feature_id, neg, selected, effect_score=0.1):
    return {
        "concept": "Golf", "layer": layer, "feature_id": feature_id, "neg": neg,
        "decoder_direction": [0.0], "top_tokens": [], "bottom_tokens": [],
        "mass_ratio_or_effect_score": effect_score, "selected": selected,
    }


def write_parquet(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(path, index=False)


def test_identical_pools_and_selection_is_exact_match(tmp_path, capsys):
    rows = [
        make_row(1, 10, False, selected=True),
        make_row(1, 20, False, selected=False),
    ]
    orig_path, reduced_path = str(tmp_path / "orig.parquet"), str(tmp_path / "reduced.parquet")
    write_parquet(orig_path, rows)
    write_parquet(reduced_path, rows)

    compare(orig_path, reduced_path)
    out = capsys.readouterr().out
    assert "Exact match" in out
    assert "DIFFER" not in out


def test_reduced_pool_subset_with_matching_selection_is_not_an_automatic_pass(tmp_path, capsys):
    # original has an extra low-overlap candidate (minmatch=1) that reduced's
    # minmatch=5 search never found at all -- common candidates still agree.
    orig_rows = [
        make_row(1, 10, False, selected=True),
        make_row(1, 99, False, selected=False),  # only in original's pool
    ]
    reduced_rows = [
        make_row(1, 10, False, selected=True),
    ]
    orig_path, reduced_path = str(tmp_path / "orig.parquet"), str(tmp_path / "reduced.parquet")
    write_parquet(orig_path, orig_rows)
    write_parquet(reduced_path, reduced_rows)

    compare(orig_path, reduced_path)
    out = capsys.readouterr().out
    assert "NOT in reduced's" in out
    assert "(1, 99, False)" in out
    assert "Not an automatic pass" in out
    assert "DIFFER" not in out  # the one common candidate still matched


def test_differing_selection_status_is_flagged_clearly(tmp_path, capsys):
    orig_rows = [make_row(1, 10, False, selected=True, effect_score=0.5)]
    reduced_rows = [make_row(1, 10, False, selected=False, effect_score=0.05)]
    orig_path, reduced_path = str(tmp_path / "orig.parquet"), str(tmp_path / "reduced.parquet")
    write_parquet(orig_path, orig_rows)
    write_parquet(reduced_path, reduced_rows)

    compare(orig_path, reduced_path)
    out = capsys.readouterr().out
    assert "1 candidate(s) DIFFER" in out
    assert "(1, 10, False)" in out
    assert "0.5" in out and "0.05" in out
    assert "did NOT reproduce the same selected FC" in out


def test_candidate_only_in_reduced_pool_triggers_warning(tmp_path, capsys):
    # Should not happen in practice (minmatch=5 candidates should be a subset
    # of minmatch=1's), but the script must surface it loudly, not hide it.
    orig_rows = [make_row(1, 10, False, selected=True)]
    reduced_rows = [make_row(1, 10, False, selected=True), make_row(1, 77, False, selected=True)]
    orig_path, reduced_path = str(tmp_path / "orig.parquet"), str(tmp_path / "reduced.parquet")
    write_parquet(orig_path, orig_rows)
    write_parquet(reduced_path, reduced_rows)

    compare(orig_path, reduced_path)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "(1, 77, False)" in out


def test_load_candidates_keys_by_layer_id_neg(tmp_path):
    rows = [make_row(2, 33, True, selected=True, effect_score=0.7)]
    path = str(tmp_path / "x.parquet")
    write_parquet(path, rows)

    loaded = load_candidates(path)
    assert loaded == {(2, 33, True): {"selected": True, "effect_score": 0.7}}
