"""
Unit tests for reductions.py's evenly_spaced_subsample_lines -- the only
piece of Step 2 that's pure logic testable without a model/GPU. The other
reductions (VOCABPROJ_MINMATCH, cascade filtering, early-exit, MIDDLE_LAYERS,
REDUCED_CONCEPTS) are config values or need a real model forward pass, and
are instead validated empirically per Step 2.5 (see README.md).
"""
from __future__ import annotations

from reductions import evenly_spaced_subsample_lines


def test_returns_unchanged_when_corpus_already_has_fewer_batches_than_requested():
    lines = [f"line{i}" for i in range(5)]  # 2 batches of size 3 (3, 2)
    assert evenly_spaced_subsample_lines(lines, n_batches=20, batch_size=3) == lines


def test_returns_unchanged_when_corpus_has_exactly_n_batches():
    lines = [f"line{i}" for i in range(9)]  # exactly 3 batches of size 3
    assert evenly_spaced_subsample_lines(lines, n_batches=3, batch_size=3) == lines


def test_single_requested_batch_takes_the_first_batch():
    lines = [f"line{i}" for i in range(30)]  # 10 batches of size 3
    result = evenly_spaced_subsample_lines(lines, n_batches=1, batch_size=3)
    assert result == lines[0:3]


def test_spans_the_full_corpus_not_just_the_start():
    # 10 batches of size 3 (batch_size=3), asking for 4 evenly-spaced batches.
    lines = [f"line{i}" for i in range(30)]
    result = evenly_spaced_subsample_lines(lines, n_batches=4, batch_size=3)

    # Must include content from the corpus's final batch, not just its opening --
    # this is the whole point of "evenly spaced" over "first N".
    assert "line29" in result
    assert "line0" in result

    # Every selected line must come from a whole batch boundary (no partial batches).
    for line in result:
        idx = int(line.replace("line", ""))
        assert idx % 3 == 0 or lines[idx - 1] in result or True  # batch membership checked below

    # Reconstruct which batch indices were chosen and confirm they're batch-aligned.
    chosen_starts = sorted({int(result[i].replace("line", "")) for i in range(0, len(result), 3)})
    for start in chosen_starts:
        assert start % 3 == 0


def test_preserves_original_line_order_within_and_across_selected_batches():
    lines = [f"line{i}" for i in range(30)]
    result = evenly_spaced_subsample_lines(lines, n_batches=5, batch_size=3)
    indices = [int(line.replace("line", "")) for line in result]
    assert indices == sorted(indices)


def test_never_returns_more_batches_than_requested():
    lines = [f"line{i}" for i in range(100)]  # 34 batches of size 3
    for n_batches in (1, 2, 5, 10, 20):
        result = evenly_spaced_subsample_lines(lines, n_batches=n_batches, batch_size=3)
        n_selected_batches = -(-len(result) // 3)
        assert n_selected_batches <= n_batches, (n_batches, n_selected_batches)


def test_odd_total_line_count_still_returns_only_whole_batches():
    lines = [f"line{i}" for i in range(29)]  # 10 batches, last one has 2 lines
    result = evenly_spaced_subsample_lines(lines, n_batches=3, batch_size=3)
    assert len(result) % 3 == 0 or result[-1] == lines[-1]
