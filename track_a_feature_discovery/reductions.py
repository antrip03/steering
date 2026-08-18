"""
Runtime-reduction config and helpers, shared by discover.py (opt-in, via
--reduced) and run_kaggle.py (always on). Every reduction here is a
config-level or reordering change -- none of them touch PISCES's selection
criteria or any data file. See README.md's "Runtime reductions" section for
the full rationale and Step 2.5's validation results.

Verified against source, not assumed:
- VOCABPROJ_MINMATCH: PISCES's own paper (arXiv:2505.22586, Appendix A.1)
  states "Features with an intersection size greater than a threshold alpha
  (we used alpha = 4) are selected" -- confirmed by reading the actual PDF
  text, not just trusting the secondhand claim. Note "greater than alpha"
  is a strict `>`, but search_features' own threshold check is `>= minmatch`
  (see pisces_ref/feature_finder.py::search_features) -- so reproducing
  "intersection size > 4" exactly requires minmatch=5, not minmatch=4.
  DROPPED as a reduction after real-run validation -- see this constant's
  own comment below for what was found and why it's back to 1.
- MIDDLE_LAYERS (layers 3-12): NOT independently verified against EMBER's
  Figure 7 -- that paper wasn't available to check against (unlike PISCES's
  own alpha=4, which was verified directly from the PDF). Flagged here so
  this isn't silently presented as equally confirmed.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 2.1 -- corpus size for effect measurement (get_feature_effect) -- DROPPED
# ---------------------------------------------------------------------------
# Tried EFFECT_MEASUREMENT_BATCHES=20 (evenly spaced through the corpus, not
# just the first 20, so the measured effect reflects the whole article's
# content, not just its opening). Real-run validation on Golf/layer-1
# demonstrated this changes the selected FC, not just its speed: two
# identical original-settings (85-batch) runs matched exactly (69/69), so
# the divergence against the 20-batch --reduced run (30/69 exact match) is
# real, not GPU noise. A meaningful chunk of that (~9 candidates) had
# near-zero pos_effect that flipped sign purely because 20 evenly-spaced
# batches measure a smaller, different sample than the full 85 -- a genuine
# correctness cost, not a rounding error, for a project whose whole point is
# getting the *right* feature set. Reverted: --reduced no longer shrinks the
# corpus at all; effect measurement always runs on every line of
# wikipedia_content, same as original settings. The 12-hour-Kaggle-session
# problem this was meant to solve should be handled by spreading the full
# job across multiple sessions via checkpointing (already built into
# get_feature_effect/filter_features_by_mmlu), not by measuring less data.
# EFFECT_MEASUREMENT_BATCHES kept below (unused by default) as a documented
# historical value and a standing target for evenly_spaced_subsample_lines
# if a less-aggressive reduction (e.g. 40 batches) is ever worth revisiting.
EFFECT_MEASUREMENT_BATCHES = 20
EFFECT_BATCH_SIZE = 3  # matches pisces_ref/feature_finder.py's own default


def evenly_spaced_subsample_lines(lines: list[str], n_batches: int, batch_size: int = EFFECT_BATCH_SIZE) -> list[str]:
    """Pick n_batches whole batches (batch_size lines each), evenly spaced
    across the full line range, and return their concatenated lines in order.
    If the corpus already has <= n_batches batches, returns it unchanged.

    Evenly-spaced rather than "first N": a concept's article can front-load
    generic scene-setting/definitional content and only get topic-specific
    deeper in -- sampling only the start risks a systematically different
    (and likely weaker) effect signal than the full run would see.
    """
    total_batches = max(1, -(-len(lines) // batch_size))  # ceil division
    if n_batches >= total_batches:
        return lines

    if n_batches <= 1:
        chosen_batch_indices = [0]
    else:
        chosen_batch_indices = sorted({
            round(i * (total_batches - 1) / (n_batches - 1)) for i in range(n_batches)
        })

    selected: list[str] = []
    for b in chosen_batch_indices:
        start = b * batch_size
        selected.extend(lines[start:start + batch_size])
    return selected


# ---------------------------------------------------------------------------
# 2.2 -- candidate reduction -- DROPPED, kept at the original value
# ---------------------------------------------------------------------------
# Tried minmatch=5, matching PISCES's own paper (alpha=4, see module
# docstring for the >4 vs >=5 nuance). Real-run validation on Golf found it
# is not a viable reduction: minmatch=5 collapsed the candidate pool to ZERO
# at both a tested early layer (1) and a tested middle layer (6), and
# discover.py's --minmatch override (independent of --reduced) confirmed
# it's not a confound with the other three reductions or a layer-1-specific
# VocabProj weakness -- the value itself is the problem. minmatch_sweep.py's
# sweep at layer 6 (1 through 5) then showed why no intermediate value
# helps either: 92 candidates at minmatch=1, 0 at every value from 2 to 5 --
# a hard cliff, not a gradual falloff. Leading hypothesis (not verified
# against other concepts): PISCES's own hand-curated seed lists are
# tightly-clustered proper nouns (e.g. Harry Potter's character names) that
# plausibly co-occur within a single feature's top-tokens naturally, while
# this project's TF-IDF-extracted lists for concepts like Golf are more
# generic vocabulary mapping to more scattered feature directions. Back to
# 1 (the original, unrestricted value) -- --minmatch remains available to
# re-test this on a per-run basis (e.g. against a proper-noun-heavy concept
# like Homo Sapiens) without needing to change this default again.
VOCABPROJ_MINMATCH = 1

# Cascade filtering: a cheap pre-pass on a small subset of the (already
# reduced) 20 batches, used to drop the bottom half of candidates by effect
# score before running the full measurement. This is a heuristic speed-up
# (unlike early-exit below, which is provably exact) -- it changes which
# candidates get the full 20-batch treatment, so it's the one reduction in
# this module that is NOT guaranteed bit-identical to the unreduced baseline.
# Step 2.5's validation checks empirically whether it changes the final FC in
# practice; if it does, don't silently keep it (see README).
CASCADE_PREFILTER_BATCHES = 5
CASCADE_KEEP_FRACTION = 0.5

# Early-exit: enabled inside pisces_ref/feature_finder.py::get_feature_effect
# itself (fork, steering-fixes branch) via early_exit_after_batches. Unlike
# cascade filtering above, this IS provably exact -- see that function's
# docstring for the worst-case-bound argument. Checking after this many
# batches (of the already-reduced EFFECT_MEASUREMENT_BATCHES) whether a
# candidate can mathematically still pass, given the known [-1, 1] bound on
# softmax-diff values.
EARLY_EXIT_AFTER_BATCHES = 5
EARLY_EXIT_MARGIN = 1e-6


# ---------------------------------------------------------------------------
# 2.3 -- layer scope
# ---------------------------------------------------------------------------
# Restricted to layers 3-12 per EMBER's Figure 7 analysis of where Gemma-2-2B
# concept features concentrate (per the task instruction -- NOT independently
# verified against that paper's actual text, unlike VOCABPROJ_MINMATCH above;
# flagged as such rather than presented as equally confirmed). Config value,
# not hardcoded inline, so it's a one-line change to revisit or widen back to
# the full 26-layer range.
MIDDLE_LAYERS = list(range(3, 13))  # layers 3..12 inclusive


# ---------------------------------------------------------------------------
# 2.4 -- concept scope
# ---------------------------------------------------------------------------
# 6 of the 15 concepts in data/project_concepts.json, chosen to preserve the
# five-category entanglement design rather than an arbitrary subset:
#   - Poison, Uranium       -- general-concept / specific-instance pair
#   - Homo Sapiens           -- maximal-entanglement stress test
#   - Golf                   -- clean low-entanglement control; also one of
#                               the concepts PISCES's own paper reports
#                               results for, giving a direct comparison point
#   - Gun, Mass Shooting     -- intersectional pair
REDUCED_CONCEPTS = ["Poison", "Uranium", "Homo Sapiens", "Golf", "Gun", "Mass Shooting"]
