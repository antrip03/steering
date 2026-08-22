"""
Unit tests for discover.py's cascade_filter_candidates plumbing -- added
after a real Kaggle run hit the bare "No changes made to the model in
layer X" AssertionError inside this function (it calls get_feature_effect
directly, bypassing filter_features_by_effect_and_activations, which
already forwarded debug_log_noop_edits -- this was a separate, unwired
call site). No GPU/real SAE needed -- get_feature_effect is mocked to
isolate the plumbing regression, matching pisces_ref/test_feature_finder.py's
approach for the equivalent filter_features_by_mmlu fix.
"""
from __future__ import annotations

import collections
import sys
import types

# discover.py imports feature_finder.py (-> editor.py -> sae_lens) and
# evals.py (transformers/datasets/openai/transformer_lens/google.generativeai)
# at module level -- none of that is needed to test cascade_filter_candidates'
# plumbing, and sae_lens/transformer_lens's own import chain has been
# observed to hang in this environment. Stub both out before importing.
if "sae_lens" not in sys.modules:
    fake_sae_lens = types.ModuleType("sae_lens")
    fake_sae_lens.SAE = type("SAE", (), {"from_pretrained": staticmethod(lambda release, sae_id, device: (object(), {}))})
    sys.modules["sae_lens"] = fake_sae_lens

if "evals" not in sys.modules:
    fake_evals = types.ModuleType("evals")
    for _name in (
        "eval_alpaca", "TransformerLensModel", "GeminiEvaluator", "evaluate_mmlu",
        "evaluate_open_ended", "OpenEndedQuestion",
    ):
        setattr(fake_evals, _name, type(_name, (), {}))
    fake_evals.MCQAEvaluations = type("MCQAEvaluations", (), {"RANK_BASED": "RANK_BASED", "NEXT_TOKEN": "NEXT_TOKEN"})
    sys.modules["evals"] = fake_evals

import discover  # noqa: E402
from editor import Feature  # noqa: E402


class FakeModel:
    def to_single_token(self, tok: str) -> int:
        return hash(tok) % 1000  # arbitrary but stable within a test


def test_cascade_filter_candidates_forwards_debug_log_noop_edits(monkeypatch):
    captured = []

    def fake_get_feature_effect(model, candidates, signs, lines, pos_tok_ids, neg_tok_ids, batch_size=3, debug_log_noop_edits=False):
        captured.append(debug_log_noop_edits)
        # empty per-feature effect dicts are enough for score()'s np.mean guard
        return {(f.layer, f.id): [] for f in candidates}, {(f.layer, f.id): [] for f in candidates}

    monkeypatch.setattr(discover, "get_feature_effect", fake_get_feature_effect)

    model = FakeModel()
    candidates = [Feature(layer=1, id=1, neg=False), Feature(layer=1, id=2, neg=False)]
    forget_text = "line one\nline two\nline three"

    discover.cascade_filter_candidates(
        model, candidates, forget_text, signs=None, seed_tokens=[" golf"], neg_toks=[" the"],
        cascade_batches=1, keep_fraction=0.5, debug_log_noop_edits=True,
    )
    assert captured == [True]

    captured.clear()
    discover.cascade_filter_candidates(
        model, candidates, forget_text, signs=None, seed_tokens=[" golf"], neg_toks=[" the"],
        cascade_batches=1, keep_fraction=0.5,
    )
    assert captured == [False], "default must stay False -- no change to existing behavior for callers that don't opt in"


def test_build_run_tag_distinguishes_every_configuration():
    """Every real run in this investigation so far has been a distinct
    configuration -- original, --reduced, --reduced --enable-cascade,
    --minmatch N (with or without --reduced) -- and out_path used to be
    concept-only, so every one of these silently overwrote the same file.
    This is the fix: each combination must produce a different tag."""
    seen = set()
    configs = [
        (False, None, False),
        (True, None, False),
        (True, None, True),
        (False, 5, False),
        (True, 1, False),
        (True, 1, True),
    ]
    for reduced, minmatch, enable_cascade in configs:
        tag = discover.build_run_tag(reduced, minmatch, enable_cascade)
        assert tag not in seen, f"collision: {(reduced, minmatch, enable_cascade)} produced a tag already used: {tag}"
        seen.add(tag)

    assert discover.build_run_tag(False, None, False) == "original"
    assert discover.build_run_tag(True, None, False) == "reduced"
    assert discover.build_run_tag(True, None, True) == "reduced_cascade"
    assert discover.build_run_tag(True, 5, False) == "reduced_minmatch5"
    # enable_cascade only matters with reduced=True -- must not leak into the original tag
    assert discover.build_run_tag(False, None, True) == "original"


def test_build_layers_slug_is_stable_regardless_of_input_order():
    assert discover.build_layers_slug(None) == "all_layers"
    assert discover.build_layers_slug([6]) == "layers_6"
    assert discover.build_layers_slug([9, 3, 7]) == discover.build_layers_slug([3, 7, 9]) == "layers_3_7_9"


def test_parse_feature_spec():
    f = discover.parse_feature_spec("4:661:1")
    assert (f.layer, f.id, f.neg) == (4, 661, True)
    f = discover.parse_feature_spec("20:14668:0")
    assert (f.layer, f.id, f.neg) == (20, 14668, False)


def test_features_override_skips_search_and_infers_layers(monkeypatch):
    """--features (e.g. to check whether the downstream filters would
    promote a specific, already-known candidate confirmed present via
    --candidates-only) must skip vocab-projection search entirely, and infer
    which layers to build lookups for from the override list itself -- a
    real Harry Potter run's --layers wouldn't necessarily match a
    hand-picked feature's actual layer, and silently excluding it from
    saes_by_layer/lls would KeyError deep in the row-building loop instead
    of doing what was asked."""
    def fail_search_features(*args, **kwargs):
        raise AssertionError("search_features must not be called when features_override is given")
    monkeypatch.setattr(discover, "search_features", fail_search_features)

    captured = {}

    class FakeLookup:
        def __init__(self):
            self.t = collections.defaultdict(lambda: ["tok_a", "tok_b"])
            self.b = collections.defaultdict(lambda: ["tok_c", "tok_d"])

    def fake_build_all_layer_lookups(model, model_name, layers=None, device="cuda", saes_out=None):
        captured["layers"] = layers
        lls = {}
        for layer in layers:
            lls[layer] = FakeLookup()
            if saes_out is not None:
                saes_out[layer] = object()
        return lls

    monkeypatch.setattr(discover, "build_all_layer_lookups", fake_build_all_layer_lookups)
    monkeypatch.setattr(discover, "derive_seed_tokens_for_concept", lambda concept, cvs, is_single_token: [" golf"])
    monkeypatch.setattr(discover, "get_neg_toks", lambda is_single_token: [" the"])

    model = FakeModel()
    cvs = [{"Concept": "Golf", "wikipedia_content": "irrelevant"}]
    features = [Feature(layer=9, id=2, neg=True), Feature(layer=5, id=1, neg=False)]

    df = discover.discover_concept(
        model, cvs, "Golf", layers=[1, 2, 3],  # deliberately wrong/unrelated -- must be ignored
        features_override=features, candidates_only=True,
    )

    assert captured["layers"] == [5, 9], "layers must be inferred (sorted) from features_override, not the passed-in layers="
    assert len(df) == 2
    assert set(zip(df.layer, df.feature_id, df.neg)) == {(9, 2, True), (5, 1, False)}
