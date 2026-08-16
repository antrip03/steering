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
