"""
Unit tests for vocab_projection.py's ranking logic, against small synthetic
tensors. Deliberately does not import torch model weights, a real SAE, or
sae_loader (which needs pisces_ref/editor.py's SAE-hub download path) --
only `build_layer_lookup`, which is pure given `sae.W_dec` and `model.W_U` /
`model.to_string`. No GPU or model download required.
"""
from __future__ import annotations

import torch

from vocab_projection import LayerLookup, build_layer_lookup


class FakeSAE:
    """Minimal stand-in for a loaded sae_lens SAE: only `W_dec` is used by
    build_layer_lookup."""

    def __init__(self, w_dec: torch.Tensor):
        self.W_dec = w_dec


class FakeModel:
    """Minimal stand-in for a HookedTransformer: only `W_U` and `to_string`
    are used by build_layer_lookup. Vocab is just str(i) for token id i, so
    expected rankings are easy to state directly in terms of token ids."""

    def __init__(self, w_u: torch.Tensor):
        self.W_U = w_u

    def to_string(self, token_id) -> str:
        return str(int(token_id))


def test_build_layer_lookup_ranks_top_and_bottom_tokens_correctly():
    # d_model=2, d_vocab=4. Unembedding directions chosen so each vocab token's
    # projection value is easy to reason about by hand.
    w_u = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.0],  # dim 0 contributes +1 to tok0, -1 to tok2
            [0.0, 1.0, 0.0, -1.0],  # dim 1 contributes +1 to tok1, -1 to tok3
        ]
    )
    # One feature whose decoder direction points purely along dim 0 -> should
    # rank tok0 highest, tok2 lowest, tok1/tok3 tied in the middle.
    w_dec = torch.tensor([[1.0, 0.0]])
    sae = FakeSAE(w_dec)
    model = FakeModel(w_u)

    lookup = build_layer_lookup(model, sae, top_k=2)

    assert isinstance(lookup, LayerLookup)
    assert len(lookup.t) == 1
    assert len(lookup.b) == 1

    # top-2 (highest projection) for feature 0: tok0 (+1) beats the tok1/tok3 tie (0)
    assert lookup.t[0][0] == "0"
    # bottom-2 (lowest projection) for feature 0: tok2 (-1) is the clear minimum
    assert lookup.b[0][0] == "2"


def test_build_layer_lookup_multiple_features_independent():
    w_u = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    w_dec = torch.tensor(
        [
            [1.0, 0.0],  # feature 0 -> favors tok0, disfavors tok1
            [0.0, 1.0],  # feature 1 -> favors tok2, disfavors tok3
        ]
    )
    sae = FakeSAE(w_dec)
    model = FakeModel(w_u)

    lookup = build_layer_lookup(model, sae, top_k=1)

    assert lookup.t[0] == ["0"]
    assert lookup.b[0] == ["1"]
    assert lookup.t[1] == ["2"]
    assert lookup.b[1] == ["3"]


def test_build_layer_lookup_top_k_larger_than_vocab_is_clamped():
    # top_k=50 (the real default) with a vocab of only 3 tokens shouldn't crash.
    w_u = torch.eye(3)
    w_dec = torch.tensor([[1.0, 0.0, 0.0]])
    sae = FakeSAE(w_dec)
    model = FakeModel(w_u)

    lookup = build_layer_lookup(model, sae, top_k=50)

    assert len(lookup.t[0]) == 3
    assert len(lookup.b[0]) == 3
