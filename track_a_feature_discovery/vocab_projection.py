"""
Reconstructs PISCES's missing vocabulary-projection feature-discovery step.

pisces_ref/feature_finder.py::search_features expects a precomputed lookup table
(`lls`, with top/bottom vocabulary-projected tokens per SAE feature) as an INPUT --
but nothing in the PISCES repo actually builds that lookup. The one public demo
(erasing_harry_potter.ipynb) only shows five hand-picked features with no live
discovery step. This module builds `lls` for real: for each SAE layer, project
every decoder direction through the model's unembedding matrix, rank vocabulary
tokens by projected logit, and keep the top/bottom-K per feature. The result plugs
directly into PISCES's own unmodified `search_features` / `filter_features_by_*`
pipeline (imported below, not reimplemented).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PISCES_REF = Path(__file__).resolve().parent.parent / "pisces_ref"
if str(PISCES_REF) not in sys.path:
    sys.path.insert(0, str(PISCES_REF))

from editor import SAEConfig  # noqa: E402

TOP_K = 50  # top/bottom tokens kept per feature; matches the artifact schema


@dataclass
class LayerLookup:
    """What `feature_finder.search_features` expects as `lls[layer]`: per-feature
    top (`t`) and bottom (`b`) projected-token lists, ordered most- to least-extreme.
    `search_features` only ever does `set(ll.t[i]).intersection(set(tokens))`, so
    plain lists of token strings are sufficient -- no need to match PISCES's
    original (unknown, since never implemented) lookup type exactly."""

    t: list[list[str]]
    b: list[list[str]]


def build_layer_lookup(model, sae, top_k: int = TOP_K) -> LayerLookup:
    """Vocabulary-project every decoder direction in `sae` through `model`'s
    unembedding matrix (the technique PISCES's paper calls "vocabulary
    projection") and rank tokens per feature."""
    with torch.no_grad():
        # sae.W_dec: [n_features, d_model]. PISCES's "gemmascope-mlp" SAEs are
        # trained on hook_mlp_out (the MLP's contribution to the residual stream,
        # d_model-dimensional) -- see pisces_ref/editor.py, which encodes
        # `model.blocks[layer].mlp.W_out @ sae.W_enc[:, feature.id]` where W_out
        # is [d_mlp, d_model], so sae.W_enc/W_dec operate in d_model space, not
        # d_mlp.
        projected = sae.W_dec.float() @ model.W_U.float()  # [n_features, d_vocab]

        _, top_ids = projected.topk(top_k, dim=-1)
        _, bottom_ids = projected.topk(top_k, dim=-1, largest=False)

    t = [[model.to_string(tok_id) for tok_id in row] for row in top_ids]
    b = [[model.to_string(tok_id) for tok_id in row] for row in bottom_ids]

    return LayerLookup(t=t, b=b)


def build_all_layer_lookups(
    model,
    model_name: str,
    layers=None,
    sae_size: str = "16k",
    saes_out: dict | None = None,
) -> dict[int, LayerLookup]:
    """Build the full `lls` dict that PISCES's `search_features(model, lls, tokens,
    ...)` expects, one `LayerLookup` per layer.

    If `saes_out` is passed, it's populated with the loaded SAE objects keyed by
    layer, so callers (e.g. run_discovery.py) can reuse them for decoder-direction
    extraction without reloading from disk/hub.
    """
    if layers is None:
        layers = range(model.cfg.n_layers)

    lls = {}
    for layer in layers:
        sae = SAEConfig(model_name=model_name, layer=layer, type="mlp", size=sae_size).get()
        if saes_out is not None:
            saes_out[layer] = sae
        lls[layer] = build_layer_lookup(model, sae)

    return lls
