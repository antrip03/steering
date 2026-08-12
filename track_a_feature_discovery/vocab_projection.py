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
pipeline (imported elsewhere, not reimplemented here).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

TOP_K = 50  # top/bottom tokens kept per feature; matches the artifact schema

# build_layer_lookup processes SAE features in chunks of this size rather than
# projecting all of them through model.W_U in one matmul -- a single
# [n_features, d_vocab] fp32 tensor is ~15.6GiB for a 16k-width SAE against
# Gemma-2-2B-it's ~256k vocab (16384 * 256000 * 4 bytes), which OOMs on a T4
# (16GB) and is wasteful everywhere else too, since only the top/bottom-k
# per feature are ever kept. 2000 features/chunk -> ~1.9GiB per chunk
# (2000 * 256000 * 4 bytes), comfortable alongside a resident bf16/fp16 model.
FEATURE_CHUNK_SIZE = 2000


@dataclass
class LayerLookup:
    """What `feature_finder.search_features` expects as `lls[layer]`: per-feature
    top (`t`) and bottom (`b`) projected-token lists, ordered most- to least-extreme.
    `search_features` only ever does `set(ll.t[i]).intersection(set(tokens))`, so
    plain lists of token strings are sufficient -- no need to match PISCES's
    original (unknown, since never implemented) lookup type exactly."""

    t: list[list[str]]
    b: list[list[str]]


def build_layer_lookup(model, sae, top_k: int = TOP_K, chunk_size: int = FEATURE_CHUNK_SIZE) -> LayerLookup:
    """Vocabulary-project every decoder direction in `sae` through `model`'s
    unembedding matrix (the technique PISCES's paper calls "vocabulary
    projection") and rank tokens per feature.

    Only needs `sae.W_dec` and `model.W_U`/`model.to_string` -- takes an
    already-loaded SAE rather than loading one itself, so it's testable
    against small synthetic tensors without touching sae_loader or a real
    model (see test_vocab_projection.py).

    Processes features in chunks of `chunk_size` rather than projecting the
    whole SAE through model.W_U in a single matmul -- see FEATURE_CHUNK_SIZE's
    comment for why the unchunked version is a real OOM risk, not a
    theoretical one. Each chunk's full [chunk_size, d_vocab] projection is
    discarded immediately after its top/bottom-k are extracted, so peak memory
    scales with chunk_size, not with the SAE's total feature count. Output is
    identical to the unchunked computation -- topk is applied per-row, so
    chunking rows changes nothing about any individual row's result.
    """
    # sae.W_dec: [n_features, d_model]. PISCES's "gemmascope-mlp" SAEs are
    # trained on hook_mlp_out (the MLP's contribution to the residual stream,
    # d_model-dimensional) -- see pisces_ref/editor.py, which encodes
    # `model.blocks[layer].mlp.W_out @ sae.W_enc[:, feature.id]` where W_out
    # is [d_mlp, d_model], so sae.W_enc/W_dec operate in d_model space, not
    # d_mlp.
    w_u = model.W_U.float()
    n_features = sae.W_dec.shape[0]
    k = min(top_k, w_u.shape[-1])

    top_ids_chunks = []
    bottom_ids_chunks = []
    with torch.no_grad():
        for start in range(0, n_features, chunk_size):
            chunk = sae.W_dec[start:start + chunk_size].float() @ w_u  # [chunk_size, d_vocab]
            _, top_ids = chunk.topk(k, dim=-1)
            _, bottom_ids = chunk.topk(k, dim=-1, largest=False)
            top_ids_chunks.append(top_ids)
            bottom_ids_chunks.append(bottom_ids)
            del chunk

    top_ids = torch.cat(top_ids_chunks, dim=0)
    bottom_ids = torch.cat(bottom_ids_chunks, dim=0)

    t = [[model.to_string(tok_id) for tok_id in row] for row in top_ids]
    b = [[model.to_string(tok_id) for tok_id in row] for row in bottom_ids]

    return LayerLookup(t=t, b=b)


def build_all_layer_lookups(
    model,
    model_name: str,
    layers=None,
    sae_size: str | None = None,
    device: str = "cuda",
    saes_out: dict | None = None,
    chunk_size: int = FEATURE_CHUNK_SIZE,
) -> dict[int, LayerLookup]:
    """Build the full `lls` dict that PISCES's `search_features(model, lls, tokens,
    ...)` expects, one `LayerLookup` per layer.

    If `saes_out` is passed, it's populated with the loaded SAE objects keyed by
    layer, so callers (e.g. discover.py) can reuse them for decoder-direction
    extraction without reloading from disk/hub.

    Imports sae_loader lazily (it pulls in pisces_ref/editor.py -> sae_lens,
    a real dependency only needed once we actually load SAEs) so that
    build_layer_lookup above -- the pure, unit-tested piece -- stays importable
    without sae_lens/transformer_lens installed.
    """
    from sae_loader import load_all_layer_saes

    if layers is None:
        layers = range(model.cfg.n_layers)
    layers = list(layers)

    saes = load_all_layer_saes(model_name, layers, sae_size=sae_size, device=device)
    if saes_out is not None:
        saes_out.update(saes)

    return {layer: build_layer_lookup(model, sae, chunk_size=chunk_size) for layer, sae in saes.items()}
