"""
Loads PISCES's pretrained per-layer MLP SAEs.

Matches `pisces_ref/editor.py`'s own `SAEConfig` pattern exactly -- the same
release/sae_id derivation and the same default SAE width ("16k" for Gemma,
"32k" for Llama) that `pisces_ref/feature_finder.py::get_feature_saes` uses.
Factored out from `vocab_projection.py` so loading (this module) and
vocabulary projection (which is the only genuinely new algorithmic piece) are
independently testable.
"""
from __future__ import annotations

import sys
from pathlib import Path

PISCES_REF = Path(__file__).resolve().parent.parent / "pisces_ref"
if str(PISCES_REF) not in sys.path:
    sys.path.insert(0, str(PISCES_REF))

from editor import SAEConfig  # noqa: E402

_DEFAULT_SIZE_BY_FAMILY = {"gemma": "16k", "llama": "32k"}


def default_sae_size(model_name: str) -> str:
    """"16k" for gemma models, "32k" for llama -- mirrors the ternary inline
    in feature_finder.py::get_feature_saes rather than reimplementing it
    differently here."""
    family = "gemma" if "gemma" in model_name else "llama"
    return _DEFAULT_SIZE_BY_FAMILY[family]


def load_layer_sae(
    model_name: str,
    layer: int,
    sae_size: str | None = None,
    device: str = "cuda",
    sae_type: str = "mlp",
):
    """Load one layer's pretrained SAE via SAEConfig (downloads/loads from the
    hub via sae_lens under the hood)."""
    size = sae_size or default_sae_size(model_name)
    return SAEConfig(model_name=model_name, layer=layer, type=sae_type, size=size, device=device).get()


def load_all_layer_saes(
    model_name: str,
    layers,
    sae_size: str | None = None,
    device: str = "cuda",
    sae_type: str = "mlp",
) -> dict[int, object]:
    """Load every requested layer's SAE, keyed by layer number."""
    return {
        layer: load_layer_sae(model_name, layer, sae_size=sae_size, device=device, sae_type=sae_type)
        for layer in layers
    }
