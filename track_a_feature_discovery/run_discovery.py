"""
CLI: run vocabulary-projection feature discovery for one or more of the 15
natural concepts and write artifacts/features/<concept>.parquet per
schema.FeatureRecord.

Sanity-check path (do this before running all 15 concepts, per project plan):

    python run_discovery.py --concept "Harry Potter"

then compare the resulting `selected=True` rows against the five hand-picked
features hardcoded in pisces_ref/erasing_harry_potter.ipynb:
    Feature(1, 8965, True), Feature(1, 13394, False), Feature(4, 661, True),
    Feature(20, 11104, True), Feature(20, 14668, False)

Requires a CUDA GPU, PISCES's deps (see requirements.txt), and per-concept
token lists filled in in concept_tokens.json (only Harry Potter's are seeded --
see that file's _comment for why the other 14 are left null on purpose).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
PISCES_REF = ROOT / "pisces_ref"
for p in (ROOT, PISCES_REF, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import FeatureRecord, MODEL_NAME, NATURAL_CONCEPTS  # noqa: E402
from editor import get_mlp_act_signs  # noqa: E402
from feature_finder import (  # noqa: E402
    filter_features_by_effect_and_activations,
    filter_features_by_mmlu,
    search_features,
)
from vocab_projection import build_all_layer_lookups  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts" / "features"
CVS_PATH = ROOT / "data" / "cvs.json"
TOKENS_PATH = Path(__file__).resolve().parent / "concept_tokens.json"


def load_concept_data(concept: str) -> dict:
    with open(CVS_PATH, encoding="utf-8") as f:
        cvs = json.load(f)
    for row in cvs:
        if row["Concept"] == concept:
            return row
    raise KeyError(f"Concept {concept!r} not found in {CVS_PATH}")


def load_concept_tokens(concept: str) -> dict:
    with open(TOKENS_PATH, encoding="utf-8") as f:
        tokens = json.load(f)
    entry = tokens.get(concept)
    if entry is None or entry.get("search_tokens") is None:
        raise ValueError(
            f"No search_tokens configured for {concept!r} in {TOKENS_PATH}. "
            "This concept's token lists need to be curated before it can run "
            "(see concept_tokens.json's _comment)."
        )
    return entry


def discover_concept(model, concept: str, layers=None) -> pd.DataFrame:
    concept_data = load_concept_data(concept)
    concept_tokens = load_concept_tokens(concept)

    search_tokens = concept_tokens["search_tokens"]
    pos_toks = concept_tokens["pos_toks"]
    neg_toks = concept_tokens.get("neg_toks")
    if not neg_toks:
        raise ValueError(
            f"neg_toks not set for {concept!r}; required by "
            "filter_features_by_effect_and_activations (near-domain contrastive "
            "tokens whose probability should not drop on erasure)."
        )

    saes_by_layer: dict[int, object] = {}
    lls = build_all_layer_lookups(model, MODEL_NAME, layers=layers, saes_out=saes_by_layer)

    candidates = search_features(model, lls, search_tokens, minmatch=1)
    print(f"[{concept}] {len(candidates)} candidate features from vocab-projection search")

    forget_text = concept_data["wikipedia_content"]
    signs = get_mlp_act_signs(model, pos_toks, forget_text.splitlines()[:1000])

    effect_filtered, (pos_effects, neg_effects, activations) = filter_features_by_effect_and_activations(
        model, candidates, forget_text, signs, pos_toks, neg_toks, filter_by_act=True
    )
    selected = filter_features_by_mmlu(model, effect_filtered, signs)
    selected_keys = {(f.layer, f.id, f.neg) for f in selected}
    print(f"[{concept}] {len(selected)} features survived filtering (selected=True)")

    rows = []
    for feature in candidates:
        sae = saes_by_layer[feature.layer]
        decoder_direction = sae.W_dec[feature.id].detach().float().cpu().tolist()
        ll = lls[feature.layer]
        top_tokens = ll.t[feature.id]
        bottom_tokens = ll.b[feature.id]

        pos_effect_vals = pos_effects.get((feature.layer, feature.id))
        effect_score = float(sum(pos_effect_vals) / len(pos_effect_vals)) if pos_effect_vals else None

        rows.append(
            FeatureRecord(
                concept=concept,
                layer=feature.layer,
                feature_id=feature.id,
                neg=feature.neg,
                decoder_direction=decoder_direction,
                top_tokens=top_tokens,
                bottom_tokens=bottom_tokens,
                mass_ratio_or_effect_score=effect_score,
                selected=(feature.layer, feature.id, feature.neg) in selected_keys,
            ).to_dict()
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concept",
        action="append",
        dest="concepts",
        help="Concept name (repeatable). Defaults to all 15 natural concepts with configured tokens.",
    )
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Restrict to specific layers (faster iteration).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    concepts = args.concepts or NATURAL_CONCEPTS

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=args.device)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for concept in concepts:
        try:
            df = discover_concept(model, concept, layers=args.layers)
        except ValueError as e:
            print(f"[{concept}] SKIPPED: {e}")
            continue

        out_path = ARTIFACTS_DIR / f"{concept.lower().replace(' ', '_')}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[{concept}] wrote {len(df)} candidate features to {out_path}")


if __name__ == "__main__":
    main()
