"""
Track A orchestration entrypoint.

For each concept: builds the vocabulary-projection lookup (vocab_projection.py,
step 2 of the data flow in this directory's README), derives seed tokens and
neg_toks (seed_tokens.py, steps 3-4), then runs PISCES's own unmodified
`search_features` + `filter_features_by_effect_and_activations` +
`filter_features_by_mmlu` (steps 5-6) and writes
`artifacts/features/<concept>.parquet` per `schema.FeatureRecord` -- both the
full candidate pool (`selected=False`) and the final filtered set
(`selected=True`), matching the shared schema Tracks B/C read.

Sanity-check path (do this before running all 15 concepts, per project plan):

    python discover.py --concept "Harry Potter"

then compare the resulting `selected=True` rows against the five hand-picked
features hardcoded in pisces_ref/erasing_harry_potter.ipynb:
    Feature(1, 8965, True), Feature(1, 13394, False), Feature(4, 661, True),
    Feature(20, 11104, True), Feature(20, 14668, False)

CONSTRUCTION-ONLY CHECKPOINT: this module is written and unit-testable (see
test_vocab_projection.py / test_seed_tokens.py), but per the project's current
phase should NOT be executed against the real model / real concept data
without explicit sign-off -- the seed-token and neg_toks methodological
choices in seed_tokens.py need a team review pass first. See this directory's
README.

Requires a CUDA GPU, PISCES's deps (see requirements.txt), and network/hub
access to download Gemma-2-2B-it and the GemmaScope MLP SAEs.
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
from seed_tokens import derive_seed_tokens_for_concept, get_neg_toks  # noqa: E402
from vocab_projection import build_all_layer_lookups  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts" / "features"
CVS_PATH = ROOT / "data" / "cvs.json"


def load_cvs() -> list[dict]:
    with open(CVS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_concept_data(cvs: list[dict], concept: str) -> dict:
    for row in cvs:
        if row["Concept"] == concept:
            return row
    raise KeyError(f"Concept {concept!r} not found in {CVS_PATH}")


def discover_concept(model, cvs: list[dict], concept: str, layers=None) -> pd.DataFrame:
    concept_data = get_concept_data(cvs, concept)

    def is_single_token(tok: str) -> bool:
        return len(model.to_tokens(tok, prepend_bos=False)[0]) == 1

    seed_tokens = derive_seed_tokens_for_concept(concept, cvs, is_single_token)
    if not seed_tokens:
        raise ValueError(
            f"No single-token seed words could be extracted for {concept!r} from "
            "its wikipedia_content -- check seed_tokens.extract_seed_tokens / "
            "consider raising top_n."
        )
    neg_toks = get_neg_toks(is_single_token)

    saes_by_layer: dict[int, object] = {}
    lls = build_all_layer_lookups(model, MODEL_NAME, layers=layers, saes_out=saes_by_layer)

    candidates = search_features(model, lls, seed_tokens, minmatch=1)
    print(f"[{concept}] seed_tokens={seed_tokens} neg_toks={neg_toks}")
    print(f"[{concept}] {len(candidates)} candidate features from vocab-projection search")

    forget_text = concept_data["wikipedia_content"]
    signs = get_mlp_act_signs(model, seed_tokens, forget_text.splitlines()[:1000])

    effect_filtered, (pos_effects, neg_effects, activations) = filter_features_by_effect_and_activations(
        model, candidates, forget_text, signs, seed_tokens, neg_toks, filter_by_act=True
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
        help="Concept name (repeatable). Defaults to all 15 natural concepts.",
    )
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Restrict to specific layers (faster iteration).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    concepts = args.concepts or NATURAL_CONCEPTS
    cvs = load_cvs()

    from sae_lens import HookedSAETransformer

    # must be HookedSAETransformer: feature_finder.py's get_feature_effect calls
    # run_with_cache_with_saes, which only exists on this subclass, not plain HookedTransformer.
    model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=args.device)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for concept in concepts:
        try:
            df = discover_concept(model, cvs, concept, layers=args.layers)
        except ValueError as e:
            print(f"[{concept}] SKIPPED: {e}")
            continue

        out_path = ARTIFACTS_DIR / f"{concept.lower().replace(' ', '_')}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[{concept}] wrote {len(df)} candidate features to {out_path}")


if __name__ == "__main__":
    main()
