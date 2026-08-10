"""
Track C: runs PISCES's own (unmodified) `unlearn_concept` context manager per
concept and evaluates efficacy/specificity with PISCES's own (unmodified)
evals.py functions. Produces the `efficacy` / `specificity_*` columns of the
combined results table (schema.ConceptResultRow).

Can start immediately and independently of Track A: first validate against
the five hardcoded Harry Potter features from erasing_harry_potter.ipynb
(--hardcoded-hp), then switch to reading Track A's selected features once
those are available.
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
for p in (ROOT, PISCES_REF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import CVS_PATH, ConceptResultRow, MODEL_NAME, NATURAL_CONCEPTS  # noqa: E402
from editor import Concept, Feature, get_mlp_act_signs, unlearn_concept  # noqa: E402
from evals import (  # noqa: E402
    GeminiEvaluator,
    MCQAEvaluations,
    OpenEndedQuestion,
    TransformerLensModel,
    evaluate_mmlu,
    evaluate_open_ended,
)

FEATURES_DIR = ROOT / "artifacts" / "features"
TOKENS_PATH = ROOT / "track_a_feature_discovery" / "concept_tokens.json"

# The five features hand-picked by PISCES's authors for Harry Potter, hardcoded
# in pisces_ref/erasing_harry_potter.ipynb -- used to validate this track
# end-to-end before Track A produces real selected-feature sets.
HARDCODED_HP_FEATURES = [
    Feature(1, 8965, True),
    Feature(1, 13394, False),
    Feature(4, 661, True),
    Feature(20, 11104, True),
    Feature(20, 14668, False),
]


def load_concept_data(concept: str) -> dict:
    with open(CVS_PATH, encoding="utf-8") as f:
        cvs = json.load(f)
    for row in cvs:
        if row["Concept"] == concept:
            return row
    raise KeyError(f"Concept {concept!r} not found in {CVS_PATH}")


def load_selected_features(concept: str) -> list[Feature]:
    path = FEATURES_DIR / f"{concept.lower().replace(' ', '_')}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No Track A output for {concept!r} at {path}. "
            "Run track_a_feature_discovery/discover.py first, or use "
            "--hardcoded-hp for the Harry Potter validation path."
        )
    df = pd.read_parquet(path)
    selected = df[df["selected"]]
    return [Feature(layer=int(r.layer), id=int(r.feature_id), neg=bool(r.neg)) for r in selected.itertuples()]


def load_pos_toks(concept: str) -> list[str]:
    with open(TOKENS_PATH, encoding="utf-8") as f:
        tokens = json.load(f)
    entry = tokens.get(concept) or {}
    pos_toks = entry.get("pos_toks")
    if not pos_toks:
        raise ValueError(f"No pos_toks configured for {concept!r} in {TOKENS_PATH}.")
    return pos_toks


def evaluate_concept(
    model,
    concept_name: str,
    features: list[Feature],
    pos_toks: list[str],
    k: float = 0.4,
    value: float = 36,
    mmlu_limit: int = 300,
) -> ConceptResultRow:
    """k/value default to the notebook's hardcoded Harry Potter hyperparameters
    (PISCES calls these tau/mu). PISCES's own per-concept hyperparameter search
    (feature_finder.py::find_hps) tunes these properly; wiring that in is a
    follow-up, not required for the efficacy/specificity baseline this project
    needs."""
    concept_data = load_concept_data(concept_name)
    concept = Concept(name=concept_name, k=k, value=value, features=features)

    wrapped = TransformerLensModel(model, it=True)
    evaluator = GeminiEvaluator()  # requires GEMINI_API_KEY env var

    signs = get_mlp_act_signs(model, pos_toks, concept_data["wikipedia_content"].splitlines()[:1000])

    qa_test = [OpenEndedQuestion(question=x["q"], answer=x["a"]) for x in concept_data["QA_test"]]
    simdom_test = [OpenEndedQuestion(question=x["q"], answer=x["a"]) for x in concept_data["SimdomQA_test"]]

    with unlearn_concept(model, concept, linscale=True, signs=signs):
        efficacy_res = evaluate_open_ended(wrapped, evaluator, qa_test, verbose=False)
        simdom_res = evaluate_open_ended(wrapped, evaluator, simdom_test, verbose=False)
        mmlu_res, _ = evaluate_mmlu(model, True, limit=mmlu_limit, evaluation_type=MCQAEvaluations.RANK_BASED, verbose=False)

    return ConceptResultRow(
        concept=concept_name,
        efficacy=efficacy_res.score_from_total,
        specificity_simdomain=simdom_res.score_from_total,
        specificity_mmlu=mmlu_res.score_from_total,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", action="append", dest="concepts", help="Concept name (repeatable). Defaults to all 15.")
    parser.add_argument(
        "--hardcoded-hp",
        action="store_true",
        help="Validate against the 5 hand-picked Harry Potter features instead of reading Track A output.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mmlu-limit", type=int, default=300)
    args = parser.parse_args()

    from sae_lens import HookedSAETransformer

    # must be HookedSAETransformer: feature_finder.py's get_feature_effect calls
    # run_with_cache_with_saes, which only exists on this subclass, not plain HookedTransformer.
    # dtype=bfloat16 (default is float32) to roughly halve the model's memory footprint on
    # memory-constrained hosts -- note this does NOT affect SAE memory: SAE.from_pretrained
    # (pisces_ref/editor.py's SAEConfig.get()) has no dtype override, SAEs load at whatever
    # precision their pretrained checkpoint ships (float32 for GemmaScope releases).
    model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=args.device, dtype=torch.bfloat16)

    rows = []
    if args.hardcoded_hp:
        pos_toks = load_pos_toks("Harry Potter")
        rows.append(evaluate_concept(model, "Harry Potter", HARDCODED_HP_FEATURES, pos_toks, mmlu_limit=args.mmlu_limit))
    else:
        concepts = args.concepts or NATURAL_CONCEPTS
        for concept in concepts:
            try:
                features = load_selected_features(concept)
                pos_toks = load_pos_toks(concept)
            except (FileNotFoundError, ValueError) as e:
                print(f"[{concept}] SKIPPED: {e}")
                continue
            rows.append(evaluate_concept(model, concept, features, pos_toks, mmlu_limit=args.mmlu_limit))

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "erasure_eval_results.parquet"
    pd.DataFrame([r.to_dict() for r in rows]).to_parquet(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
