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

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PISCES_REF = ROOT / "pisces_ref"
# hub_storage lives under track_a_feature_discovery/ (Track C reuses Track
# A's push-to-hub mechanism as-is rather than duplicating it) -- needed for
# write_and_maybe_push's `from hub_storage import push_run_output` below.
TRACK_A = ROOT / "track_a_feature_discovery"
for p in (ROOT, PISCES_REF, TRACK_A):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import CVS_PATH, ConceptResultRow, MODEL_NAME, NATURAL_CONCEPTS, erasure_result_filename, feature_artifact_filename  # noqa: E402
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


def load_selected_features(concept: str, max_features: int | None = None) -> list[Feature]:
    path = FEATURES_DIR / feature_artifact_filename(concept)
    if not path.exists():
        raise FileNotFoundError(
            f"No Track A output for {concept!r} at {path}. "
            "Run track_a_feature_discovery/discover.py first, or use "
            "--hardcoded-hp for the Harry Potter validation path."
        )
    df = pd.read_parquet(path)
    selected = df[df["selected"]]
    if max_features is not None and len(selected) > max_features:
        # Track A's selection filters are far more permissive than PISCES's
        # own validated example (5 hand-picked Harry Potter features) --
        # observed selecting 46-435 features per concept across all 10
        # completed concepts, and editing that many features simultaneously
        # produced degenerate, repetition-collapsed generation (confirmed by
        # inspecting real model output, not just the aggregate metrics) even
        # at conservative k/value settings. Capping to the top-N by effect
        # magnitude keeps the edit closer to the validated reference scale
        # without discarding Track A's already-completed selection work.
        selected = selected.reindex(selected["mass_ratio_or_effect_score"].abs().sort_values(ascending=False).index).head(max_features)
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
        # batch_size=10 (evaluate_mmlu's default) was observed hitting a genuine
        # (not just fragmentation) CUDA OOM on the L4's 24GB -- "4.69 GiB
        # needed, 1.27 GiB free", not close. A smaller batch lowers the peak
        # memory of each batched forward pass at some cost to eval speed.
        mmlu_res, _ = evaluate_mmlu(model, True, limit=mmlu_limit, batch_size=4, evaluation_type=MCQAEvaluations.RANK_BASED, verbose=False)

    # TEMPORARY diagnostic -- dumps a few raw model answers + Gemini's raw
    # grading response, to tell apart "model output is genuinely garbage
    # after editing" from "output is fine but something in the grading path
    # scores it wrong". Remove once Golf's near-chance-MMLU/zero-specificity
    # result is understood.
    print(f"=== DEBUG: {concept_name} sample efficacy answers ===", flush=True)
    for (q, a), r in list(zip(efficacy_res.qas, efficacy_res.responses))[:3]:
        print(f"  Q: {q!r}\n  MODEL ANSWER: {a!r}\n  GEMINI GRADING RESPONSE: {r!r}\n", flush=True)
    print(f"=== DEBUG: {concept_name} sample specificity_simdomain answers ===", flush=True)
    for (q, a), r in list(zip(simdom_res.qas, simdom_res.responses))[:3]:
        print(f"  Q: {q!r}\n  MODEL ANSWER: {a!r}\n  GEMINI GRADING RESPONSE: {r!r}\n", flush=True)

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
    parser.add_argument(
        "--k", type=float, default=0.4,
        help="Concept.k (PISCES's tau). Default is the notebook's hardcoded Harry Potter "
             "value -- see evaluate_concept()'s docstring on why this isn't per-concept tuned yet.",
    )
    parser.add_argument(
        "--value", type=float, default=36,
        help="Concept.value (PISCES's mu). Same caveat as --k.",
    )
    parser.add_argument(
        "--max-features", type=int, default=None,
        help="Cap the number of Track A-selected features actually edited, keeping the "
             "top-N by |mass_ratio_or_effect_score|. Track A's selection filters are far "
             "more permissive than PISCES's own validated Harry Potter example (5 features) "
             "-- observed selecting 46-435 features per concept, which produced "
             "repetition-collapsed, incoherent generation even at conservative k/value. "
             "Unset (None) uses all selected features, the prior behavior.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload each concept's result parquet to hub_storage.HF_REPO_ID after writing it "
             "locally, same mechanism Track A uses. Requires an HF token with repo.write scope.",
    )
    args = parser.parse_args()

    from sae_lens import HookedSAETransformer

    # must be HookedSAETransformer: feature_finder.py's get_feature_effect calls
    # run_with_cache_with_saes, which only exists on this subclass, not plain HookedTransformer.
    # dtype=bfloat16 (default is float32) to roughly halve the model's memory footprint --
    # note this does NOT affect SAE memory: SAE.from_pretrained (pisces_ref/editor.py's
    # SAEConfig.get()) has no dtype override, SAEs load at whatever precision their pretrained
    # checkpoint ships (float32 for GemmaScope releases). editor.py's get_hswaps_full_signed/
    # get_hswaps_full used to do raw matmuls directly between model.blocks[i].mlp.W_out and
    # sae.W_enc/sae.decode(...), which requires matching dtypes -- bf16 vs the SAE's fp32 raised
    # "addmv input tensors must have the same dtype". Fixed at the source (both functions now
    # bridge W_out to a temporary fp32 copy for the SAE-facing ops, cast back to the model's
    # native dtype before the weight swap), so bf16 is now safe on any device, not just CUDA.
    # bf16 also meaningfully helps CPU-only runs: it roughly halves both the model's static
    # footprint and the transient peak memory of each forward pass, directly relevant on
    # memory-constrained hosts -- CPU bf16 *speed* isn't guaranteed to improve (no dedicated
    # bf16 kernels on most consumer CPUs), only memory headroom.
    model_dtype = torch.bfloat16
    # See discover.py's identical comment: from_pretrained_no_processing skips several
    # weight-folding/centering passes that plausibly cause a large transient memory
    # spike during loading; safe here since nothing in this pipeline touches attention
    # weights or relies on LayerNorm folding/centering.
    model = HookedSAETransformer.from_pretrained_no_processing(MODEL_NAME, device=args.device, dtype=model_dtype)
    # See discover.py's identical comment: nothing in this pipeline trains/backprops, but
    # none of pisces_ref's forward-pass call sites wrap model(...) in torch.no_grad(), so
    # every forward pass retains a full backward-computation graph unless we force it off
    # here -- observed in practice as free RAM collapsing from ~22GB to ~300MB within a
    # few hundred passes during the Track A discovery run.
    model.requires_grad_(False)

    out_dir = ROOT / "artifacts" / "erasure_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_and_maybe_push(row: ConceptResultRow) -> None:
        # Written and pushed immediately after each concept, not batched
        # until the whole run finishes -- a crash on concept 5 of 10 (Gemini
        # API hiccup, GPU issue, anything) must not lose concepts 1-4's
        # already-computed results, and separate invocations (different
        # concepts on different machines) must not clobber each other's
        # output the way a single shared erasure_eval_results.parquet would.
        out_path = out_dir / erasure_result_filename(row.concept)
        pd.DataFrame([row.to_dict()]).to_parquet(out_path, index=False)
        print(f"[{row.concept}] wrote {out_path}")
        if args.push_to_hub:
            from hub_storage import push_run_output
            url = push_run_output(out_path)
            print(f"[{row.concept}] pushed to {url}")

    with torch.no_grad():
        if args.hardcoded_hp:
            pos_toks = load_pos_toks("Harry Potter")
            row = evaluate_concept(model, "Harry Potter", HARDCODED_HP_FEATURES, pos_toks, k=args.k, value=args.value, mmlu_limit=args.mmlu_limit)
            write_and_maybe_push(row)
        else:
            concepts = args.concepts or NATURAL_CONCEPTS
            for concept in concepts:
                try:
                    features = load_selected_features(concept, max_features=args.max_features)
                    pos_toks = load_pos_toks(concept)
                except (FileNotFoundError, ValueError) as e:
                    print(f"[{concept}] SKIPPED: {e}")
                    continue
                row = evaluate_concept(model, concept, features, pos_toks, k=args.k, value=args.value, mmlu_limit=args.mmlu_limit)
                write_and_maybe_push(row)


if __name__ == "__main__":
    main()
