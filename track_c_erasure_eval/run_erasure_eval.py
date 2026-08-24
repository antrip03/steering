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
import random
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

# evaluate_mmlu's own random.sample() has no fixed seed, so every call --
# across different concepts, different hyperparameter candidates, even
# repeated calls at identical settings -- draws a different random 300
# (or --mmlu-screen-limit) question subset. Observed directly: Golf's
# specificity_mmlu moved 0.307 -> 0.363 -> 0.263 across three otherwise
# identical re-runs, noise large enough to swamp real hyperparameter
# effects. Reseeding immediately before every evaluate_mmlu call (both the
# cheap screening passes and the final scored one) makes every call within
# each limit size draw the SAME question subset, so differences between
# concepts/candidates reflect the edit, not the sample.
MMLU_SEED = 42


def evaluate_mmlu_fixed(model, limit: int, batch_size: int = 4):
    random.seed(MMLU_SEED)
    return evaluate_mmlu(model, True, limit=limit, batch_size=batch_size, evaluation_type=MCQAEvaluations.RANK_BASED, verbose=False)


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


def load_selected_features_df(concept: str) -> pd.DataFrame:
    path = FEATURES_DIR / feature_artifact_filename(concept)
    if not path.exists():
        raise FileNotFoundError(
            f"No Track A output for {concept!r} at {path}. "
            "Run track_a_feature_discovery/discover.py first, or use "
            "--hardcoded-hp for the Harry Potter validation path."
        )
    df = pd.read_parquet(path)
    return df[df["selected"]]


def truncate_features(selected: pd.DataFrame, max_features: int | None) -> list[Feature]:
    """Keep only the top max_features rows by |mass_ratio_or_effect_score|
    (all of them if max_features is None or already <= the count).

    Track A's selection filters are far more permissive than PISCES's own
    validated example (5 hand-picked Harry Potter features) -- observed
    selecting 46-435 features per concept across all 10 completed concepts,
    and editing that many features simultaneously produced degenerate,
    repetition-collapsed generation (confirmed by inspecting real model
    output, not just the aggregate metrics) even at conservative k/value
    settings. Capping to the top-N by effect magnitude keeps the edit closer
    to the validated reference scale without discarding Track A's
    already-completed selection work."""
    if max_features is not None and len(selected) > max_features:
        selected = selected.reindex(selected["mass_ratio_or_effect_score"].abs().sort_values(ascending=False).index).head(max_features)
    return [Feature(layer=int(r.layer), id=int(r.feature_id), neg=bool(r.neg)) for r in selected.itertuples()]


def load_selected_features(concept: str, max_features: int | None = None) -> list[Feature]:
    return truncate_features(load_selected_features_df(concept), max_features)


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
        mmlu_res, _ = evaluate_mmlu_fixed(model, limit=mmlu_limit)

    return ConceptResultRow(
        concept=concept_name,
        efficacy=efficacy_res.score_from_total,
        specificity_simdomain=simdom_res.score_from_total,
        specificity_mmlu=mmlu_res.score_from_total,
    )


def screen_max_features(
    model,
    concept_name: str,
    selected_df: pd.DataFrame,
    pos_toks: list[str],
    candidates: list[int],
    k: float,
    value: float,
    mmlu_screen_limit: int = 60,
) -> int:
    """Cheap per-concept hyperparameter screen: tries each candidate
    max_features value, scoring only with MMLU (no Gemini calls needed, so
    this is fast) at a smaller question count than the final scored eval,
    and returns whichever candidate kept specificity_mmlu healthiest.

    Not PISCES's own find_hps (a 10x10 grid that also re-runs the expensive
    feature-filtering step per combination) -- that's too costly to run per
    concept under deadline pressure. This only screens max_features (the
    axis a manual sweep on Golf showed was most sensitive) at a fixed,
    already-reasonable (k, value), using a cheap proxy metric instead of the
    full efficacy+specificity_simdomain+specificity_mmlu eval."""
    concept_data = load_concept_data(concept_name)
    signs = get_mlp_act_signs(model, pos_toks, concept_data["wikipedia_content"].splitlines()[:1000])

    print(f"[{concept_name}] Screening max_features candidates {candidates} (MMLU-only, limit={mmlu_screen_limit})...", flush=True)
    best_mf, best_score = candidates[0], -1.0
    for mf in candidates:
        features = truncate_features(selected_df, mf)
        concept = Concept(name=concept_name, k=k, value=value, features=features)
        with unlearn_concept(model, concept, linscale=True, signs=signs):
            mmlu_res = evaluate_mmlu_fixed(model, limit=mmlu_screen_limit)[0]
        print(f"[{concept_name}]   max_features={mf}: mmlu_screen={mmlu_res.score_from_total:.3f}", flush=True)
        if mmlu_res.score_from_total > best_score:
            best_mf, best_score = mf, mmlu_res.score_from_total
    print(f"[{concept_name}] Selected max_features={best_mf} (mmlu_screen={best_score:.3f})", flush=True)
    return best_mf


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
        "--screen-features", default=None,
        help="Comma-separated max_features candidates (e.g. '5,10,20') to screen per concept "
             "with a cheap MMLU-only pass (see screen_max_features()) before running the full "
             "Gemini-based eval with whichever candidate scored best. Overrides --max-features "
             "per concept when set.",
    )
    parser.add_argument(
        "--mmlu-screen-limit", type=int, default=60,
        help="MMLU question count for --screen-features's cheap screening pass (smaller than "
             "--mmlu-limit since it only needs to rank candidates, not produce a final score).",
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
            screen_candidates = [int(x) for x in args.screen_features.split(",")] if args.screen_features else None
            for concept in concepts:
                try:
                    selected_df = load_selected_features_df(concept)
                    pos_toks = load_pos_toks(concept)
                except (FileNotFoundError, ValueError) as e:
                    print(f"[{concept}] SKIPPED: {e}")
                    continue

                max_features = args.max_features
                if screen_candidates:
                    max_features = screen_max_features(
                        model, concept, selected_df, pos_toks, screen_candidates,
                        k=args.k, value=args.value, mmlu_screen_limit=args.mmlu_screen_limit,
                    )
                features = truncate_features(selected_df, max_features)

                row = evaluate_concept(model, concept, features, pos_toks, k=args.k, value=args.value, mmlu_limit=args.mmlu_limit)
                write_and_maybe_push(row)


if __name__ == "__main__":
    main()
