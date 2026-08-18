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

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PISCES_REF = ROOT / "pisces_ref"
for p in (ROOT, PISCES_REF, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import CVS_PATH, FeatureRecord, MODEL_NAME, NATURAL_CONCEPTS  # noqa: E402
from editor import get_mlp_act_signs  # noqa: E402
from feature_finder import (  # noqa: E402
    filter_features_by_effect_and_activations,
    filter_features_by_mmlu,
    get_feature_effect,
    search_features,
)
from seed_tokens import derive_seed_tokens_for_concept, get_neg_toks  # noqa: E402
from vocab_projection import build_all_layer_lookups  # noqa: E402
from reductions import (  # noqa: E402
    CASCADE_KEEP_FRACTION,
    CASCADE_PREFILTER_BATCHES,
    EARLY_EXIT_AFTER_BATCHES,
    EARLY_EXIT_MARGIN,
    MIDDLE_LAYERS,
    REDUCED_CONCEPTS,
    VOCABPROJ_MINMATCH,
    evenly_spaced_subsample_lines,
)

ARTIFACTS_DIR = ROOT / "artifacts" / "features"


def load_cvs() -> list[dict]:
    with open(CVS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_concept_data(cvs: list[dict], concept: str) -> dict:
    for row in cvs:
        if row["Concept"] == concept:
            return row
    raise KeyError(f"Concept {concept!r} not found in {CVS_PATH}")


def cascade_filter_candidates(
    model, candidates, forget_text, signs, seed_tokens, neg_toks, cascade_batches, keep_fraction,
    batch_size=3, debug_log_noop_edits: bool = False,
):
    """Cheap pre-pass on a small (evenly-spaced) subset of batches, used to
    drop the weakest half of candidates before the full effect measurement
    runs on the survivors. Uses the same underlying get_feature_effect
    PISCES's own filter_features_by_effect_and_activations calls internally,
    just on fewer batches and without the final threshold decision.

    Unlike early-exit (see pisces_ref/feature_finder.py::get_feature_effect),
    this is a HEURISTIC speed-up, not a provably-exact one -- a candidate
    that's weak on the cascade subset but happens to be strong specifically
    on the batches the cascade subset skipped could be dropped here even
    though the unreduced run would have kept it. Step 2.5 validates this
    empirically; if it changes the selected FC, don't silently keep it.

    debug_log_noop_edits must be threaded through explicitly here: this
    function calls get_feature_effect DIRECTLY (bypassing
    filter_features_by_effect_and_activations, which already forwards the
    flag), so it's a separate call site from the one get_feature_effect's
    own no-op protection was originally wired into -- a real Kaggle run hit
    exactly this gap (bare "No changes made to the model" crash here, one
    candidate in, on --reduced's very first stage) before this was added."""
    lines = forget_text.splitlines()
    cascade_lines = evenly_spaced_subsample_lines(lines, cascade_batches)

    pos_tok_ids = [model.to_single_token(tok) for tok in seed_tokens]
    neg_tok_ids = [model.to_single_token(tok) for tok in neg_toks]

    pos_effects, neg_effects = get_feature_effect(
        model, candidates, signs, cascade_lines, pos_tok_ids, neg_tok_ids, batch_size=batch_size,
        debug_log_noop_edits=debug_log_noop_edits,
    )

    def score(feature):
        key = (feature.layer, feature.id)
        pos_effect = float(np.mean(pos_effects[key])) if pos_effects[key] else 0.0
        neg_effect = float(np.mean(neg_effects[key])) if neg_effects[key] else 0.0
        # both selection criteria (pos_effect > 0, neg_effect < -2) put on the
        # same "higher is more likely to survive" 0-centered scale
        return max(pos_effect, -neg_effect / 2)

    ranked = sorted(candidates, key=score, reverse=True)
    keep_n = max(1, int(len(ranked) * keep_fraction))
    survivors = ranked[:keep_n]
    print(f"  cascade prefilter: {len(candidates)} -> {len(survivors)} candidates ({cascade_batches} batches, keep_fraction={keep_fraction})")
    return survivors


def discover_concept(
    model, cvs: list[dict], concept: str, layers=None, device: str = "cuda",
    reduced: bool = False, debug_log_noop_edits: bool = False, minmatch_override: int | None = None,
    disable_cascade: bool = False,
) -> pd.DataFrame:
    """reduced=True currently applies cascade prefiltering and early-exit
    (see reductions.py for each one's rationale). Two of the original four
    Step 2 reductions have been tried and dropped after real-run validation
    on Golf showed they change the selected FC, not just its speed: VocabProj
    minmatch tightening (minmatch=5 collapsed the candidate pool to zero
    with no viable intermediate value -- see reductions.py's
    VOCABPROJ_MINMATCH comment), and the reduced+evenly-spaced
    effect-measurement corpus (two identical 85-batch original-settings runs
    matched exactly, proving the divergence against a 20-batch run was real,
    not GPU noise -- see reductions.py's EFFECT_MEASUREMENT_BATCHES
    comment). Early-exit stays because it's provably exact, not a guess --
    see get_feature_effect's docstring. Cascade's own validation is still in
    progress (see disable_cascade below and README.md's Step 2.5 writeup).
    reduced=False (default) preserves the exact original, unrestricted
    behavior -- this flag exists specifically so the same function can be
    called both ways for Step 2.5's validation diff.

    debug_log_noop_edits=True (default False) enables pisces_ref/editor.py's
    replace_mlp_rows diagnostic instead of letting a no-op edit crash with
    "No changes made to the model in layer X" -- see
    track_a_feature_discovery/README.md's investigation section for what it
    distinguishes and why.

    minmatch_override, if given, overrides VOCABPROJ_MINMATCH's value
    directly (currently 1, same as the default) -- kept as a standing way to
    re-test minmatch on a per-run basis (e.g. against a different concept)
    without needing another code change, after minmatch_sweep.py showed
    Golf's collapse was a hard cliff (92 candidates at 1, zero at every
    value 2-5), not a gradual one.

    disable_cascade, if True (only meaningful with reduced=True), skips the
    cascade prefilter step entirely -- ALL candidates go straight to effect
    measurement, instead of only the top CASCADE_KEEP_FRACTION by cascade's
    cheap heuristic score. Originally built to isolate the (since-dropped)
    reduced effect-measurement corpus from cascade specifically -- now that
    the corpus reduction is gone, reduced=True + disable_cascade=True is
    functionally just original settings plus early-exit (provably lossless),
    so it should reproduce the original FC almost exactly and mainly serves
    as a sanity check on that claim. The more informative test now that the
    corpus confound is removed is a plain --reduced run (cascade +
    early-exit, full corpus): it isolates cascade's own effect on the FC
    cleanly, which the original 30/69-match result couldn't, since it had
    both reductions active at once."""
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
    lls = build_all_layer_lookups(model, MODEL_NAME, layers=layers, device=device, saes_out=saes_by_layer)

    if minmatch_override is not None:
        minmatch = minmatch_override
    else:
        minmatch = VOCABPROJ_MINMATCH if reduced else 1
    candidates = search_features(model, lls, seed_tokens, minmatch=minmatch, layers=layers)
    print(f"[{concept}] seed_tokens={seed_tokens} neg_toks={neg_toks} minmatch={minmatch}")
    print(f"[{concept}] {len(candidates)} candidate features from vocab-projection search")

    forget_text = concept_data["wikipedia_content"]
    signs = get_mlp_act_signs(model, seed_tokens, forget_text.splitlines()[:1000])

    concept_slug = concept.lower().replace(" ", "_")
    checkpoint_dir = ARTIFACTS_DIR.parent / "checkpoints" / concept_slug
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    effect_candidates = candidates
    effect_text = forget_text
    early_exit_after_batches = None
    if reduced:
        if disable_cascade:
            print(f"  cascade prefilter: SKIPPED (--disable-cascade) -- all {len(candidates)} candidates go to effect measurement")
        else:
            effect_candidates = cascade_filter_candidates(
                model, candidates, forget_text, signs, seed_tokens, neg_toks,
                CASCADE_PREFILTER_BATCHES, CASCADE_KEEP_FRACTION,
                debug_log_noop_edits=debug_log_noop_edits,
            )
        # Corpus-size reduction (EFFECT_MEASUREMENT_BATCHES) dropped -- see
        # reductions.py's comment: real-run validation showed it changes the
        # selected FC, not just its speed. effect_text stays the full
        # forget_text regardless of `reduced`. early-exit stays on: it's
        # provably exact (see get_feature_effect's docstring), so it's free
        # speedup with no accuracy cost, unlike the corpus reduction was.
        early_exit_after_batches = EARLY_EXIT_AFTER_BATCHES
        print(f"  effect measurement: full corpus ({len(forget_text.splitlines())} lines, corpus reduction "
              f"dropped), early_exit_after_batches={early_exit_after_batches}")

    effect_filtered, (pos_effects, neg_effects, activations) = filter_features_by_effect_and_activations(
        model, effect_candidates, effect_text, signs, seed_tokens, neg_toks, filter_by_act=True,
        checkpoint_dir=str(checkpoint_dir),
        early_exit_after_batches=early_exit_after_batches, early_exit_margin=EARLY_EXIT_MARGIN,
        debug_log_noop_edits=debug_log_noop_edits,
    )
    selected = filter_features_by_mmlu(
        model, effect_filtered, signs,
        checkpoint_path=str(checkpoint_dir / "mmlu.ckpt"),
        debug_log_noop_edits=debug_log_noop_edits,
    )
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
    parser.add_argument(
        "--reduced",
        action="store_true",
        help=(
            "Apply the remaining Step 2 runtime reductions (see reductions.py): "
            "cascade prefiltering and early-exit. VocabProj minmatch tightening "
            "and the reduced effect-measurement corpus were both tried and dropped "
            "after real-run validation showed they change the selected FC, not "
            "just its speed (see VOCABPROJ_MINMATCH / EFFECT_MEASUREMENT_BATCHES "
            "comments in reductions.py) -- effect measurement always runs on the "
            "full corpus now, --reduced or not. Also changes the --concept "
            "and --layers defaults (not overrides -- pass either explicitly to "
            "override) to REDUCED_CONCEPTS / MIDDLE_LAYERS. Omit for the exact "
            "original, unrestricted behavior."
        ),
    )
    parser.add_argument(
        "--debug-log-noop-edits",
        action="store_true",
        help=(
            "If a layer edit turns out to be a no-op, log which of two causes it "
            "was (empty switch list vs. a real edit that collapsed to identical "
            "storage) and continue, instead of crashing with pisces_ref/editor.py's "
            "'No changes made to the model in layer X' assertion. See "
            "track_a_feature_discovery/README.md's investigation section."
        ),
    )
    parser.add_argument(
        "--minmatch",
        type=int,
        default=None,
        help=(
            "Override VocabProj's minmatch directly (default is 1, same as original "
            "settings -- the tightened value of 5 was tried and dropped, see "
            "reductions.py). Kept as a standing way to re-test minmatch on a "
            "per-run/per-concept basis without another code change."
        ),
    )
    parser.add_argument(
        "--disable-cascade",
        action="store_true",
        help=(
            "Only meaningful with --reduced: skip the cascade prefilter step entirely "
            "(all candidates go straight to effect measurement, instead of only the top "
            "CASCADE_KEEP_FRACTION by cascade's heuristic score). With the corpus-size "
            "reduction dropped, --reduced --disable-cascade is now essentially original "
            "settings plus early-exit (provably lossless) -- mainly a sanity check. See "
            "reductions.py / README.md's Step 2.5 writeup for the full context."
        ),
    )
    args = parser.parse_args()

    if args.reduced:
        concepts = args.concepts or REDUCED_CONCEPTS
        layers = args.layers if args.layers is not None else MIDDLE_LAYERS
    else:
        concepts = args.concepts or NATURAL_CONCEPTS
        layers = args.layers
    cvs = load_cvs()

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
    # from_pretrained_no_processing (vs. plain from_pretrained) skips several weight-
    # folding/centering passes (fold_ln, center_writing_weights, center_unembed,
    # refactor_factored_attn_matrices, fold_value_biases) -- each of these operates on
    # full-model-sized weight tensors and plausibly creates full intermediate copies
    # during conversion, which is the likely cause of a large transient memory spike
    # observed during model loading (free RAM briefly collapsing to ~1GB before the
    # model even finishes loading, independent of the eventual bf16/fp32 dtype choice).
    # transformer_lens itself warns "with reduced precision, it is advised to use
    # from_pretrained_no_processing instead" when dtype != float32. Safe for this
    # project: nothing here touches attention weights (W_Q/W_K/W_V) or relies on
    # LayerNorm folding/centering -- only model.W_U (vocab_projection.py) and
    # model.blocks[i].mlp.W_out (editor.py) are used, and Gemma already skips
    # center_unembed/center_writing_weights internally regardless (RMSNorm, not
    # LayerNorm, plus logit softcap makes center_unembed inapplicable -- see the
    # warnings transformer_lens already prints for this model).
    model = HookedSAETransformer.from_pretrained_no_processing(MODEL_NAME, device=args.device, dtype=model_dtype)
    # Nothing in this pipeline trains/backprops -- it's pure inference. None of
    # pisces_ref's forward-pass call sites (get_feature_effect, get_feature_activations,
    # get_mlp_act_signs, evaluate_mmlu) wrap their model(...) calls in torch.no_grad(),
    # so every one of the thousands of forward passes in a real discovery run builds and
    # retains a full backward-computation graph for a 2B-param model -- observed in
    # practice as free RAM collapsing from ~22GB to ~300MB within a few hundred passes.
    # requires_grad_(False) covers the base model's parameters; the outer no_grad() below
    # additionally covers the SAE parameters (loaded separately, not covered by the line
    # above) and anything else in the call tree, so nothing anywhere builds a graph.
    model.requires_grad_(False)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for concept in concepts:
            try:
                df = discover_concept(
                    model, cvs, concept, layers=layers, device=args.device, reduced=args.reduced,
                    debug_log_noop_edits=args.debug_log_noop_edits, minmatch_override=args.minmatch,
                    disable_cascade=args.disable_cascade,
                )
            except ValueError as e:
                print(f"[{concept}] SKIPPED: {e}")
                continue

            out_path = ARTIFACTS_DIR / f"{concept.lower().replace(' ', '_')}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[{concept}] wrote {len(df)} candidate features to {out_path}")


if __name__ == "__main__":
    main()
