# Track C — Erasure + evaluation

Runs PISCES's own (unmodified) `unlearn_concept` context manager
(`pisces_ref/editor.py`) per concept and evaluates efficacy/specificity by
reusing PISCES's own (unmodified) evaluation functions
(`pisces_ref/evals.py`). Produces the `efficacy` / `specificity_simdomain` /
`specificity_mmlu` columns of the combined results table
(`schema.ConceptResultRow`).

## Can start independently of Track A

`run_erasure_eval.py --hardcoded-hp` runs the pipeline against the five
features hand-picked by PISCES's authors for Harry Potter (hardcoded in
`pisces_ref/erasing_harry_potter.ipynb`), bypassing Track A entirely. Use this
to validate the erasure + evaluation plumbing before Track A produces real
selected-feature sets. Once Track A has written
`artifacts/features/<concept>.parquet`, drop `--hardcoded-hp` and this track
reads the `selected=True` rows from there instead.

## What it does per concept

1. Loads the concept's `Feature` list (Track A's selected set, or the
   hardcoded HP set).
2. Computes MLP activation signs (`get_mlp_act_signs`) from the concept's
   `pos_toks` (from `track_a_feature_discovery/concept_tokens.json` — shared
   with Track A rather than duplicated) against `wikipedia_content` from
   `data/cvs.json`.
3. Runs `unlearn_concept(model, concept, linscale=True, signs=signs)`.
4. Under that context: evaluates `QA_test` (efficacy — should be *low* after a
   good erasure), `SimdomQA_test` (specificity to a similar domain — should
   stay *high*), and MMLU (general specificity — should stay *high*).

Writes `artifacts/erasure_eval_results.parquet`.

## Requirements

CUDA GPU, PISCES's dependencies, and a `GEMINI_API_KEY` environment variable
(`evals.GeminiEvaluator` grades open-ended answers via the Gemini API).

## Not yet implemented (explicitly deferred, per project scope)

- Hyperparameter search per concept (`feature_finder.py::find_hps`) — `k`/`value`
  (tau/mu) currently default to the notebook's hardcoded Harry Potter values
  for every concept, which is a real limitation once this runs on the other 14.
- Robustness / relearning-attack accuracy — deferred for this phase.
