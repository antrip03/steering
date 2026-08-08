# Track A — Feature discovery

**Critical path.** Tracks B and C both consume this track's output; until it
produces real data they should work against mock `FeatureRecord` rows.

## The gap this fills

PISCES's own repo (`pisces_ref/`) never implements the vocabulary-projection
lookup (`lls`) that `feature_finder.py::search_features` expects as an *input*.
The only public demo (`erasing_harry_potter.ipynb`) shows five hand-picked
features with no live discovery step. See the top-level README for why we're
reconstructing this ourselves rather than using the two alternatives we
checked (a companion feature-descriptions repo covering only a sampled subset,
and Neuronpedia's bulk exports, which would add an external dependency).

## Files

- `vocab_projection.py` — builds the `lls` lookup: loads each layer's SAE the
  same way `pisces_ref/editor.py`'s `SAEConfig` does, projects every decoder
  direction (`sae.W_dec`) through the model's unembedding matrix (`model.W_U`),
  and ranks top/bottom-50 vocabulary tokens per feature. This is the only new
  algorithmic piece — everything downstream is PISCES's own, unmodified code.
- `concept_tokens.json` — per-concept `search_tokens` / `pos_toks` / `neg_toks`
  needed to drive `search_features` and
  `filter_features_by_effect_and_activations`. **Only Harry Potter is filled
  in** (taken from the notebook's signs-computation cell); the other 14
  natural concepts are left `null` on purpose — PISCES doesn't publish these
  anywhere, and guessing them for the sensitive concepts (Suicide, Mass
  Shooting, Rape, Opioid) isn't something to do silently. Fill in `neg_toks`
  (near-domain contrastive tokens whose probability should *not* drop on
  erasure — this one has no PISCES precedent at all, even for Harry Potter)
  before running a new concept.
- `run_discovery.py` — CLI entrypoint. For each concept: builds the lookup,
  runs PISCES's own `search_features` to get every candidate feature, runs
  PISCES's own `filter_features_by_effect_and_activations` +
  `filter_features_by_mmlu` to get the selected subset, and writes
  `artifacts/features/<concept>.parquet` with **every candidate considered**
  (needed later for Track B's filter pull-in-rate metric), each row matching
  `schema.FeatureRecord`.

## Sanity check before running all 15 concepts

```
python run_discovery.py --concept "Harry Potter"
```

Compare the `selected=True` rows against the five features hardcoded in
`pisces_ref/erasing_harry_potter.ipynb`:
`Feature(1, 8965, True)`, `Feature(1, 13394, False)`, `Feature(4, 661, True)`,
`Feature(20, 11104, True)`, `Feature(20, 14668, False)`.

Only once that matches (or the discrepancy is understood) should the other 14
concepts be run — and only after their `concept_tokens.json` entries are filled
in.

## Requirements

CUDA GPU, PISCES's dependencies (`../requirements.txt`), and network/hub
access to download Gemma-2-2B-it and the GemmaScope MLP SAEs.
