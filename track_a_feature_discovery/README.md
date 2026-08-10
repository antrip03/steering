# Track A — Feature discovery

**Critical path.** Tracks B and C both consume this track's output; until it
produces real data they should work against mock `FeatureRecord` rows.

## Status: construction complete, not yet run

Everything below is written and unit-tested. **`discover.py` has not been run
against the real model or real concept data.** Running it — even just the
Harry Potter sanity check — requires loading Gemma-2-2B-it and the GemmaScope
MLP SAEs and needs explicit sign-off first, because two real methodological
choices below (seed-token construction, `neg_toks`) haven't had a team review
pass yet. See "Open design decisions" below for exactly what to review.

## The gap this fills

PISCES's own repo (`pisces_ref/`) never implements the vocabulary-projection
lookup (`lls`) that `feature_finder.py::search_features` expects as an *input*,
nor the seed-token / `neg_toks` lists that `search_features` and
`filter_features_by_effect_and_activations` need as arguments. The only public
demo (`erasing_harry_potter.ipynb`) shows five hand-picked features with no
live discovery step and no reusable token lists. See the top-level README for
why we're reconstructing this ourselves rather than using the two alternatives
we checked (a companion feature-descriptions repo covering only a sampled
subset, and Neuronpedia's bulk exports, which would add an external
dependency).

## Data flow (traced from `pisces_ref/feature_finder.py::find_hps`)

1. **Load pretrained SAEs**, one per MLP layer (`sae_loader.py`).
2. **Build `lls`**: for every feature in every layer's SAE, project the
   decoder direction through the unembedding matrix and rank top/bottom-K
   vocabulary tokens (`vocab_projection.py`). This is the only genuinely new
   *algorithmic* piece — everything downstream is PISCES's own, unmodified
   code.
3. **Derive seed tokens** per concept (`seed_tokens.py`). Not present in
   `cvs.json`; this single list plays three roles in PISCES's own code, all
   under the name `pos_toks`: seeding `search_features`, driving
   `get_mlp_act_signs`, and the positive side of
   `filter_features_by_effect_and_activations`.
4. **Derive `neg_toks`** per concept (`seed_tokens.py`) — a second, separate
   token list used only in the effect filter. Also not present in `cvs.json`
   and no construction recipe exists anywhere in `pisces_ref`.
5. `candidates = search_features(model, lls, seed_tokens)`.
6. `signs = get_mlp_act_signs(...)`; `filtered = filter_features_by_effect_and_activations(...)`;
   `filtered = filter_features_by_mmlu(...)`. `filtered` is the final selected
   feature set (`FC`) for the concept.
7. Cache both `candidates` (`selected=False`) and `filtered`
   (`selected=True`) per `schema.FeatureRecord`.

Steps 5–7 call `pisces_ref/feature_finder.py`'s `search_features`,
`filter_features_by_effect_and_activations`, and `filter_features_by_mmlu`
**unmodified**. Only steps 1–4 are new code.

## Files

- `sae_loader.py` — loads per-layer pretrained SAEs via `SAEConfig`, matching
  `pisces_ref/editor.py`'s own pattern (same release/`sae_id` derivation, same
  default width: "16k" for Gemma, "32k" for Llama).
- `vocab_projection.py` — builds the `lls` lookup (step 2): projects every
  decoder direction (`sae.W_dec`) through the model's unembedding matrix
  (`model.W_U`) and ranks the top/bottom-50 vocabulary tokens per feature.
  `build_layer_lookup` is a pure function of an already-loaded SAE and model
  (no hub/GPU dependency), which is what makes it unit-testable against
  synthetic tensors; `build_all_layer_lookups` wires it up to `sae_loader.py`
  for real use.
- `seed_tokens.py` — implements the seed-token and `neg_toks` construction
  (steps 3–4) as clearly documented, named functions — not inlined magic
  numbers. See "Open design decisions" below.
- `discover.py` — orchestration entrypoint: for each concept, runs steps
  1–7 and writes `artifacts/features/<concept>.parquet` with **every
  candidate considered** (needed for Track B's filter pull-in-rate metric),
  each row matching `schema.FeatureRecord`.
- `test_vocab_projection.py`, `test_seed_tokens.py` — unit tests, both run
  without a GPU, a real SAE, or a downloaded tokenizer (see "Testing" below).

## Open design decisions

Both of these are real methodological choices, not incidental implementation
detail, and should get a team review pass — especially before running the
sensitive concepts (Suicide, Mass Shooting, Rape, Opioid) — before being
baked into an actual run.

### Seed-token construction (`extract_seed_tokens` / `derive_seed_tokens_for_concept`)

Implemented as: extract salient **single-token** words from the concept's
`wikipedia_content`, scored TF-IDF-style — `frequency in this concept's
article / (1 + document frequency across the other 14 concepts' articles)` —
so words that are common to every article (regardless of raw frequency) rank
low, and words distinctive to this concept rank high. Ties are broken by raw
frequency, then alphabetically, for determinism. Candidates are tried
highest-score-first and skipped if they don't tokenize to exactly one token
in the target model's vocabulary (`search_features` asserts this); the
surviving top-N are returned in ` Word` form (leading space, matching Gemma's
word-initial tokenization convention) and used as `pos_toks` everywhere
PISCES's code expects that name.

Why this and not something else: it needs no external corpus or NER model,
it's fully automatic (no per-concept curation, which doesn't scale to 15
concepts and doesn't generalize to future ones), and it's easy to audit — the
score is one formula and the output is a short, readable token list per
concept, printed by `discover.py` before it does anything else with it. The
known weakness: pure word-frequency ranking has no notion of importance
beyond how often a word is mentioned, so a concept whose Wikipedia article
mentions its own name less often than a generic descriptor could rank the
descriptor higher. Worth checking directly against the Harry Potter sanity
check, since PISCES's own hand-picked tokens for that concept are
recoverable from `erasing_harry_potter.ipynb` (` Harry`, ` Potter`,
` Hermione`, ` Weasley`, ` Hogwarts`, ` Snape`, ` Malfoy`, ` Voldemort`) and
can be compared against what `derive_seed_tokens_for_concept` produces.

### `neg_toks` construction (`NEUTRAL_NEG_TOKENS` / `get_neg_toks`)

Implemented as: a small, fixed set of high-frequency, semantically neutral
English function words (`" the"`, `" and"`, `" is"`, `" of"`, `" to"`,
`" a"`, `" in"`, `" that"`), reused identically across every concept, kept as
a single named constant in `seed_tokens.py`.

Why this and not something else: `neg_toks` has **no precedent anywhere in
`pisces_ref`**, not even for Harry Potter — `filter_features_by_effect_and_activations`
takes it as a required argument, but no PISCES artifact (paper, notebook, or
code comment) says what it should contain. A fixed, topic-agnostic list is
the simplest defensible default and needs no per-concept curation. The
tradeoff: because it's the same for every concept, it can only catch a
candidate feature's effect on *generic* language, not on a concept's
specific near-domain (e.g. it won't catch a "Harry Potter" feature that also
suppresses "Lord of the Rings" vocabulary) — that's what the token-overlap
entanglement metric (Track B, metric c) is *separately* designed to measure,
so this isn't trying to duplicate it. If the Harry Potter sanity check shows
the fixed list is too weak or too strong a filter, `get_neg_toks` is the one
place to change it.

## Testing

```
pip install --break-system-packages -r requirements.txt   # or just: pip install pytest torch
cd track_a_feature_discovery
pytest test_vocab_projection.py test_seed_tokens.py -v
```

Both test files run without a GPU, without `sae_lens`/`transformer_lens`
installed, and without downloading anything:

- `test_vocab_projection.py` exercises `build_layer_lookup` against small
  hand-built tensors with a fake SAE/model (only `W_dec`/`W_U`/`to_string`
  are used), so the ranking logic is checked directly against
  hand-computable expected rankings. `build_all_layer_lookups` (which does
  need `sae_loader.py` → `pisces_ref/editor.py` → `sae_lens`) is not unit
  tested here — it's a thin wiring function with nothing to unit-test beyond
  what `build_layer_lookup` already covers, and `sae_lens`'s own loading
  path isn't ours to test.
- `test_seed_tokens.py` exercises `extract_seed_tokens` /
  `derive_seed_tokens_for_concept` / `get_neg_toks` against small
  hand-written text snippets with a known expected token set, using a fake
  `is_single_token` predicate instead of a real tokenizer.

## Sanity check before running all 15 concepts — DO NOT RUN YET

```
python discover.py --concept "Harry Potter"
```

Compare the `selected=True` rows against the five features hardcoded in
`pisces_ref/erasing_harry_potter.ipynb`:
`Feature(1, 8965, True)`, `Feature(1, 13394, False)`, `Feature(4, 661, True)`,
`Feature(20, 11104, True)`, `Feature(20, 14668, False)`. Also compare the
printed `seed_tokens=` line against PISCES's own hand-picked Harry Potter
tokens listed above, as a check on the TF-IDF extraction heuristic
specifically.

Only once that matches (or the discrepancy is understood), and only after the
two open design decisions above have had a team review pass, should the
other 14 concepts be run.

## Requirements

CUDA GPU, PISCES's dependencies (`../requirements.txt`), and network/hub
access to download Gemma-2-2B-it and the GemmaScope MLP SAEs.
