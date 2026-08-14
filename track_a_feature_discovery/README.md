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

## Runtime reductions (`--reduced` / `reductions.py`, `kaggle` branch)

`discover.py --reduced` applies four config-level speed-ups on top of the
unmodified `pisces_ref` selection logic (nothing here changes what "selected"
*means*, only how much work goes into measuring it). All constants live in
`reductions.py`, shared with `run_kaggle.py`. Full rationale and per-item
source verification is in that module's docstring; summary:

1. **Effect-measurement corpus**: `get_feature_effect` originally runs over
   every line of a concept's `wikipedia_content` (~85 batches for Golf).
   `--reduced` uses `evenly_spaced_subsample_lines()` to take 20
   representative batches spread across the whole article, not just its
   opening.
2. **VocabProj threshold**: default `minmatch=1` is far looser than PISCES's
   own paper (Appendix A.1: "intersection size greater than a threshold alpha
   (we used alpha = 4)", verified against the actual PDF text — note the
   strict `>` means `minmatch=5`, not `4`, reproduces it exactly).
   `--reduced` also runs a cheap `CASCADE_PREFILTER_BATCHES`-batch pre-pass
   (`cascade_filter_candidates`) that drops the bottom `1 -
   CASCADE_KEEP_FRACTION` of candidates by a pos/neg-effect heuristic score
   before the full (already-reduced) measurement — this one is a **heuristic**,
   not provably exact, unlike the early-exit below. Also enables
   `early_exit_after_batches` (`pisces_ref/feature_finder.py::get_feature_effect`,
   `steering-fixes` fork branch): a provably-safe short-circuit that stops
   evaluating a candidate once its remaining batches mathematically cannot
   change the keep/drop decision, given softmax-diff values are bounded in
   `[-1, 1]`.
3. **Layer scope**: `MIDDLE_LAYERS = range(3, 13)`, per the task's EMBER
   Figure 7 citation — **not independently verified** here (unlike the
   alpha=4 claim above; the EMBER paper wasn't available locally to check
   against). Used only as the `--reduced` default for `--layers`; pass
   `--layers` explicitly to override.
4. **Concept scope**: `REDUCED_CONCEPTS` (6 of 15), preserving the five-
   category entanglement design — general/specific pair (Poison, Uranium),
   max-entanglement stress test (Homo Sapiens), clean low-entanglement
   control that's also a PISCES paper concept (Golf), intersectional pair
   (Gun, Mass Shooting). Used only as the `--reduced` default for
   `--concept`; pass `--concept` explicitly to override.

**Validation status (Step 2.5)**: the task's required check — run Golf once
at original settings and once at `--reduced`, diff the selected FC, report
exact matches/differences — needs a real Gemma-2-2B-it forward-pass run and
has **not been executed**. This machine has no CUDA GPU (AMD Radeon 680M
iGPU) and 14.3GB RAM, and a prior session already observed near-OOM loading
the model alone in fp32; a full-corpus, all-26-layer, `minmatch=1` "original
settings" run is the most expensive possible configuration in this whole
project and was not attempted here. Treat every reduction above as
**unvalidated** against the real model until this run is done — this
directory's non-`--reduced` default remains PISCES's original, unrestricted
behavior specifically so that comparison stays possible once a suitable
machine is available.

### `build_layer_lookup` OOM fix (chunked vocabulary projection)

`vocab_projection.py::build_layer_lookup` used to project an SAE's *entire*
feature set through `model.W_U` in one matmul: `sae.W_dec.float() @
model.W_U.float()` materializes a dense `[n_features, d_vocab]` fp32 tensor
— for a 16k-width SAE against Gemma-2-2B-it's ~256k vocab, that's `16384 ×
256000 × 4 bytes ≈ 15.6 GiB`, which OOMs on a 16GB T4 (and contributed
memory pressure on any smaller machine) despite only the top/bottom-`k` per
feature ever being used downstream. Fixed by processing features in chunks
of `FEATURE_CHUNK_SIZE` (default 2000, ≈1.9GiB peak per chunk), extracting
each chunk's top/bottom-k immediately and discarding the chunk's full
projection before the next one — `chunk_size` is a parameter on both
`build_layer_lookup` and `build_all_layer_lookups`, not a hardcoded value.

**Verified**: `test_vocab_projection.py::test_chunked_matches_unchunked_on_random_features`
confirms chunked output is byte-identical to unchunked (random, non-tied
weights; a non-divisible feature count against several chunk sizes,
including a size that forces a partial last chunk) — this is a pure
reshaping-of-computation change, not an approximation, so exact equality is
the correct bar, not "close enough." `test_chunking_never_materializes_more_than_one_chunk_at_once`
directly checks the memory-bound property by spying on every `topk` call's
input shape.

**Not verified**: an actual `[16384, 256000]` OOM on real GPU hardware, and
real peak-memory numbers before/after the fix, since this machine has no
CUDA device. The synthetic tests above prove the fix is *correct*; they
don't independently prove the *original* bug actually manifested as
described (an external report referenced this fix alongside a claimed
correlation to an earlier, separately-observed CPU crash — that report
wasn't available to verify against, so treat the crash-correlation claim as
unconfirmed; the OOM arithmetic itself, independent of that report, is
directly verifiable by reading the code and was confirmed that way).

### `replace_mlp_rows` "No changes made" assertion — investigation (unresolved)

`pisces_ref/editor.py::replace_mlp_rows` asserts that every layer edit
actually changes `W_out`:
```python
assert not torch.allclose(model.blocks[layer].mlp.W_out, layer_backups[layer]), \
    f"No changes made to the model in layer {layer}"
```
This was reported firing during a real Golf/layer-1 discovery run (fp16, T4)
after 69 candidates were found and one full 85-batch pass completed. That
run's logs/artifacts aren't available on this machine (no CUDA GPU here, and
`artifacts/` doesn't exist locally), so the exact candidate/layer that
triggered it, and how many of the 69 candidates are affected, are **not
known** — the investigation below is code-level reasoning plus a synthetic,
GPU-free reproduction, not a run against the real failure.

**Traced the edit math** (`get_hswaps_full_signed`/`get_hswaps_full`,
`editor.py:279-432`): `clean`, `dirty`, `error_term`, `proj`, `affected`,
`fixed_affected` are all computed in fp32 (`w_out = ...W_out.float()`
upcasts before any of it). The one fp16 downcast is
`hswaps.append((index, fixed_affected.to(model_dtype)))`, which only runs
*after* `not torch.allclose(fixed_affected, clean)` already passed in fp32.
So the originally-proposed fp16 hypothesis, if true, is specifically about
that downcast — not about the edit math itself producing a trivially-small
delta (that case is already filtered out before reaching `hswaps`).

**Found two distinct, code-grounded mechanisms that produce this exact
symptom**, only one of which is the fp16 hypothesis:

1. **fp16 storage-rounding collapse** (the proposed hypothesis): the fp32
   edit is confirmed non-negligible, but `.to(model_dtype)` rounds it away.
   A synthetic CPU reproduction (coherent single-direction delta at
   `d_model=2304`, mirroring how one SAE latent's decode contributes to a
   row) confirms this is *possible*, but only below roughly a **1e-7
   relative** perturbation of the row's own magnitude — about 1000x smaller
   than fp16's ~1e-3 rounding granularity alone would suggest, because
   `torch.allclose`'s default `rtol=1e-5` is tighter than fp16 precision, so
   *every one* of 2304 dimensions has to round-collapse simultaneously for
   the whole-row comparison to pass. A single element collapses around
   ~1.5e-4 to 3e-4 relative; the full-row requirement is far stricter than
   that. Whether this is "easy" to trigger for a real SAE decoder direction
   depends on how concentrated/sparse that direction is — not something
   knowable without the actual SAE weights.
2. **Empty edit set (dead SAE encoder feature)** — not proposed in the
   original hypothesis, found while tracing the code: `encoded = w_out @
   sae.W_enc[:, feature.id]`. If that feature's encoder column is exactly
   zero (a common, well-documented phenomenon in trained SAEs — "dead
   features" — and structurally unrelated to `search_features`, which only
   ever looks at the *decoder* direction), `thresh=0` and `high_indices=[]`,
   so `hswaps` for that layer is empty. `replace_mlp_rows` then has nothing
   to assign, `changed` stays byte-identical to `backup`, and the assertion
   fires **deterministically, in any precision** — nothing to do with fp16.

These two are trivially distinguishable by one fact this investigation
doesn't have: whether `switches` was empty or non-empty at the point of
failure. Added `debug_log_noop_edits` (default `False`, existing behavior
completely unchanged) to `replace_mlp_rows`/`steer_features`/
`unlearn_concept` (`pisces_ref/editor.py`) so the next real run can capture
this directly instead of just crashing — when a no-op is detected and the
flag is on, it logs which of the two cases occurred (and, for the
non-empty case, each affected index's edit-delta norm and relative
magnitude) and continues instead of raising. Verified with
`pisces_ref/test_editor.py` (5 tests, no GPU/real SAE needed — stubs
`sae_lens` since `editor.py` imports it at module level purely for other
functions): default behavior is unchanged (still asserts on a no-op, still
applies/reverts real edits normally), and the diagnostic correctly
distinguishes both no-op causes, including a realistic collapsed-delta case
(a 2e-4 relative perturbation, which is fp32-visible but fp16-identical).

**Update — real data from a Kaggle T4 run (`discover.py --concept Golf
--layers 1 --debug-log-noop-edits`, `bf16` — note `discover.py` hardcodes
`bfloat16` unconditionally, not fp16; "fp16" above should be read as
"whichever narrow-precision dtype is in play"):** the diagnostic fired,
confirming mechanism 2 (empty switches / dead SAE encoder column) — every
occurrence observed so far says `0 switches were computed at all`, never
the storage-collapse variant. Exactly one debug line appeared per batch
(never zero, never two+ within a batch), consistent with a single
recurring dead candidate rather than a systemic failure — but the original
diagnostic only logged the layer, not the feature, so this couldn't be
confirmed with certainty. **Fixed**: `replace_mlp_rows` now also accepts
`layer_features` (the `Feature` objects `steer_features` already has in
scope) purely for the log line, so the next run will print the exact
`Feature(layer=.., id=.., neg=..)` involved instead of just the layer.
Still not known: whether it's the same feature every batch, and whether
the storage-collapse mechanism ever occurs in this run — needs a rerun
with the updated diagnostic.

**Not done, and out of scope for this pass**: counting how many of the
real 69 candidates are affected, and any actual fix (skip-and-log vs.
fp32-comparison-only mixed precision) — per the task's own instruction to
report first and wait for review. Whoever has GPU access to the failing
run should pass `debug_log_noop_edits=True`
through to `unlearn_concept`/`get_feature_effect` for that run; the printed
output directly answers which mechanism (or both) is occurring and how often.

### CUDA OOM a few batches into a real run — root cause found and fixed

Separate from the assertion above: the same `--debug-log-noop-edits` Golf/
layer-1 run hit a real `torch.OutOfMemoryError` a few batches in (`Tried to
allocate 536.00 MiB... 1.66 GiB is reserved by PyTorch but unallocated`),
crashing during a plain forward pass unrelated to the no-op investigation.

Root cause: `SAEConfig.get()` (`pisces_ref/editor.py`) had no caching, and
`get_hswaps_full(_signed)` calls it once per candidate feature **per
batch** via `unlearn_concept` — for Golf/layer-1's 69 candidates × 85
batches, that's up to **5,865 separate fresh `SAE.from_pretrained` loads**
of the exact same never-changing layer-1 SAE. Thousands of ~300-400MB
alloc/free cycles of an unchanging object is exactly what fragments
PyTorch's caching allocator until a normal-sized allocation can't find
contiguous space despite nominal free memory left — consistent with the
"reserved but unallocated" detail in the error.

**Fixed**: `SAEConfig.get()` now caches by `(release, sae_id, device)`, so
each distinct SAE is loaded exactly once and reused across every
subsequent call. Safe because the SAE is never mutated by any caller (only
`.W_enc`/`.W_dec`/`.encode`/`.decode` are read). Verified with a test that
mocks `SAE.from_pretrained` and confirms identical `(release, sae_id,
device)` returns the same object and calls the real loader exactly once,
while a different layer correctly gets its own separate load.

Not verified against the real Kaggle failure directly (no GPU on this
machine) — the fix addresses the exact mechanism the error points to, but
whether it fully eliminates the OOM on a real 85-batch run, or just delays
it, needs confirming on an actual rerun.

## Requirements

CUDA GPU, PISCES's dependencies (`../requirements.txt`), and network/hub
access to download Gemma-2-2B-it and the GemmaScope MLP SAEs.
