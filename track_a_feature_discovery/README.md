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

`discover.py --reduced` applies three config-level speed-ups on top of the
unmodified `pisces_ref` selection logic (nothing here changes what "selected"
*means*, only how much work goes into measuring it) — originally four; the
VocabProj-threshold tightening (item 2 below) was tried, real-run validated,
and **dropped** (see "Validation status" below for the full story). All
constants live in `reductions.py`, shared with `run_kaggle.py`. Full
rationale and per-item source verification is in that module's docstring;
summary:

1. **Effect-measurement corpus**: `get_feature_effect` originally runs over
   every line of a concept's `wikipedia_content` (~85 batches for Golf).
   `--reduced` uses `evenly_spaced_subsample_lines()` to take 20
   representative batches spread across the whole article, not just its
   opening.
2. **~~VocabProj threshold~~ (dropped, back to `minmatch=1`)**: tried
   `minmatch=5`, matching PISCES's own paper (Appendix A.1: "intersection
   size greater than a threshold alpha (we used alpha = 4)", verified
   against the actual PDF text — the strict `>` means `minmatch=5`, not
   `4`, reproduces it exactly). Real-run validation on Golf found it
   collapses the entire candidate pool to zero, at both a tested early
   layer and a tested middle layer, with no viable intermediate value (92
   candidates at `minmatch=1`, zero at every value from 2 to 5 — a hard
   cliff). `VOCABPROJ_MINMATCH` is back to 1 in `reductions.py`; `--minmatch`
   remains available on `discover.py` to re-test this per-run without
   another code change. `--reduced` also runs a cheap
   `CASCADE_PREFILTER_BATCHES`-batch pre-pass (`cascade_filter_candidates`)
   that drops the bottom `1 - CASCADE_KEEP_FRACTION` of candidates by a
   pos/neg-effect heuristic score before the full (already-reduced)
   measurement — this one is a **heuristic**, not provably exact, unlike
   the early-exit below. Also enables `early_exit_after_batches`
   (`pisces_ref/feature_finder.py::get_feature_effect`, `steering-fixes`
   fork branch): a provably-safe short-circuit that stops evaluating a
   candidate once its remaining batches mathematically cannot change the
   keep/drop decision, given softmax-diff values are bounded in `[-1, 1]`.
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

**Validation status (Step 2.5) — DONE, and it FAILED.** Both runs executed
on a real Kaggle T4 for Golf/layer-1 (this development machine has no CUDA
GPU, so this required real GPU hardware, obtained separately):

| | original (`minmatch=1`) | `--reduced` (`minmatch=5`) |
|---|---|---|
| candidates found | 69 | **0** |
| `selected=True` | 58 | **0** |

`VOCABPROJ_MINMATCH=5` does not just shift a few borderline candidates —
it **eliminates the entire candidate pool** for this concept/layer. None
of layer 1's 16,384 SAE features had a top/bottom-50 vocab-projection
overlap of ≥5 tokens with Golf's 8 seed tokens, versus 69 features clearing
the ≥1 bar. This is the exact "if results diverge, report clearly rather
than silently keeping the reduction" scenario the task called for — and
they diverge completely. **`VOCABPROJ_MINMATCH=5` is not safe to keep as
the `--reduced` default without further work.**

**Follow-up — isolated, and confirmed not layer-1-specific.** Added
`--minmatch` (independent of `--reduced`) to test `VOCABPROJ_MINMATCH=5`
alone, with the other three reductions off (85 batches, no cascade
prefilter, no early-exit) and at a genuine middle layer:

```
python discover.py --concept Golf --layers 6 --minmatch 5
```

Result: **0 candidates**, same as layer 1. This rules out the "layer-1 is
an early layer with less semantically specialized VocabProj" hypothesis —
the collapse happens at layer 6 too, with the other three reductions
completely out of the picture. `VOCABPROJ_MINMATCH=5` itself is the
problem, not a confound with cascade filtering/reduced corpus/early-exit,
and not an early-layer artifact.

What's still open: this project's seed tokens are auto-extracted via
TF-IDF (`seed_tokens.py`), not PISCES's own hand-curated lists — the
paper's `alpha=4` finding was calibrated against their token construction
method, which may produce systematically different (denser) per-feature
token overlap than this project's shorter, more targeted seed lists. That
remains the leading candidate explanation, untested. Options for a next
step, not decided here: try an intermediate `--minmatch` value (2 or 3);
test at additional middle layers to see if 6 was unusual; or drop this
reduction and keep `minmatch=1` while retaining the other three, which
remain unaffected by this finding.

**Follow-up — swept, and there's no intermediate value.** Built
`minmatch_sweep.py` (loads the model/SAE once, then sweeps several
`minmatch` values cheaply — `search_features` itself is fast; only the
downstream effect-measurement/activation/MMLU stages, not needed to answer
this, are what's expensive) and ran it at layer 6 for `minmatch` 1 through
5:

```
minmatch  candidates
       1          92
       2           0
       3           0
       4           0
       5           0
```

Not a gradual falloff — a hard cliff between 1 and 2. Every value from 2
to 5 is equally catastrophic; there is no viable intermediate threshold
for Golf's seed-token list at this layer. Leading hypothesis (not
verified against other concepts): PISCES's own hand-curated seed lists are
tightly-clustered proper nouns (e.g. Harry Potter's character names) that
plausibly co-occur within a single feature's top-tokens naturally, while
this project's TF-IDF-extracted lists for concepts like Golf (`golf`,
`Tour`, `hole`, `par`, `golfer`, `PGA`, `tee`, `Championship`) are more
generic English vocabulary that likely maps to more scattered feature
directions — making even 2-token co-occurrence within one feature rare
regardless of the exact threshold chosen.

**Conclusion for this reduction**: `VOCABPROJ_MINMATCH` should stay at 1
(i.e. drop this specific reduction) rather than any value ≥2, at least for
concepts with TF-IDF-derived generic-vocabulary seed lists like Golf. The
other three reductions (cascade filtering, reduced effect-measurement
corpus, early-exit) are unaffected by this finding and remain viable.

**Done in code.** `VOCABPROJ_MINMATCH` in `reductions.py` is now `1`, so
`--reduced` no longer tightens VocabProj's threshold at all — only the
other three reductions remain in the bundle. `discover.py --minmatch`
stays available as a standing per-run override (not a default change) for
re-testing this against a different concept later, e.g. one with more
proper-noun-heavy, PISCES-paper-like seed tokens (Homo Sapiens is the
obvious candidate given its role as this project's max-entanglement stress
test) — untested so far, since every result above is Golf-only.

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

**Update — confirmed fixed for this stage on a real rerun.** The SAE-caching
fix worked: the 85-batch effect-measurement stage that previously crashed
at batch 4-5 completed all 85 batches (~94 minutes). The feature-ID logging
also resolved the earlier open question — every no-op debug line across
all 85 batches named the exact same candidate, `Feature(layer=1,
id=13028, neg=False)`, confirming it's one recurring dead SAE encoder
column, not a systemic problem or multiple different candidates.

### A second, separate CUDA OOM — in a different function entirely

Immediately after the effect-measurement stage completed, the very next
stage (`get_feature_activations`, called from
`filter_features_by_effect_and_activations` when `filter_by_act=True`) hit
its *own* `torch.OutOfMemoryError`, at batch 64/85, with memory now
essentially fully exhausted (`14.54/14.56 GiB in use, 18.81 MiB free`) —
notably worse than the first OOM's numbers, and via a completely different
mechanism (`model.run_with_cache_with_saes`, not the per-candidate
`unlearn_concept` edit loop). This function also had **no checkpointing at
all**, meaning a bare crash-and-retry would have required redoing the
entire ~94-minute effect-measurement stage first, for nothing.

Root cause: `run_with_cache_with_saes` caches every hook point across the
whole 26-layer model by default, but this function only ever reads one
`hook_sae_acts_post` tensor per unique layer among the candidates (here,
just layer 1). **Fixed**: added a `names_filter` restricting the cache to
exactly the hooks this function reads, and added checkpoint/resume
mirroring `get_feature_effect`'s own (save `results` + next batch index
after every batch; resume and skip completed batches on restart).
`names_filter` is a long-standing, stable `transformer_lens` parameter, and
the traceback confirms `run_with_cache_with_saes` forwards to
`self.run_with_cache`, but — same caveat as the first fix — whether this
fully eliminates the OOM rather than just reducing it isn't verified
against the real failure (no GPU on this machine). Verified with 3 tests
against a stub model: `names_filter` receives exactly the needed hook names
and excludes others, firing counts accumulate correctly across batches, and
checkpoint resume skips completed batches while adding to (not replacing)
the prior count.

**Update — confirmed fixed on a real rerun, and a third issue found.**
Both stages completed successfully: the effect-measurement stage that
previously crashed at batch 4-5 ran all 85 batches (~1h43m this time), and
the activation-counting stage that previously crashed at batch 64/85 also
completed all 85 batches — in just 66 seconds, confirming `names_filter`
helped both memory *and* speed (restricting the cache is much cheaper than
building it for the whole model). Next: `filter_features_by_mmlu`
(`pisces_ref/feature_finder.py`) stalled for several minutes with no
visible output.

Root cause, found by reading `evaluate_mmlu` (`pisces_ref/evals.py`):
`load_dataset("cais/mmlu", "all")` with no `split=` argument eagerly
generates *every* split in the "all" config — test, validation, dev, and
`auxiliary_train` (99,842 rows) — even though the function only ever reads
`ds["test"]`. The stall was that unused split being generated for nothing;
GPU utilization staying active during the stall (confirmed on the real
run) is consistent with this — genuine, if wasteful, work, not a hang.
**Fixed**: `load_dataset("cais/mmlu", "all", split="test")` loads only the
split that's actually used. Not covered by a dedicated test — `evals.py`'s
import chain (`transformers`, `openai`, `peft`, `transformer_lens`,
`google.generativeai`) is heavy enough that mocking it all for one line
relying on a well-established `datasets` library contract
(`split=` returns a `Dataset` directly, not a `DatasetDict`) wasn't judged
worth it; relying on code review and the next real run instead.

### A fourth issue — the same no-op-edit crash, one call site later, unprotected

The MMLU stall turned out to be a false alarm (real work, not a hang — see
above), but the same process (still running from before the MMLU fix
existed, so it hit the un-fixed dataset-loading path again) then crashed
with the exact bare `AssertionError: No changes made to the model in layer
1` inside `filter_features_by_mmlu` — no diagnostic output at all, because
`debug_log_noop_edits` was only ever wired into `get_feature_effect`'s
`unlearn_concept` call. `filter_features_by_mmlu` has its own, separate
`unlearn_concept` call site that was never touched.

**A real correction to the earlier writeup in this section**: it was
claimed that a candidate whose edit is a no-op throughout the effect-
measurement stage would "fail the selection criterion and get filtered
out." That's wrong. `filter_features_by_effect_and_activations`'s effect
check is `if pos_effect > 0 or neg_effect < -2: continue` (i.e., *remove*
the feature) — for a fully-inert candidate, `pos_effect=0` and
`neg_effect=0`, and neither `0 > 0` nor `0 < -2` is true, so the feature is
**kept**, not removed, by this check alone. Only the separate real-forward-
pass activation check (`activations[...] == 0`) is positioned to catch a
truly inert candidate, and it measures something different (whether the
feature's SAE encoder fires on real text) than whether
`get_hswaps_full(_signed)` can find a qualifying weight-edit for it under
the `k=0.9` threshold and current dtype. A candidate can plausibly fire in
real activations while still having no viable edit, which is exactly
consistent with reaching `filter_features_by_mmlu` and crashing there.

**Fixed**: `debug_log_noop_edits` now threads into
`filter_features_by_mmlu`'s `unlearn_concept` call too — same log-and-
continue behavior as the other call site, verified with a new test that
mocks `unlearn_concept` directly to isolate the plumbing regression from
the (already separately tested) SAE-editing machinery. Still open: with
this candidate's edit now a no-op that gets logged rather than crashing,
its MMLU score will reflect the unedited baseline model (since nothing was
actually changed), which will very likely pass the specificity check
trivially — an inert candidate could end up marked `selected=True` for
reasons unrelated to genuinely erasing anything. Whether that's a real
problem worth a methodology change (e.g., an explicit "no-op edit ->
auto-exclude" step) isn't decided here; flagging it rather than silently
changing PISCES's selection semantics.

### A fifth call site — cascade_filter_candidates, discover.py-side this time

With `VOCABPROJ_MINMATCH` back to 1, the first real `--reduced` run to
actually reach cascade filtering (previously blocked entirely by the
minmatch bug) hit the exact same bare assertion again — one candidate into
`cascade_filter_candidates`'s own `get_feature_effect` call. This function
lives in `discover.py`, not `pisces_ref`, and calls `get_feature_effect`
directly, bypassing `filter_features_by_effect_and_activations` (already
wired) entirely — a separate, unwired path. **Fixed**: threaded
`debug_log_noop_edits` through here too, and proactively did the same for
`run_kaggle.py`'s three equivalent call sites (which have the identical
gap but haven't been run for real yet), adding `--debug-log-noop-edits`
there as well. Verified with a new test mocking `get_feature_effect`
directly.

This makes five separate places `debug_log_noop_edits` needed wiring
(`get_feature_effect`'s own loop, `filter_features_by_effect_and_activations`
forwarding it, `filter_features_by_mmlu`, `cascade_filter_candidates` in
both scripts) — the same latent bug, found incrementally because each real
run only exercises the call sites its current stage reaches. Worth
treating with suspicion if a `--debug-log-noop-edits` run still crashes
bare: check whether a new, still-unwired path was hit before assuming the
diagnostic itself is broken.

### Step 2.5, completed for real — cascade + reduced corpus + early-exit, Golf/layer-1

With `minmatch=1` fixed and all five `debug_log_noop_edits` call sites
wired, `discover.py --concept Golf --layers 1 --reduced` ran to completion
for the first time. `compare_fc.py` result:

```
Candidate pool: original=69  reduced=69  common=69
selected=True in original: 58   selected=True in reduced: 29
exact selection-status matches: 30/69
39 candidate(s) DIFFER
```

**This is a real, substantial divergence — not a pass.** But it splits
into two mechanistically different groups, worth separating before drawing
conclusions:

- **~30 of the 39** show `effect_score=nan` in the reduced run — these are
  candidates the cascade prefilter's 5-batch heuristic pre-pass dropped
  before the full 20-batch measurement ever ran on them. That's cascade
  working exactly as designed (it's *supposed* to cut the pool before
  measurement). The concerning part: almost every one of these had a
  comfortably negative (not borderline) `pos_effect` in the *original*
  run's full measurement — clearly a keeper, not a marginal case. Cascade's
  cheap heuristic score ranked them into the bottom half anyway. That's
  evidence the 5-batch heuristic doesn't correlate well with the real
  20/85-batch outcome, at least at this layer.
- **~9 of the 39** have a real, non-nan effect score in *both* runs, but
  the sign flips near zero — e.g. `Feature(1, 5110, False)`: original
  `pos_effect=+2.2e-7` (excluded), reduced `pos_effect=-8.3e-7` (kept).
  Both values are within a few millionths of zero. Nothing was dropped by
  cascade here; different batch sampling (85 full batches vs. 20
  evenly-spaced) pushed an already-near-zero value across the `pos_effect
  > 0` boundary in different directions. This points to something
  separate from cascade's correctness: at Golf/layer-1, `pos_effect`
  appears to sit at the numerical noise floor for most candidates, which
  means the selection outcome here may be inherently unstable regardless
  of which reduction (if any) is applied — even two "original settings"
  runs could plausibly disagree on some of these. Untested: whether this
  noise-floor behavior is layer-1-specific (a weak/early layer, same
  category of concern as the earlier minmatch investigation) or shows up
  at middle layers too.

**Not concluded here**: whether this means cascade filtering should be
dropped (like `VOCABPROJ_MINMATCH` was), whether it needs a better
heuristic score, or whether the real problem is that layer-1 validation
runs are too noisy to trust for *any* reduction's validation and a middle
layer should be used instead. All three are live options.

### Determinism check — original settings run twice, identical

Ran `discover.py --concept Golf --layers 1` (no `--reduced`) a second
time and diffed it against the first original run with `compare_fc.py`:

```
Candidate pool: original=69  reduced=69  common=69
selected=True in original: 58   selected=True in reduced: 58
exact selection-status matches: 69/69
>>> Exact match <<<
```

**This confirms the earlier `--reduced` divergence is real, not GPU
floating-point noise.** Two runs of the identical config, identical code,
identical data produced identical output — Golf/layer-1 is fully
deterministic under original settings. That sharpens the interpretation of
the 30/69 exact-match result above: the ~9-candidate near-zero sign-flip
group isn't run-to-run instability, it's a genuine consequence of the
reduced 20-batch corpus measuring a different (smaller) sample than the
full 85 batches — a real effect, separate from cascade's own ~30-candidate
contribution.

**Follow-up in progress**: added `--disable-cascade` (only meaningful with
`--reduced`) to isolate reduced-corpus + early-exit from cascade
specifically — skips the cascade prefilter entirely, sending all 69
candidates straight to the reduced-corpus measurement. Tests whether the
~9-candidate group persists on its own, without cascade also in the
picture. Not yet run.

**Superseded by the decision below**: the corpus-size reduction this test
was designed to isolate from cascade has itself been dropped (see the next
section) — with it gone, `--reduced --disable-cascade` is now essentially
just original settings plus early-exit (provably lossless), so it's no
longer a meaningful isolation test, mainly a sanity check that early-exit
really is lossless. The more useful test going forward is a plain
`--reduced` run (cascade + early-exit only, full corpus) against the
Golf/layer-1 original baseline — that now isolates cascade's own effect on
the FC cleanly, which the original 30/69-match result couldn't, since the
corpus reduction was confounded with it at the time.

### Corpus-size reduction dropped — back to full batches

Discussed the actual point of `EFFECT_MEASUREMENT_BATCHES` (why it existed:
the real production job is 6 concepts × `MIDDLE_LAYERS`, and candidates
scale with how many layers are searched — running all 10 middle layers
together for one concept could mean 700-900+ candidates, and at 85 batches
that's plausibly 10+ hours for effect measurement alone, for one concept).
Weighed against the determinism-check finding above (the divergence is
real, not noise) and decided: correctness over speed. **Dropped** —
`EFFECT_MEASUREMENT_BATCHES` is no longer applied by `--reduced` in either
`discover.py` or `run_kaggle.py`; effect measurement always runs on the
full `wikipedia_content` corpus now, `--reduced` or not. Early-exit stays
(provably exact — free speedup, no accuracy cost, unlike the corpus
reduction was). The original 12-hour-Kaggle-session problem this was meant
to solve should be handled by spreading the full job across multiple
sessions via the checkpoint/resume already built into `get_feature_effect`
and `filter_features_by_mmlu`, not by measuring less data.

`EFFECT_MEASUREMENT_BATCHES=20` is kept in `reductions.py` as a documented,
unused-by-default historical value, in case a less-aggressive reduction
(e.g. 40 batches) is worth revisiting later.

**Next real test**: a plain `discover.py --concept Golf --layers 1 --reduced --debug-log-noop-edits`
run (cascade + early-exit, full corpus this time) diffed against
`golf_original.parquet` — this now cleanly isolates cascade's own
contribution to the earlier divergence, without the corpus reduction
confounding it. Not yet run.

## Requirements

CUDA GPU, PISCES's dependencies (`../requirements.txt`), and network/hub
access to download Gemma-2-2B-it and the GemmaScope MLP SAEs.
