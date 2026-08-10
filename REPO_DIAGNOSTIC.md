# Repo Diagnostic — All Four Tracks

**Status: post-fix pass.** This is an update to the original full-repo diagnostic below — Blocking #1–#3 and the
disputed `unlearn_concept` finding (Correctness risk #0) have been fixed and verified; see "Fix pass results" at
the top for what changed and how it was checked. The rest of the original report (Correctness risk #5/#6,
Informational #7–#12) is preserved for reference, with items updated in place where the fix pass touched them.

Full-repo scope: `track_a_feature_discovery/`, `track_b_entanglement/`, `track_c_erasure_eval/`,
`track_d_analysis/`, `schema.py`, and `pisces_ref/`. Method: AST-based import/signature/dataclass audits
(scripted, not spot-checked) plus targeted empirical checks (`data/cvs.json` field presence, synthetic
`pd.merge` repro, a shared fixture + integration tests exercising real code).

This pass explicitly did not investigate `wikipedia_content` narrative-vs-encyclopedic quality (on hold) and did
not run the Harry Potter sanity check. **HF authentication is still not configured** — no tokenizer/model/SAE
loads were attempted in either the original or fix pass.

**Latest pass (this update): both the dependency pins and the `HookedSAETransformer` bug below are now
APPLIED**, not just proposed/found. See "Applied — dependency pins" and "Applied — `HookedSAETransformer` fix"
below.

---

## Applied — dependency pins

`requirements.txt` now pins `numpy==1.26.4`, `pyarrow==15.0.2`, `datasets==2.21.0`, `transformers==4.52.4`,
`huggingface_hub==0.32.3`, `transformer_lens==2.15.4`, `sae_lens==5.10.5`, `peft==0.15.2`,
`dataclasses-json==0.6.7` — the version set researched and verified in the pinned-dependency proposal (era-matched
to `pisces_ref`'s own last commit, `2025-05-30`, and its notebook's recorded kernel Python version `3.12.5`;
resolves the confirmed NumPy 1.x/2.x ABI mismatch behind the earlier nondeterministic segfaults). `torch` is left
unpinned — not implicated in the crash, and the installed version already satisfies `transformer_lens 2.15.4`'s
`torch>=2.2` floor.

**Now installed and empirically verified, under a real Python 3.12 environment.** A fresh Python 3.12.10
interpreter (installed via `winget`, alongside the existing 3.13 install, not replacing it) and venv were
created outside the repo, and `pip install -r requirements.txt` was run for real.

- **First two attempts failed on an unrelated Windows path-length issue**, not a dependency problem: pip fully
  resolved all ~150 transitive packages with zero conflicts both times, but installation itself failed with
  `OSError: [Errno 2] No such file or directory` on one of `torch`'s deeply-nested internal header files, because
  the venv's own path (deep inside a long temp directory) plus torch's path pushed past Windows' 260-character
  `MAX_PATH` limit. Fixed by recreating the venv at a short path (`C:\pv312`) rather than enabling Windows Long
  Path support system-wide (a machine-level registry change, out of scope to make unilaterally).
- **Third attempt succeeded cleanly**: exit code 0, ~17 minutes (1033s), zero `ERROR`/conflict/incompatibility
  lines anywhere in the log. `pip freeze` confirms every one of the 9 pinned packages installed at *exactly* the
  pinned version (`numpy==1.26.4`, `pyarrow==15.0.2`, `datasets==2.21.0`, `transformers==4.52.4`,
  `huggingface-hub==0.32.3`, `transformer-lens==2.15.4`, `sae-lens==5.10.5`, `peft==0.15.2`,
  `dataclasses-json==0.6.7`); `torch==2.13.0` installed unpinned, satisfying the `>=2.2` floor.
- **The ABI fix itself is now confirmed, not just plausible:** the same `from transformers import
  PreTrainedModel; import datasets` reproduction that previously produced three different outcomes (instant
  segfault, a 2h39m hang, a 63s success) under the old unpinned environment was re-run **5 times as 5 independent
  processes** under this new environment. **All 5 succeeded**, exit code 0 every time, 12-33 seconds each — no
  crashes, no hangs, consistent timing. This is meaningfully different from the old environment's pattern, not
  just one more data point in the same noise.
- **Full 18-test suite re-run under this interpreter: 18/18 passed**, 29.36s, one unrelated harmless deprecation
  warning (`google.generativeai` package sunset notice).

**Left in place for reuse:** the Python 3.12.10 interpreter (`C:\Users\anshu\AppData\Local\Programs\Python\Python312`)
and the verified venv (`C:\pv312`) were not deleted after verification — rebuilding took ~30 minutes cumulative
across the three install attempts, and both are needed for any future real run once HF/model access is
configured. Fully reversible/removable if not wanted.

---

## Applied — `HookedSAETransformer` fix

### 6. [RESOLVED] `feature_finder.py:108` calls `model.run_with_cache_with_saes(...)`, which only exists on `HookedSAETransformer`

**Precise details, confirmed by reading the actual `sae_lens 5.10.5` wheel source (not inferred):**

- **Exact call site:** `pisces_ref/feature_finder.py:108`, inside `get_feature_activations`, reached via
  `filter_features_by_effect_and_activations(..., filter_by_act=True)` — which `discover.py` always passes as
  `True`:
  ```python
  cache = model.run_with_cache_with_saes(batch, saes=saes, return_type=None)[1]
  ```
- **Exact method:** `run_with_cache_with_saes` is defined only on `sae_lens.HookedSAETransformer`
  (`sae_lens/analysis/hooked_sae_transformer.py:53,183`), a subclass of `transformer_lens.HookedTransformer` —
  not on the base class, and not monkey-patched onto it anywhere (`sae_lens/__init__.py` and a full-package grep
  for `HookedTransformer.<name> =`/`setattr` both came up empty). Confirmed against the currently-installed,
  bleeding-edge `transformer_lens 3.7.0` too — `hasattr(transformer_lens.HookedTransformer,
  'run_with_cache_with_saes')` → `False`. True on every version checked; not a version-pinning problem.
- **`from_pretrained` is inherited unchanged:** `HookedSAETransformer` does not override `from_pretrained` — its
  `__init__` just calls `super().__init__(*model_args, **model_kwargs)` and adds an empty
  `self.acts_to_saes: dict[str, SAE] = {}` bookkeeping dict. Same signature, same `device=` kwarg, confirmed from
  source — no additional setup required to construct it.
- **No SAE pre-attachment step needed either:** `run_with_cache_with_saes`'s full body (read directly) is:
  ```python
  with self.saes(saes=saes, reset_saes_end=reset_saes_end, use_error_term=use_error_term):
      return self.run_with_cache(*model_args, return_cache_object=..., remove_batch_dim=..., **kwargs)
  ```
  It attaches and detaches the given SAEs itself via the `saes=` argument at call time — `feature_finder.py:108`'s
  existing call already passes `saes=saes`, so no separate `model.add_sae(...)` call was ever missing.
- **Exact construction sites, confirmed to be the only two in the repo** (grepped every `.py` file; every other
  `HookedTransformer` reference is a type annotation or `isinstance` check inside `pisces_ref/evals.py`, never a
  construction): `track_a_feature_discovery/discover.py:151` (pre-fix) and
  `track_c_erasure_eval/run_erasure_eval.py:139` (pre-fix) — both used the identical pattern
  (`from transformer_lens import HookedTransformer` then `HookedTransformer.from_pretrained(MODEL_NAME,
  device=args.device)`).

**Fix applied:** both files now do
```python
from sae_lens import HookedSAETransformer

# must be HookedSAETransformer: feature_finder.py's get_feature_effect calls
# run_with_cache_with_saes, which only exists on this subclass, not plain HookedTransformer.
model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=args.device)
```
at `discover.py:149-153` and `run_erasure_eval.py:137-141`. Confirmed via grep that no stale `HookedTransformer`
reference (import, type hint, `isinstance` check) remains in either file.

**Not yet empirically verified against a live model** — this fix is confirmed correct from source inspection
(three independent points: the method only existing on the subclass, `from_pretrained` being inherited unchanged,
and `run_with_cache_with_saes` being self-sufficient given `saes=`), but has not been run against Gemma-2-2B-it,
since that needs HF/model access this pass still doesn't have.

**Verified:** full test suite (18/18: Track A 12, B 2, C 2, D 2) passes after both changes. `discover.py`/
`run_erasure_eval.py` compile cleanly; the currently-installed `sae_lens` (still the old, unpinned
bleeding-edge version — `pip install` against the new pins hasn't run) confirms `HookedSAETransformer` is
importable and exposes `run_with_cache_with_saes`. Track C's test (`load_selected_features` against fixture
data, which imports the full `evals.py` chain) succeeded in 63s this run — a third distinct outcome for that
same import chain (previously: instant segfault, then a 2h39m slow success) — reinforcing, not contradicting,
the nondeterministic-ABI-conflict finding already on record above; not treated as new information about the fix
itself.

---

## Applied — `GeminiEvaluator` dead-model default fix

### 7. [RESOLVED] `pisces_ref/evals.py`'s `GeminiEvaluator` defaulted to a model with zero quota on this project's API key

**Found while troubleshooting Gemini quota errors during the concept-set-swap QA generation task** (unrelated
original purpose, but the same API key/account, so directly relevant): `GeminiEvaluator.__init__`
(`pisces_ref/evals.py:505`, pre-fix) defaulted to `model_name="models/gemini-2.0-flash"`.

**Confirmed dead, not just rate-limited** — a direct `gai.GenerativeModel("models/gemini-2.0-flash").generate_content(...)`
call (isolated from `evals.py`'s heavier, occasionally-flaky `transformers`/`datasets` import chain, to get a
clean signal) returns:
```
429 ResourceExhausted: Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 0, model: gemini-2.0-flash
```
`limit: 0` — this model has no quota allocation on this key at all, not a temporary cap. This is a live blocker
for **any** real Track C evaluation run: both call sites that construct a `GeminiEvaluator`
(`pisces_ref/feature_finder.py:243` inside `find_hps`, and `track_c_erasure_eval/run_erasure_eval.py:104`) do so
with no explicit `model_name`, so both would hit this the moment `evaluate_open_ended`/`_send_request` is
actually called — the same category of "confirmed-dead default in vendored code, blocks real runs" as the
earlier `gcg_multiple`/`os` fixes.

**Fix applied:** changed the default to `model_name="models/gemini-flash-latest"`, confirmed working with a live
call (`generate_content('Say OK')` → `"OK"`) using the same API key. Both construction sites pick this up
automatically since neither passes an explicit override.

**Not otherwise verified against a real Track C run** (`evaluate_open_ended`'s actual grading behavior,
response parsing, etc.) — only that the model itself accepts and responds to requests. That's a separate,
larger verification still gated on the standing HF/model-access boundary.

---

## Fix pass results

### 0. [RESOLVED — was a disputed finding] `unlearn_concept` call sites inside `pisces_ref/feature_finder.py` used stale kwargs

Re-audited `pisces_ref`'s own internal calls to `unlearn_concept` (not just track-code calls into `pisces_ref`,
which is what the original full-repo pass checked and is why it missed this). **Confirmed broken before the
fix:** `feature_finder.py:90,161,268` all called `unlearn_concept(model, concept, full=True, signed=True,
signs=signs, linscale=...)`, but `editor.py`'s actual signature is `unlearn_concept(model, concept, signs=None,
linscale=False)` — no `full`/`signed` params. This would have raised `TypeError: unlearn_concept() got an
unexpected keyword argument 'full'` at runtime from `get_feature_effect`, `filter_features_by_mmlu`, and
`find_hps` — i.e. it would have broken Track A's `discover.py` the moment it reached
`filter_features_by_effect_and_activations`/`filter_features_by_mmlu`, even after Blocking #1/#2 were fixed.

**Fix:** dropped `full=True, signed=True` from all three call sites, leaving `signs=`/`linscale=` as-is.

**Verified, AST-based (corrected from an earlier grep-only claim — see the follow-up verification pass):**
parsed `pisces_ref/editor.py` to extract `unlearn_concept`'s actual current parameter list
(`['model', 'concept', 'signs', 'linscale']`, no `**kwargs`), then parsed every `.py` file in the repo, resolved
every `ast.Call` node whose callee name is `unlearn_concept`, and checked each call site's keyword-argument
names against that parameter list and its positional-argument count against the parameter count. All 4 call
sites in the repo (`feature_finder.py:90,161,268`, `run_erasure_eval.py:112`) resolve cleanly — 2 positional
args plus only `signs=`/`linscale=` keywords, nothing else. This check is structurally different from (and
stronger than) a plain grep for the string `"full=True"`: it would also catch a reordered, renamed, or
multi-line stale kwarg that a literal-string grep could miss.

### 1. [RESOLVED] `pisces_ref/evals.py` imported a nonexistent `gcg_multiple` module at the top level

**Fix:** removed the two top-level `from gcg_multiple import ...` lines; moved them inside `get_gcg_suffix`
(the only function that uses them — dead code for this project phase, GCG/robustness is explicitly deferred).
`get_gcg_suffix_tl` calls `get_gcg_suffix`, so it's covered too.

**Verified, AST-based:** parsed `evals.py` and checked `tree.body` (top-level statements only, not nested) for
any `ImportFrom` node with `module == "gcg_multiple"` — none found. Separately walked the full tree for *every*
`gcg_multiple` `ImportFrom` at any nesting depth and identified its enclosing function by containment: both
remaining references (`run`, `GCGConfig`) resolve to inside `get_gcg_suffix`, confirming they're properly scoped
and not reachable at module-import time. Also confirmed by direct import attempts that got past the point they
used to fail at. See item 2 below for the full end-to-end import story, since a second, unrelated issue was hit
next.

### 2. [RESOLVED] `pisces_ref/evals.py:508` used `os.getenv` without importing `os`

**Fix:** added `import os` near the top of `evals.py`.

**Verified:** AST scan confirms `os` is now imported; `GeminiEvaluator.__init__` no longer references an
undefined name.

**New finding surfaced while verifying the full import chain (informational, not part of this fix pass's
scope):** getting `pisces_ref/evals.py` to actually import end-to-end in *this* sandbox required installing the
packages already listed in `requirements.txt` (they weren't installed here) — `openai`, `datasets`, `peft`,
`dataclasses-json`, `google-generativeai`, `transformer_lens`, `sae_lens` all installed cleanly, but a standalone
`python -c "from transformers import PreTrainedModel; import datasets"` **segfaults** in this environment
(confirmed via a bisected repro — isolates to the `transformers` → `datasets` import combination specifically, a
native ABI conflict between their compiled extensions, unrelated to anything in this project's code). Oddly,
the *same* import chain run via `pytest` (collecting `track_c_erasure_eval/test_run_erasure_eval.py`, see item 5)
did **not** segfault — it succeeded, but took **2 hours 39 minutes** to complete import + collection, which is
wildly impractical regardless of correctness. This is a pre-existing environment/dependency-version issue in
this specific sandbox (Windows, Python 3.13), not a code defect, and not something this fix pass's "pure-logic"
mandate covers — flagging clearly rather than spending further time pinning/downgrading packages to chase it
down. **Practical implication:** don't assume a quick `pip install -r requirements.txt` + import check will be
fast or crash-free in every environment; a real GPU box with a properly pinned dependency set (the environment
this project actually expects) may not hit this at all.

### 3. [RESOLVED] Track D's silent all-NaN merge bug

**Fix:** `track_d_analysis/correlation_analysis.py::build_combined_results` now selects only
`["concept", "efficacy", "specificity_simdomain", "specificity_mmlu"]` from `eval_df` before merging with
`entanglement_df`, eliminating the column-name collision that pandas was silently resolving via `_x`/`_y`
suffixes. A code comment now states explicitly: **Track B is the sole source of truth for the `entanglement_*`
columns** — Track C is never expected to populate them.

**Verified two ways:**
1. Re-ran the diagnostic's original synthetic `pd.merge` repro against the fixed logic — the real
   `entanglement_cosine`/`entanglement_pullin_rate`/`entanglement_token_overlap` values now survive the merge
   and `reindex` (previously always `NaN`).
2. New integration test `track_d_analysis/test_correlation_analysis.py::test_build_combined_results_preserves_real_entanglement_values`
   exercises the actual `build_combined_results()` function against a fixture (see item 5) and asserts every
   entanglement/outcome column is non-null. **Passes.**

### 4. [RESOLVED] Track B bypassed the schema when constructing output rows

**Fix:** `track_b_entanglement/entanglement_metrics.py::compute_all` now constructs a real
`ConceptResultRow(concept=..., entanglement_cosine=..., entanglement_pullin_rate=..., entanglement_token_overlap=...)`
and calls `.to_dict()`, so a future field rename/addition in `schema.py` now raises a constructor error here
instead of silently drifting out of sync — matching the letter of the original proposed fix.

**Regression caught and fixed during implementation, not just proposed:** `ConceptResultRow.to_dict()` uses
`dataclasses.asdict()`, which includes *every* schema field — so the naive version of this fix would have made
Track B's output also carry `efficacy`/`specificity_simdomain`/`specificity_mmlu` as `None`, which would have
**reintroduced the exact same `pd.merge` collision bug from item 3, mirrored onto the other three columns**
(Track B's `None`s colliding with Track C's real values this time). Fixed by trimming the constructed dict down
to only the four entanglement-relevant keys before appending it to the output rows. Both the schema-validation
benefit and the collision-avoidance are now covered by the same construction site.

**Verified:** `track_b_entanglement/test_entanglement_metrics.py` (new) passes, and
`test_build_combined_results_preserves_real_entanglement_values` (item 3) — which runs Track B's and Track D's
real code together — confirms no collision was reintroduced.

### 5. [DONE] Shared mock fixture + integration tests for Tracks B, C, D

Built `tests/fixtures.py`: synthetic `FeatureRecord` rows (3 fake concepts, varied candidate/selected-feature
mixes and a controlled token-overlap relationship between two of them — deliberately *not* uniform, so metrics
that should vary across concepts don't accidentally look constant and mask bugs behind a degenerate input) and
synthetic `ConceptResultRow`-shaped eval rows, plus `write_fake_features_dir`/`write_fake_eval_results` helpers
that write them in the exact parquet shape/naming convention the real tracks use.

New integration tests, all passing, all real code exercised (nothing mocked beyond the module-level path
constants, via `monkeypatch`):
- `track_b_entanglement/test_entanglement_metrics.py` — `compute_all()` against the fixture (2 tests).
- `track_d_analysis/test_correlation_analysis.py` — `build_combined_results()` + `compute_correlations()`
  against both fixtures together, through the real merge (2 tests). **This is the test that would have caught
  Blocking #3 automatically**, without a manual diagnostic pass.
- `track_c_erasure_eval/test_run_erasure_eval.py` — `load_selected_features()` (the output-shape logic that
  doesn't need a model), per the task's explicit scoping (2 tests). Subject to the import-chain caveat in item 2
  above: this test *does* pass, but importing it is what surfaced the `transformers`/`datasets` environment
  issue in the first place.

**Full suite: 18/18 passing** (`track_a_feature_discovery` 12, `track_b_entanglement` 2, `track_c_erasure_eval`
2, `track_d_analysis` 2).

---

## Per-track completeness summary (updated)

**Track A.** Unchanged from the original assessment except for what item 0/1/2 above unblock: `discover.py` can
now actually be imported (previously blocked by `gcg_multiple`), and once it reaches
`filter_features_by_effect_and_activations`/`filter_features_by_mmlu` it will no longer hit the stale
`unlearn_concept` kwargs from item 0. 12/12 unit tests still pass. `concept_tokens.json` ownership (Correctness
risk #5, original numbering, untouched by this pass per the explicit boundary) is still an open team decision.

**Track B.** Now empirically verified, not just statically clean — `compute_all()` runs against real fixture
data and produces correct, schema-validated output (item 4/5).

**Track C.** No longer blocked at import time (items 1/2) or at the `unlearn_concept` call inside
`filter_features_by_mmlu`/`get_feature_effect` (item 0) once it gets there via Track A. Its own
`load_selected_features` logic is now verified against fixture data (item 5). `evaluate_concept` itself (needs a
live model + `GEMINI_API_KEY`) remains untested here, per the explicit no-model/no-GPU boundary — that's expected,
not a gap this pass could close. `concept_tokens.json` coupling (Correctness risk #5) is unchanged and still open.

**Track D.** The central-hypothesis-blocking bug (Blocking #3) is fixed and empirically verified against real
fixture data flowing through the actual merge and correlation code — this track can now produce real, non-NaN
correlations once B and C have real data.

---

## Carried over unchanged from the original pass (not in this fix pass's scope)

- **Correctness risk #5 — Track A ↔ Track C seed-token/`neg_toks` ownership.** Explicitly out of scope for this
  fix pass per instruction ("a team decision between two proposed options, not yet made"). Still open.
- **Correctness risk (schema.py's `to_pisces_feature` dotted-import inconsistency, originally #4).** Not in the
  approved fix list; still latent (still never called anywhere in the repo, confirmed again via grep).
- **Informational #8 — `concept_tokens.json` orphaned ownership.** Unchanged; tied to Correctness risk #5.
- **Informational #9 — HF authentication.** Still unconfigured; still not attempted. Same "what needs it" list
  as before, with updated line numbers post-fix: `pisces_ref/editor.py:205`, `pisces_ref/evals.py:757-758`,
  `track_a_feature_discovery/discover.py:153`, `track_c_erasure_eval/run_erasure_eval.py:141`.
- **Informational #10 — `requirements.txt` diff.** Superseded by "Applied — dependency pins" above:
  `requirements.txt` is now fully pinned for the ABI-implicated packages. `gcg_multiple` remains correctly absent
  (lazy, dead-code-path-only import, deliberately not vendored or pinned).
- **Informational #12 — `data/cvs.json` completeness.** Unchanged, not re-verified this pass (nothing touched
  the data).

---

## What this pass did not do

No `wikipedia_content` quality investigation. No Harry Potter sanity check. No model/SAE/gated-tokenizer load.
**No `pip install` against the newly-pinned `requirements.txt`** — the pins are applied to the file but not
installed or empirically re-tested (that's explicitly gated on the team's Python-3.12 decision). No verification
of the `HookedSAETransformer` fix against a live model (confirmed correct from source only). Did not touch the
`concept_tokens.json` ownership question. Waiting for review before the Harry Potter sanity check, a real
`pip install` of the pinned set, or any other model-touching run.
