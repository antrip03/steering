# Design proposal: batching candidate-feature evaluation in `get_feature_effect` / `filter_features_by_mmlu`

**Status: proposal only, not implemented.** Per the remediation plan's own instruction (Phase 4.4): design
first, reviewed, before any implementation. This documents the mechanism as it actually works (read from
`pisces_ref/editor.py` and `pisces_ref/feature_finder.py`, not assumed), the batching opportunity, an honest
assessment of expected speedup, and the risks — so the tradeoff can be evaluated before committing real
engineering time to it.

## 1. The mechanism, as it actually works

`unlearn_concept(model, concept, signs, linscale)` (`editor.py`) groups `concept.features` by layer, then calls
`steer_features`, which for **each layer** calls `get_hswaps_full_signed` **once**, passing the *entire* list of
features assigned to that layer. That function computes a set of `(row_index, new_row_vector)` swaps —
already batched across every feature in the concept for that layer, in one call. `replace_mlp_rows` then clones
the layer's `W_out`, applies every swap, and does one `model.blocks[layer].mlp.W_out.set_(changed)` — a single
in-place edit, regardless of how many features are being unlearned together.

**This means `steer_features`/`replace_mlp_rows` are not the bottleneck.** The actual one-at-a-time cost lives
one level up, in `feature_finder.py::get_feature_effect`:

```python
for i in _tqdm(range(0, len(forget_set), batch_size)):
    batch = forget_set[i:i+batch_size]
    clean_logits = model(batch).softmax(dim=-1)
    for feature in features:                      # <-- one candidate at a time
        f_concept = Concept(name=..., features=[feature])
        with unlearn_concept(model, f_concept, signs=signs, linscale=...):
            logits = model(batch).softmax(dim=-1)  # <-- full forward pass, per feature, per batch
        ...
```

This loop constructs a **single-feature** `Concept` and does a full forward pass under that one ablation, for
every candidate feature, for every text batch. `filter_features_by_mmlu` has the identical structure: one
`unlearn_concept` + one `evaluate_mmlu` call per surviving feature, in a Python loop.

**Important, and easy to miss:** this per-feature loop isn't an incidental inefficiency — it's inherent to what
the function measures. The whole point is each candidate feature's *individual, marginal* effect on model
output. Ablating features A and B together and measuring the combined effect doesn't tell you A's effect alone;
you need a separate forward pass with only A ablated. The redesign below doesn't eliminate that requirement — it
changes *how* those N separate measurements get computed (batched vs. sequential), not *how many* are needed.

## 2. The batching opportunity

Standard batched inference already runs every item in a batch through the **same** shared weights. The obstacle
here is that each candidate feature needs a **different** `W_out` at the edited layer. The fix: expand the batch
dimension by N (the number of candidates being measured together), and at exactly one hook point — where the
edited layer's MLP down-projection applies `W_out` — replace the single shared matmul with a batched one against
N different weight variants (one per candidate), using `torch.einsum` or `torch.bmm`:

- Stack the N per-feature `W_out` variants (each already computed by `get_hswaps_full_signed`, just not yet
  applied) into one `[N, d_mlp, d_model]` tensor.
- Repeat the input text batch N times along the batch dimension (or interleave — either ordering works as long
  as it's tracked consistently).
- At the edited layer's `hook_mlp_out` (or wherever `W_out` is applied), replace the normal single matmul with a
  grouped one: each of the N slices of the (now N×batch-sized) activation tensor gets multiplied by its own
  weight variant.
- Every other layer (everything before and after the edited one) needs no change at all — they already handle
  arbitrary batch sizes with shared weights, which is exactly what "N independent streams" looks like from their
  perspective.

This requires either a custom hook replacing the MLP down-projection at the edited layer only, or (more
invasively) patching `HookedTransformer`'s forward pass at that point. A custom hook is the lower-risk option —
`transformer_lens` is built around exactly this kind of hook-based intervention, and it keeps the change
localized to one layer's computation rather than touching the model's forward pass generally.

## 3. Scope: does this apply to `filter_features_by_mmlu` too?

Yes, same structure (`unlearn_concept` + `evaluate_mmlu`, once per surviving feature). Same batching approach
applies — expand the MMLU question batch by N, apply N weight variants at the edited layer's hook, single pass
through the rest of the model. The cost profile differs (a fixed ~100-300 question set per feature vs. a
text corpus scanned in windows), but the mechanism is identical, so this doesn't need a separate design.

## 4. Correctness validation plan (before trusting any speedup number)

1. Run both the current sequential implementation and the batched version on the same real input: the 74 actual
   Golf/layer-1 candidates already produced by a real run.
2. Diff numerically with `torch.allclose` at a tolerance appropriate to the model's dtype (bf16 has much
   coarser precision than fp32 — expect small floating-point differences from reordered summation in the
   batched matmul's reduction, which is normal and not a bug).
3. **Specifically check filtering *decisions*, not just raw numbers** — `filter_features_by_effect_and_activations`
   and `filter_features_by_mmlu` both threshold continuous values into pass/fail. A feature sitting near a
   threshold could flip between passed/failed purely from floating-point reordering noise, even if the
   underlying numbers are "close enough" by `allclose`'s standard. Report any features where batched and
   sequential disagree on the final selected/not-selected outcome, not just where the intermediate scores differ.
4. Only trust the batched implementation once both checks pass on real data — not synthetic/toy inputs, since
   the concern here (numerical drift near a threshold) is specifically about behavior with the real distribution
   of effect scores.

## 5. Expected speedup — honest assessment, not a promise

**This is the part worth being careful about before committing engineering time.** Batching N sequential forward
passes into one N×-batch-sized forward pass does **not reduce total FLOPs** — a 2B-parameter model doing N
separate passes computes the same total amount of matrix arithmetic as one pass over an N-times-larger batch
(for an early-layer edit like layer 1, essentially the entire rest of the 26-layer model gets recomputed per
candidate either way, batched or not). So this is **not** a "close to Nx, compute-bound" situation in the sense
of doing less work.

The actual source of any speedup is **overhead amortization and hardware utilization**: one large batched matmul
call vs. N small sequential ones pays fixed per-call/per-kernel-launch overhead once instead of N times, and
gets better cache/memory-bandwidth utilization from processing more data per call. Whether this yields a
meaningful speedup depends on how overhead-bound the *current* sequential implementation actually is — something
that needs to be **measured**, not assumed. Plausible range based on typical batching gains for workloads like
this: **3-10x**, not a full Nx. Could be less on CPU if the current per-call overhead is already small relative
to the matmul cost itself (this model's forward pass is substantial enough that overhead may not dominate).

**Do not schedule a multi-day GPU run assuming this redesign will cut it by 10-40x without measuring first** —
build a small prototype (e.g., batch just the 74 real Golf/layer-1 candidates together) and time it against the
current sequential version before deciding whether the full implementation is worth it.

## 6. Risks

1. **Memory scales with N.** Each weight variant is one `[d_mlp, d_model]` matrix — for Gemma-2-2B in bf16,
   ~42.5MB per variant. Batching all 74 Golf/layer-1 candidates together: ~3.1GB, fine. But the unrestricted,
   all-26-layer case could have on the order of ~1,900 candidates per concept (extrapolated, unverified — see
   the main report's §7) — batching all of them simultaneously would need tens of GB, infeasible. **The
   redesign must batch in bounded groups** (e.g., process one layer's candidates together, or chunk into groups
   of a fixed size), not attempt to batch everything at once. This adds real complexity: group-size selection,
   and making sure grouping doesn't itself become a source of subtle bugs (e.g., silently dropping a candidate
   at a group boundary).
2. **Floating-point drift affecting filtering decisions near a threshold** (see §4.3) — needs explicit checking,
   not just an `allclose` pass on raw scores.
3. **Hook correctness is a real engineering risk.** Getting the batch-expansion and per-variant weight
   application right, and confirming it only affects the intended layer's computation (not silently changing
   behavior anywhere else in the model), needs careful testing — this touches PISCES's actual forward-pass
   mechanics, a meaningfully bigger and more invasive change than any fix applied so far (which were all either
   config/device fixes or one-time non-forward-pass computations).
4. **This changes vendored (`pisces_ref`) code more deeply than anything in the fork so far.** Everything
   currently on the `steering-fixes` branch preserves PISCES's original algorithm exactly and only fixes things
   that were straightforwardly broken (crashes, wrong defaults, missing safety wrappers). This redesign is a
   genuine algorithmic restructuring, not a bugfix — worth treating with a correspondingly higher review bar.

## 7. Recommendation

Build a small, throwaway prototype first: batch the 74 real Golf/layer-1 candidates using the approach in §2,
validate correctness per §4, and measure the actual speedup before deciding whether to invest in a full,
production-quality implementation (with proper group-size handling per Risk 1, applied to both
`get_feature_effect` and `filter_features_by_mmlu`). This keeps the cost of finding out "is this worth it" small
relative to the cost of building the full thing on spec.
