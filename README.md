# Does dictionary-space entanglement predict unlearning difficulty?

A replication-and-extension of two LLM concept-erasure papers:

- **PISCES** (Gur-Arieh et al., EMNLP 2025, [arXiv:2505.22586](https://arxiv.org/abs/2505.22586))
  — erases a target concept from a language model's MLP weights by decomposing
  MLP weight vectors with a pretrained **Sparse Autoencoder (SAE)** per layer,
  finding the SAE features that correspond to the concept (via *vocabulary
  projection* — projecting each feature's decoder direction through the
  model's unembedding matrix), and negatively ablating those features in the
  weights. Vendored as a submodule at `pisces_ref/` (read-only reference,
  [github.com/yoavgur/PISCES](https://github.com/yoavgur/PISCES)).
- **EMBER** (Suslik et al., 2026, [arXiv:2606.03695](https://arxiv.org/abs/2606.03695))
  — a sibling method targeting token embeddings via Sparse Matrix
  Factorization instead of an SAE. Not touched in this phase.

Both papers report only aggregate erasure-quality metrics across many target
concepts. Neither asks *why* some concepts are easier or harder to erase than
others.

## Central hypothesis

A concept's difficulty to erase is predictable from how **entangled** its
representation is with other concepts' representations, measured in the same
**dictionary space** the erasure method itself edits (PISCES's SAE feature
basis) — as opposed to raw activation-space entanglement within a single
retain/forget split, which is what prior unlearning-difficulty literature
(Zhao et al. 2024, and related work) already measures.

This is a **replication-and-extension**, not a new discovery — "entanglement
predicts unlearning difficulty" is established elsewhere. The contribution
here is testing whether it holds when entanglement is measured (a) *across*
many distinct target concepts rather than within one retain/forget split, and
(b) in the *dictionary space* the edit itself operates in, which comes for
free as a byproduct of the erasure pipeline.

## Scope (fixed for this phase)

- **Concepts**: the 15 natural concepts in PISCES's own `data/cvs.json`
  (vendored at `data/cvs.json`) — Culture of Greece, Golf, Republic of
  Ireland, Ancient Rome, Baseball, Uranium, Suicide, Mass Shooting, Rape,
  Opioid, Harry Potter, Cannabis, Gambling, Gun, Pornography. Synthetic
  concept pairs are a later phase, gated on these results looking sound.
- **Erasure method**: PISCES only. EMBER's Sparse-MF pipeline is a later
  cross-disentangler extension.
- **Model**: Gemma-2-2B-it only. Llama-3.1-8B is a later generalization check.
- **Outcomes**: efficacy (post-erasure concept QA accuracy) and specificity
  (similar-domain accuracy, MMLU accuracy). Robustness (relearning-attack
  accuracy) is deferred.
- **Entanglement metrics**: all three —
  (a) cross-concept cosine similarity between concepts' selected SAE feature
  decoder directions,
  (b) filter pull-in rate — fraction of candidate features that pass the
  automatic vocabulary-projection threshold but aren't actually
  concept-specific,
  (c) activating-token / feature-token Jaccard overlap between a concept and
  its designated near-domain concept.

## The one real technical gap this project fills

PISCES's own repo does **not** include a runnable feature-*discovery*
pipeline. `pisces_ref/feature_finder.py::search_features` expects a
precomputed lookup table (`lls`, top/bottom vocabulary-projected tokens per
SAE feature) as an *input* — but nothing in the repo builds that lookup. The
one public demo (`erasing_harry_potter.ipynb`) only shows five features
hand-picked by the paper's authors, with no live discovery step.

`track_a_feature_discovery/vocab_projection.py` reconstructs that step: loads
each layer's pretrained SAE the same way `pisces_ref/editor.py`'s `SAEConfig`
does, projects every decoder direction through the model's unembedding
matrix, ranks top/bottom tokens per feature, and feeds that into PISCES's
existing, unmodified `search_features` + `filter_features_by_effect_and_activations`
+ `filter_features_by_mmlu`. (Two alternatives were checked and rejected: a
companion feature-descriptions repo from the same author, which only covers a
sampled subset of the dictionary, and Neuronpedia's bulk SAE exports, which
would add an external dependency.)

## Repo layout

```
/artifacts/                 # generated data (gitignored) -- per-feature parquets, results table
/data/                      # vendored copy of PISCES's cvs.json
/pisces_ref/                # PISCES, as a git submodule -- read-only reference
/track_a_feature_discovery/ # vocabulary-projection reconstruction (critical path)
/track_b_entanglement/      # the three entanglement metrics
/track_c_erasure_eval/      # per-concept erasure + evaluation
/track_d_analysis/          # joins B+C, Spearman correlations, plots
/schema.py                  # shared artifact schema every track imports from
```

See each track's own README for what it does and how to run it.

## Artifact schema

Defined once in `schema.py` (`FeatureRecord`, `ConceptResultRow`) — every
track imports from there instead of redefining the shapes locally, so the
format can't silently drift between tracks.

- **Per-feature** (`artifacts/features/<concept>.parquet`, one file per
  concept): every candidate feature considered for that concept, not just the
  final selected set (needed for the filter pull-in-rate metric). Maps
  directly onto PISCES's `editor.Feature(layer, id, neg, large)` — a row can
  be turned back into a `Feature` with no translation step.
- **Per-concept results** (`artifacts/results.parquet`, one combined table,
  one row per concept): `efficacy`, `specificity_simdomain`,
  `specificity_mmlu`, `entanglement_cosine`, `entanglement_pullin_rate`,
  `entanglement_token_overlap`.

## Work plan / track dependencies

- **Track A** (critical path) produces the per-feature artifacts everything
  else depends on. Until it has real output, B and C can work against mock
  `FeatureRecord` rows.
- **Track B** reads Track A's output, computes the three entanglement
  metrics.
- **Track C** runs erasure + evaluation. Can start immediately and
  independently of Track A — first validate against the five hardcoded Harry
  Potter features from the demo notebook (`run_erasure_eval.py --hardcoded-hp`).
- **Track D** joins B + C, computes Spearman correlations, plots.

Sanity-check order for Track A: run vocabulary-projection discovery against
just Harry Potter first and compare the recovered `selected=True` features
against the five hardcoded ones in `erasing_harry_potter.ipynb`, before
running across all 15 concepts.

## Setup

```
git submodule update --init --recursive   # pisces_ref/
pip install --break-system-packages -r requirements.txt
```

CUDA GPU required for anything touching the model (Tracks A and C). Track C
also needs a `GEMINI_API_KEY` env var (PISCES's `evals.GeminiEvaluator`
grades open-ended answers via Gemini).

## Explicitly out of scope for this phase (don't relitigate)

Synthetic concept pairs, EMBER, Llama-3.1-8B, robustness/relearning-attack
accuracy. Ask before expanding any of: metrics, concepts, method, model, or
outcome variables — scope is deliberately fixed and shouldn't drift without
the team agreeing first.
