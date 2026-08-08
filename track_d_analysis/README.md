# Track D — Correlation analysis

Joins Track B's entanglement metrics and Track C's efficacy/specificity
results on `concept`, producing the final combined results table
(`schema.ConceptResultRow`, one row per concept, all six metric/outcome
columns). Then computes Spearman correlations between each of the three
entanglement metrics and each of the three outcome variables, and scatter-plots
each pair.

This is where the project's central hypothesis actually gets tested: whether
dictionary-space entanglement (computed across many distinct target concepts)
predicts erasure difficulty the way prior literature's activation-space,
single-retain/forget-split entanglement does.

## Inputs

- `artifacts/erasure_eval_results.parquet` (Track C)
- Track A's `artifacts/features/*.parquet`, indirectly, via Track B's
  `entanglement_metrics.compute_all()`

## Outputs

- `artifacts/results.parquet` — the combined per-concept table
- `artifacts/correlations.csv` — one row per (entanglement metric, outcome)
  pair: Spearman rho, p-value, n
- `artifacts/plots/*.png` — one scatter plot per pair, points labeled by
  concept

## Status

Only meaningful once Track C has produced at least a few concepts' worth of
`efficacy`/`specificity_*` and Track B has produced the matching
`entanglement_*` columns for the same concepts (`build_combined_results()`
does an outer join, so partial data won't crash — but correlations need at
least 3 non-null pairs per (metric, outcome) to compute a rho at all).
