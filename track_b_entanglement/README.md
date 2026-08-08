# Track B — Entanglement metrics

Implements the three cross-concept entanglement metrics, all measured in the
**dictionary space** the erasure method itself edits (PISCES's SAE feature
basis) rather than raw activation space within a single retain/forget split —
that's the central methodological difference from prior unlearning-difficulty
literature this project is testing.

Reads Track A's `artifacts/features/<concept>.parquet` files
(`schema.FeatureRecord` rows — every candidate feature considered, not just
the selected ones) and produces the `entanglement_*` columns of the combined
results table (`schema.ConceptResultRow`).

## Metrics (`entanglement_metrics.py`)

- **(a) `entanglement_cosine`** — mean pairwise cosine similarity between a
  concept's selected feature decoder directions and every other concept's.
- **(b) `entanglement_pullin_rate`** — fraction of candidate features that
  passed the automatic vocabulary-projection threshold (returned by
  `search_features`) but didn't survive filtering into the final selected set.
- **(c) `entanglement_token_overlap`** — Jaccard overlap between a concept's
  selected-feature top/bottom tokens and its designated near-domain concept's
  (see `NEAR_DOMAIN_PAIRS` in `entanglement_metrics.py` — a curated
  placeholder pairing among the 15 natural concepts; PISCES itself doesn't
  name a sibling concept anywhere, only per-concept "similar domain" QA text).
  Flagged as weak for Harry Potter, which has no good real-world sibling.

## Status

Works against whatever subset of concepts Track A has produced so far —
`compute_all()` skips concepts with no parquet file yet, so this can run
against partial (even mock) Track A output during development.
