"""
Shared synthetic fixture data for cross-track integration tests (Tracks B, C, D).

Nothing here touches a model, GPU, or HF access. These are hand-built
`FeatureRecord` / `ConceptResultRow` rows, shaped exactly like real Track A /
Track C output, used to exercise the real downstream code (Track B's metrics,
Track C's output-shape logic, Track D's merge/correlation logic) end-to-end
without a real discovery or erasure run.

This is the fixture REPO_DIAGNOSTIC.md (Informational #7) flagged as missing --
its absence is why the Track D merge bug (Blocking #3) went unnoticed by
anything short of a manual diagnostic pass.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import ConceptResultRow, FeatureRecord, feature_artifact_filename  # noqa: E402

FAKE_CONCEPTS = ["Fake Concept Alpha", "Fake Concept Beta", "Fake Concept Gamma"]

D_MODEL = 8  # small synthetic decoder-direction dimensionality, arbitrary but fixed


def _direction(seed: int) -> list[float]:
    """Deterministic, distinct-but-related synthetic decoder directions --
    just needs to be a fixed-length float vector for cosine similarity to run
    against, not anything semantically meaningful."""
    return [math.sin(seed + i) for i in range(D_MODEL)]


# Per-concept (selected flag) patterns -- deliberately different pull-in rates
# (1/3, 3/4, 0/2) so metrics that vary across FAKE_CONCEPTS aren't accidentally
# constant (a constant input makes scipy's spearmanr legitimately return NaN,
# which would look like -- but isn't -- the merge bug this fixture guards
# against).
_SELECTED_PATTERNS = [
    (True, False, True),
    (True, False, False, False),
    (True, True),
]


def make_feature_records(concept: str, seed: int, shared_token: str | None = None) -> list[dict]:
    """Synthetic FeatureRecord rows for one concept, matching schema.FeatureRecord
    exactly. Row count and selected/not-selected mix vary by `seed` (1-indexed,
    matching _SELECTED_PATTERNS) so per-concept metrics aren't all identical.
    `shared_token`, if given, is added to every selected row's top_tokens --
    used to give a controlled, nonzero token_overlap between specific concept
    pairs (see write_fake_features_dir), so that metric isn't constant across
    FAKE_CONCEPTS either."""
    pattern = _SELECTED_PATTERNS[(seed - 1) % len(_SELECTED_PATTERNS)]
    rows = [
        FeatureRecord(
            concept=concept,
            layer=1 + i * 3,
            feature_id=seed * 100 + i,
            neg=(i % 2 == 0),
            decoder_direction=_direction(seed + i),
            top_tokens=[f" tok{seed}{i}a"] + ([shared_token] if selected and shared_token else []),
            bottom_tokens=[f" neg{seed}{i}a"],
            mass_ratio_or_effect_score=-0.5 if selected else 0.1,
            selected=selected,
        )
        for i, selected in enumerate(pattern)
    ]
    return [r.to_dict() for r in rows]


def make_concept_result_row(concept: str, efficacy: float, sim: float, mmlu: float) -> dict:
    """Shaped like Track C's real output: efficacy/specificity_* populated,
    entanglement_* left at the schema default (None) -- exactly how
    `dataclasses.asdict()` renders a ConceptResultRow Track C never touches
    the entanglement fields of."""
    return ConceptResultRow(
        concept=concept, efficacy=efficacy, specificity_simdomain=sim, specificity_mmlu=mmlu,
    ).to_dict()


def write_fake_features_dir(dir_path: Path) -> None:
    """Writes one feature_artifact_filename(concept) parquet per FAKE_CONCEPTS
    into dir_path, matching the exact naming convention Track A's discover.py
    uses for the official (all-layers, original-settings) run. Alpha and Beta
    (index 0, 1) share one selected-feature token, so their (mutual, per
    NEAR_DOMAIN_PAIRS in the tests that use this) token_overlap is nonzero;
    Gamma (index 2) shares nothing with either, so its overlap is 0 -- keeps
    entanglement_token_overlap from being constant across FAKE_CONCEPTS."""
    import pandas as pd

    shared_tokens = [" shared_ab", " shared_ab", None]
    dir_path.mkdir(parents=True, exist_ok=True)
    for i, concept in enumerate(FAKE_CONCEPTS):
        df = pd.DataFrame(make_feature_records(concept, seed=i + 1, shared_token=shared_tokens[i]))
        out = dir_path / feature_artifact_filename(concept)
        df.to_parquet(out, index=False)


def write_fake_eval_results(path: Path) -> None:
    """Writes a single erasure_eval_results.parquet-shaped file covering all
    FAKE_CONCEPTS, in the same shape Track C's run_erasure_eval.py writes."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        make_concept_result_row(FAKE_CONCEPTS[0], efficacy=0.05, sim=0.82, mmlu=0.60),
        make_concept_result_row(FAKE_CONCEPTS[1], efficacy=0.12, sim=0.75, mmlu=0.58),
        make_concept_result_row(FAKE_CONCEPTS[2], efficacy=0.03, sim=0.90, mmlu=0.61),
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
