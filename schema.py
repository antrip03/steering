"""
Shared artifact schema for the entanglement-predicts-unlearning-difficulty project.

Every track (A/B/C/D) imports from here instead of redefining these shapes locally,
so the per-feature and per-concept artifact formats can't silently drift between
tracks. See README.md for the project overview.

`layer` / `feature_id` / `neg` on FeatureRecord map directly onto PISCES's own
`editor.Feature(layer, id, neg, large)` dataclass (pisces_ref/editor.py) so a row
can be round-tripped into a `Feature` object with no translation step, for passing
into `unlearn_concept`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# The project's own finalized 15-concept set, vendored/constructed at
# data/project_concepts.json -- 12 concepts copied verbatim from PISCES's original
# data/cvs.json (Ancient Rome, Cannabis, Gambling, Golf, Gun, Mass Shooting, Opioid,
# Pornography, Rape, Republic of Ireland, Suicide, Uranium) plus 3 replacements
# (Poison, Patriarchy, Homo Sapiens) swapped in for Harry Potter, Culture of Greece,
# and Baseball -- see wikipedia_content_audit.md for why Harry Potter was dropped
# (its wikipedia_content was fan-fiction, not an encyclopedia article) and the
# project_concepts_audit report for the 3 new concepts' data-quality checks.
#
# data/cvs.json itself is left untouched as the original vendored reference --
# every track reads CVS_PATH below (data/project_concepts.json), not cvs.json
# directly, so there is exactly one place this path is configured.
CVS_PATH = Path(__file__).resolve().parent / "data" / "project_concepts.json"

NATURAL_CONCEPTS = [
    "Golf",
    "Republic of Ireland",
    "Ancient Rome",
    "Uranium",
    "Suicide",
    "Mass Shooting",
    "Rape",
    "Opioid",
    "Cannabis",
    "Gambling",
    "Gun",
    "Pornography",
    "Poison",
    "Patriarchy",
    "Homo Sapiens",
]

# Model is fixed to Gemma-2-2B-it for this phase.
MODEL_NAME = "gemma-2-2b-it"


@dataclass
class FeatureRecord:
    """One row of a per-concept feature-candidate parquet (artifacts/features/<concept>.parquet).

    Contains every candidate feature considered for the concept, not just the
    final selected set (`selected=True` subset) -- the full candidate pool is
    needed for the filter pull-in-rate entanglement metric (Track B, metric b).
    """

    concept: str  # matches cvs.json's "Concept" field
    layer: int  # matches PISCES's Feature.layer
    feature_id: int  # matches PISCES's Feature.id
    neg: bool  # matches PISCES's Feature.neg (sign)
    decoder_direction: list[float]  # float32[d], full precision, no downcasting
    top_tokens: list[str]  # top 50 positive-projection tokens
    bottom_tokens: list[str]  # top 50 negative-projection tokens
    mass_ratio_or_effect_score: float  # scalar from filter_features_by_effect_and_activations
    selected: bool  # did this feature survive filtering into the final FeatureCollection

    # PISCES's own removal criterion (see pisces_ref/feature_finder.py::
    # filter_features_by_effect_and_activations) is `pos_effect > 0 or
    # neg_effect < -2` -- a candidate whose pos_effect and neg_effect both sit
    # near zero survives by failing to trigger either removal condition, not
    # by showing any positive evidence of a real effect. mass_ratio_or_effect_score
    # above already holds pos_effect; neg_effect was computed by the same
    # filtering call but silently discarded before this schema existed to record
    # it. Persisting it lets a downstream consumer (or a human) judge candidate
    # strength directly -- e.g. max(abs(mass_ratio_or_effect_score),
    # abs(neg_effect_score)) as a magnitude floor -- without this project baking
    # in a specific threshold choice into `selected` itself, which stays a
    # faithful, unmodified mirror of PISCES's own criterion. Optional (defaults
    # to None) so existing FeatureRecord construction sites aren't forced to
    # supply it immediately.
    neg_effect_score: float | None = None

    def to_pisces_feature(self):
        """Round-trip into PISCES's editor.Feature dataclass (large defaults to False)."""
        from pisces_ref.editor import Feature

        return Feature(layer=self.layer, id=self.feature_id, neg=self.neg)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConceptResultRow:
    """One row of the combined per-concept results table (artifacts/results.parquet)."""

    concept: str

    # Outcome variables (Track C)
    efficacy: float | None = None  # post-erasure concept QA accuracy
    specificity_simdomain: float | None = None  # similar-domain QA accuracy
    specificity_mmlu: float | None = None  # MMLU accuracy

    # Entanglement metrics (Track B), measured in dictionary (SAE feature) space
    entanglement_cosine: float | None = None  # metric (a): cross-concept cosine similarity
    entanglement_pullin_rate: float | None = None  # metric (b): filter pull-in rate
    entanglement_token_overlap: float | None = None  # metric (c): activating-token Jaccard overlap

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_RECORD_FIELDS = [f for f in FeatureRecord.__dataclass_fields__]
CONCEPT_RESULT_FIELDS = [f for f in ConceptResultRow.__dataclass_fields__]
