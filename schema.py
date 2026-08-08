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

# The 15 natural concepts from PISCES's data/cvs.json (also vendored at data/cvs.json).
# Scope is fixed to exactly this list for the current phase.
NATURAL_CONCEPTS = [
    "Culture of Greece",
    "Golf",
    "Republic of Ireland",
    "Ancient Rome",
    "Baseball",
    "Uranium",
    "Suicide",
    "Mass Shooting",
    "Rape",
    "Opioid",
    "Harry Potter",
    "Cannabis",
    "Gambling",
    "Gun",
    "Pornography",
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
