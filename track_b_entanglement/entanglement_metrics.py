"""
Track B: the three cross-concept entanglement metrics, measured in the SAE
feature (dictionary) space PISCES itself edits -- as opposed to raw
activation-space entanglement within a single retain/forget split (what prior
unlearning-difficulty literature measures). Reads Track A's per-feature
parquet artifacts (schema.FeatureRecord) and produces the `entanglement_*`
fields of the combined results table (schema.ConceptResultRow).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema import ConceptResultRow, NATURAL_CONCEPTS, feature_artifact_filename  # noqa: E402

FEATURES_DIR = ROOT / "artifacts" / "features"

# Each concept's designated near-domain concept, for metric (c). PISCES itself
# only provides per-concept "similar domain" QA text (project_concepts.json's
# SimdomQA_* fields), not a named sibling concept among the 15 -- this pairing
# is a curated placeholder for the current phase and should get a
# domain-expert pass before being treated as ground truth.
#
# Updated for the Poison/Patriarchy/Homo Sapiens concept swap (Harry Potter,
# Culture of Greece, and Baseball removed -- see wikipedia_content_audit.md).
# Removing those three orphaned the pairings that pointed at them (Republic of
# Ireland and Ancient Rome both pointed at Culture of Greece; Golf pointed at
# Baseball); those three, plus the 3 new concepts, are the only entries that
# changed below -- every pairing among the 12 unchanged concepts is untouched.
NEAR_DOMAIN_PAIRS = {
    "Ancient Rome": "Homo Sapiens",  # was "Culture of Greece" (removed); both broad human-history/civilization topics
    "Republic of Ireland": "Ancient Rome",  # was "Culture of Greece" (removed); nation-with-deep-historical-continuity, same weak-ish quality as the original
    "Golf": "Gambling",  # was "Baseball" (removed); sports-betting is a real adjacent domain, extends the existing Gambling/Pornography "vice industries" cluster below
    "Uranium": "Gun",
    "Suicide": "Opioid",
    "Mass Shooting": "Gun",
    "Rape": "Mass Shooting",
    "Opioid": "Cannabis",
    "Cannabis": "Opioid",
    "Gambling": "Pornography",
    "Gun": "Mass Shooting",
    "Pornography": "Gambling",
    "Poison": "Opioid",  # new concept: opioid overdose is literally a poisoning mechanism -- close topical overlap
    "Patriarchy": "Rape",  # new concept: gender-based power structures and gender-based violence are closely studied together in the literature
    "Homo Sapiens": "Ancient Rome",  # new concept: mutual with Ancient Rome above -- both broad human-history/civilization topics
}


def _load(concept: str) -> pd.DataFrame:
    path = FEATURES_DIR / feature_artifact_filename(concept)
    if not path.exists():
        raise FileNotFoundError(
            f"No Track A output for {concept!r} at {path}. "
            "Run track_a_feature_discovery/discover.py first."
        )
    return pd.read_parquet(path)


def cosine_entanglement(concept_a: str, concept_b: str) -> float:
    """Metric (a): mean pairwise cosine similarity between concept_a's and
    concept_b's SELECTED feature decoder directions."""
    df_a = _load(concept_a)
    df_b = _load(concept_b)
    dirs_a = np.stack(df_a[df_a["selected"]]["decoder_direction"].to_numpy())
    dirs_b = np.stack(df_b[df_b["selected"]]["decoder_direction"].to_numpy())

    if len(dirs_a) == 0 or len(dirs_b) == 0:
        return float("nan")

    norm_a = dirs_a / np.linalg.norm(dirs_a, axis=1, keepdims=True)
    norm_b = dirs_b / np.linalg.norm(dirs_b, axis=1, keepdims=True)
    sims = norm_a @ norm_b.T
    return float(sims.mean())


def pullin_rate(concept: str) -> float:
    """Metric (b): fraction of candidate features that passed the automatic
    vocabulary-projection threshold (i.e. were returned by search_features)
    but didn't survive filtering into the final selected set -- candidates
    "pulled in" by the threshold without actually being concept-specific."""
    df = _load(concept)
    n_candidates = len(df)
    if n_candidates == 0:
        return float("nan")
    n_selected = int(df["selected"].sum())
    return (n_candidates - n_selected) / n_candidates


def token_overlap(concept: str) -> float:
    """Metric (c): Jaccard overlap between concept's selected-feature
    top/bottom tokens and its designated near-domain concept's selected-feature
    top/bottom tokens (see NEAR_DOMAIN_PAIRS).

    Returns NaN (doesn't raise) if either side's Track A output is missing --
    not just when NEAR_DOMAIN_PAIRS has no entry at all. Real risk, not
    hypothetical: half of REDUCED_CONCEPTS' own near-domain pairs point
    outside that 6-concept set (Golf -> Gambling, Poison -> Opioid,
    Homo Sapiens -> Ancient Rome are not among the 6; only Uranium -> Gun,
    Gun -> Mass Shooting, Mass Shooting -> Gun stay in-scope) -- a
    compute_all() run scoped to just REDUCED_CONCEPTS would otherwise crash
    on this metric for exactly the concepts whose pairing falls outside it,
    rather than reporting NaN for the ones the current run doesn't cover."""
    near_domain = NEAR_DOMAIN_PAIRS.get(concept)
    if near_domain is None:
        return float("nan")

    def token_set(df: pd.DataFrame) -> set[str]:
        selected = df[df["selected"]]
        tokens: set[str] = set()
        for row in selected.itertuples():
            tokens.update(row.top_tokens)
            tokens.update(row.bottom_tokens)
        return tokens

    try:
        tokens_a = token_set(_load(concept))
        tokens_b = token_set(_load(near_domain))
    except FileNotFoundError:
        return float("nan")

    if not tokens_a and not tokens_b:
        return float("nan")

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compute_all(concepts: list[str] | None = None) -> pd.DataFrame:
    """Compute all three metrics for each concept that has Track A output,
    using the mean pairwise cosine similarity against every other available
    concept for metric (a)."""
    concepts = concepts or NATURAL_CONCEPTS
    available = [c for c in concepts if (FEATURES_DIR / feature_artifact_filename(c)).exists()]

    rows = []
    for concept in available:
        others = [c for c in available if c != concept]
        cosine_scores = [cosine_entanglement(concept, other) for other in others]
        cosine_mean = float(np.nanmean(cosine_scores)) if cosine_scores else float("nan")

        # ConceptResultRow.to_dict() includes every schema field (asdict()), so
        # efficacy/specificity_* would come through as None here too -- keep only
        # the entanglement_* columns this track owns, or Track D's merge would hit
        # the exact same _x/_y collision this construction is meant to guard
        # against, just on the other three columns.
        row = ConceptResultRow(
            concept=concept,
            entanglement_cosine=cosine_mean,
            entanglement_pullin_rate=pullin_rate(concept),
            entanglement_token_overlap=token_overlap(concept),
        ).to_dict()
        rows.append({k: row[k] for k in ("concept", "entanglement_cosine", "entanglement_pullin_rate", "entanglement_token_overlap")})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = compute_all()
    print(df.to_string(index=False))
