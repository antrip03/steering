"""
Reconstructs PISCES's missing seed-token / neg_toks construction.

Neither `data/cvs.json` nor any other part of `pisces_ref` publishes a token
list for any concept -- Harry Potter's are only recoverable by reading the
signs-computation cell of `pisces_ref/erasing_harry_potter.ipynb`, and even
those aren't written down anywhere as a reusable artifact. This module makes
the derivation explicit, automatic, and documented instead of leaving it to
open-ended per-concept manual curation. Both choices below are real
methodological decisions, not incidental implementation details -- see this
track's README for the same explanation in one place for the team to review.

1. Seed tokens (`extract_seed_tokens` / `derive_seed_tokens_for_concept`):
   salient single-token words pulled out of a concept's `wikipedia_content`,
   ranked TF-IDF-style against a background corpus built from the other
   concepts' articles. This single list plays three roles in PISCES's own
   pipeline (all under the confusingly-overloaded name `pos_toks` in
   `feature_finder.py`/`editor.py`): seeding `search_features`'s
   vocabulary-projection match, driving `get_mlp_act_signs`, and the
   positive side of `filter_features_by_effect_and_activations`.

2. `neg_toks` (`NEUTRAL_NEG_TOKENS` / `get_neg_toks`): a small fixed set of
   high-frequency, semantically neutral English function words, reused
   identically across every concept. PISCES has no precedent for this
   argument at all (not even for Harry Potter) -- this is the simplest
   defensible default, kept as a single named constant so it's easy to swap
   out later if the Harry Potter sanity check suggests it matters.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

IsSingleToken = Callable[[str], bool]

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# A small, deliberately-incomplete stopword list -- just enough to keep
# generic high-frequency function words out of the ranking. It is NOT relied
# on to do the heavy lifting: the background-corpus TF-IDF division already
# downweights anything common across every concept's article; this only
# guards against a handful of extremely common words swamping concepts whose
# background count happens to be small.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "he",
    "she", "they", "his", "her", "their", "which", "who", "whom", "also",
    "such", "into", "than", "then", "there", "have", "has", "had", "not",
    "can", "could", "would", "will", "may", "might", "one", "two", "used",
    "use", "other", "some", "more", "most", "many", "each", "any", "all",
    "over", "after", "before", "between", "during", "about", "including",
    "known", "called", "often", "however", "both", "while", "when", "where",
}

DEFAULT_TOP_N = 8
MIN_WORD_LENGTH = 3

# Fixed, topic-agnostic default for `neg_toks`. filter_features_by_effect_and_activations
# checks that suppressing a candidate feature doesn't *also* tank the model's
# probability on tokens unrelated to the erased concept; these are chosen to
# be common enough that they're very likely single tokens in any BPE/SentencePiece
# vocabulary, and neutral enough to mean the same thing regardless of concept.
NEUTRAL_NEG_TOKENS = [" the", " and", " is", " of", " to", " a", " in", " that"]


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _word_freq(text: str) -> Counter:
    return Counter(w.lower() for w in _tokenize_words(text) if len(w) >= MIN_WORD_LENGTH)


def _surface_forms(text: str) -> dict[str, Counter]:
    """lowercase word -> Counter of the exact surface forms (casing) it
    appeared as in `text`. Tokenizers are case-sensitive, so when we test a
    candidate word for single-token-ness we want to try the casing it
    actually appears as (e.g. "Golf" at a sentence start vs. "golf"
    mid-sentence), not an arbitrary one."""
    forms: dict[str, Counter] = {}
    for w in _tokenize_words(text):
        lw = w.lower()
        if len(lw) < MIN_WORD_LENGTH:
            continue
        forms.setdefault(lw, Counter())[w] += 1
    return forms


def build_background_word_freq(background_texts: list[str]) -> Counter:
    """Background/document-frequency corpus for the TF-IDF-style score below:
    word -> number of *articles* (not raw occurrences) it appears in across
    `background_texts`. Counting article-presence rather than raw occurrences
    keeps one unusually long or repetitive background article from dominating
    the denominator for every word it happens to use a lot."""
    background: Counter = Counter()
    for text in background_texts:
        background.update(set(w.lower() for w in _tokenize_words(text) if len(w) >= MIN_WORD_LENGTH))
    return background


def extract_seed_tokens(
    wikipedia_content: str,
    is_single_token: IsSingleToken,
    background_word_freq: Counter | dict[str, int] | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[str]:
    """Extract up to `top_n` salient single-token seed words for a concept.

    Selection criterion: TF-IDF-style score = (frequency in
    `wikipedia_content`) / (1 + document frequency in `background_word_freq`),
    computed over stopword-filtered words. Candidates are tried
    highest-score-first (ties broken by raw frequency, then alphabetically,
    for determinism); the first `top_n` that tokenize to exactly one token
    (`is_single_token`, tested with a leading space -- e.g. " Harry" -- to
    match the target model's word-initial tokenization convention) are kept,
    in that ` Word` form, ready to pass straight into `search_features`,
    `get_mlp_act_signs`, and `filter_features_by_effect_and_activations`
    (all of which `assert`/require single-token strings).

    If `background_word_freq` is omitted, falls back to plain
    frequency-within-article ranking (still stopword-filtered) -- used for
    standalone calls/tests that don't have a background corpus handy.
    """
    freqs = _word_freq(wikipedia_content)
    surface = _surface_forms(wikipedia_content)
    background = background_word_freq or {}

    scored = []
    for word, count in freqs.items():
        if word in _STOPWORDS:
            continue
        bg = background.get(word, 0)
        score = count / (1 + bg)
        scored.append((score, count, word))

    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))

    seed_tokens: list[str] = []
    for _, _, word in scored:
        if len(seed_tokens) >= top_n:
            break
        best_surface = surface[word].most_common(1)[0][0]
        candidate = f" {best_surface}"
        if is_single_token(candidate):
            seed_tokens.append(candidate)

    return seed_tokens


def derive_seed_tokens_for_concept(
    concept: str,
    cvs: list[dict],
    is_single_token: IsSingleToken,
    top_n: int = DEFAULT_TOP_N,
) -> list[str]:
    """Convenience wrapper for the common case: builds the background corpus
    from every *other* concept's `wikipedia_content` in `cvs` (rows loaded from
    `schema.CVS_PATH`, one dict per concept with a `"Concept"` and
    `"wikipedia_content"` key, matching PISCES's original `cvs.json` row shape)
    and calls `extract_seed_tokens` for `concept`.
    """
    concept_row = next((row for row in cvs if row["Concept"] == concept), None)
    if concept_row is None:
        raise KeyError(f"Concept {concept!r} not found in cvs data")

    background_texts = [row["wikipedia_content"] for row in cvs if row["Concept"] != concept]
    background = build_background_word_freq(background_texts)

    return extract_seed_tokens(
        concept_row["wikipedia_content"],
        is_single_token,
        background_word_freq=background,
        top_n=top_n,
    )


def get_neg_toks(is_single_token: IsSingleToken | None = None) -> list[str]:
    """Returns the fixed `NEUTRAL_NEG_TOKENS` list, identical for every
    concept. If `is_single_token` is given, validates every entry against it
    and raises if any fail -- catches a tokenizer/model mismatch early rather
    than silently feeding a multi-token string into PISCES's
    `model.to_single_token`, which asserts on that."""
    if is_single_token is not None:
        bad = [tok for tok in NEUTRAL_NEG_TOKENS if not is_single_token(tok)]
        if bad:
            raise ValueError(
                f"NEUTRAL_NEG_TOKENS contains tokens that aren't single tokens for "
                f"this model's tokenizer: {bad}"
            )
    return list(NEUTRAL_NEG_TOKENS)
