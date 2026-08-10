"""
Unit tests for seed_tokens.py against small hand-written text snippets with a
known expected token set. No GPU, model, or real tokenizer required --
`is_single_token` is a plain Python predicate injected by the caller (in
discover.py it's backed by the real model; here it's a small fake vocabulary).
"""
from __future__ import annotations

from seed_tokens import (
    NEUTRAL_NEG_TOKENS,
    build_background_word_freq,
    derive_seed_tokens_for_concept,
    extract_seed_tokens,
    get_neg_toks,
)


def all_single_token(_: str) -> bool:
    return True


def make_single_token_vocab(allowed: set[str]):
    """is_single_token stand-in: only tokens in `allowed` (leading-space form)
    tokenize to a single token; everything else is treated as multi-token,
    exercising the fall-through-to-next-candidate path."""

    def _is_single_token(tok: str) -> bool:
        return tok in allowed

    return _is_single_token


def test_extract_seed_tokens_prefers_concept_specific_words_over_background():
    text = (
        "Hogwarts is a school. Harry Potter and Hermione Granger studied at "
        "Hogwarts. Harry Potter fought Voldemort at Hogwarts."
    )
    background = build_background_word_freq(
        [
            "This article, like every other article, mentions Hogwarts occasionally in passing.",
            "Hogwarts is referenced here too, alongside many other generic words repeated often.",
        ]
    )

    seed_tokens = extract_seed_tokens(text, all_single_token, background_word_freq=background, top_n=3)

    # "Harry"/"Potter" are concept-specific (absent from background) and
    # frequent in-article, so they should outrank "Hogwarts", which is
    # artificially inflated in the background corpus above.
    assert " Harry" in seed_tokens
    assert " Potter" in seed_tokens
    assert seed_tokens.index(" Harry") < seed_tokens.index(" Hogwarts") if " Hogwarts" in seed_tokens else True


def test_extract_seed_tokens_filters_stopwords():
    text = "The the the and and of of of to to Harry Harry Harry Harry"
    seed_tokens = extract_seed_tokens(text, all_single_token, top_n=5)

    assert seed_tokens == [" Harry"]


def test_extract_seed_tokens_skips_multi_token_candidates():
    text = "Quidditch Quidditch Quidditch Harry Harry"
    # Simulate " Quidditch" not being a single token in the target vocab, so
    # extraction should fall through to the next-best candidate.
    is_single_token = make_single_token_vocab({" Harry"})

    seed_tokens = extract_seed_tokens(text, is_single_token, top_n=5)

    assert seed_tokens == [" Harry"]
    assert " Quidditch" not in seed_tokens


def test_extract_seed_tokens_respects_top_n():
    text = "Alpha Alpha Alpha Beta Beta Gamma Delta"
    seed_tokens = extract_seed_tokens(text, all_single_token, top_n=2)

    assert len(seed_tokens) == 2
    assert seed_tokens[0] == " Alpha"  # highest raw frequency, no background supplied


def test_extract_seed_tokens_uses_most_common_surface_casing():
    text = "golf golf golf Golf is played on grass. golf clubs vary."
    seed_tokens = extract_seed_tokens(text, all_single_token, top_n=1)

    assert seed_tokens == [" golf"]


def test_derive_seed_tokens_for_concept_builds_background_from_other_rows():
    cvs = [
        {"Concept": "Harry Potter", "wikipedia_content": "Harry Potter and Hermione at Hogwarts. Harry fought Voldemort."},
        {"Concept": "Golf", "wikipedia_content": "Golf is a sport. Golf clubs and golf balls are used in golf."},
    ]

    seed_tokens = derive_seed_tokens_for_concept("Harry Potter", cvs, all_single_token, top_n=3)

    assert " Harry" in seed_tokens
    assert " Golf" not in seed_tokens


def test_derive_seed_tokens_for_concept_raises_on_unknown_concept():
    cvs = [{"Concept": "Golf", "wikipedia_content": "Golf golf golf."}]
    try:
        derive_seed_tokens_for_concept("Not A Concept", cvs, all_single_token)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown concept")


def test_get_neg_toks_returns_fixed_list():
    toks = get_neg_toks()
    assert toks == NEUTRAL_NEG_TOKENS
    # returned list must be a copy, not the module-level constant itself
    toks.append(" mutated")
    assert " mutated" not in NEUTRAL_NEG_TOKENS


def test_get_neg_toks_validates_against_is_single_token():
    def only_the_is_single(tok: str) -> bool:
        return tok == " the"

    try:
        get_neg_toks(is_single_token=only_the_is_single)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when neg_toks aren't all single tokens")
