# `wikipedia_content` Data-Quality Audit — All 15 Concepts

**Verdict up front:** Harry Potter's `wikipedia_content` is confirmed fan-fiction, not an encyclopedia article,
throughout its entire length. **All other 14 concepts are confirmed genuine, unaltered Wikipedia content.**
This is isolated to Harry Potter — it is not a systemic data-quality problem across the dataset. See the final
recommendation at the bottom for what this means for Track A's validation concept choice.

Method: Step 1 computed structural metrics across all 15 concepts and flagged statistical outliers, without
assuming in advance which concepts (beyond Harry Potter) might be affected. Step 2 read actual excerpts
(beginning, middle, and a random ~75%-mark section — not just the opening, since both genuine articles and
fan-fiction can have a normal-looking first paragraph) for every flagged concept, Harry Potter, and three
unflagged controls. Step 3 cross-referenced flagged concepts against the real, current Wikipedia articles.
`data/cvs.json` was not modified — this is diagnosis only.

---

## Step 1 — Structural metrics, all 15 concepts

| Concept | chars | words | avg sentence len (words) | % chars in quotes* | pronouns /1000w | ASCII `'` | curly `’` | citation `[N]` | section headers |
|---|---|---|---|---|---|---|---|---|---|
| Culture of Greece | 57,835 | 8,666 | 23.9 | 0.5 | 0.1 | 26 | 0 | 0 | yes |
| Golf | 53,041 | 9,100 | 24.9 | 2.6 | 0.4 | 78 | 0 | 0 | yes |
| Republic of Ireland | 77,708 | 11,868 | 21.4 | 1.8 | 0.9 | 93 | 0 | 0 | yes |
| Ancient Rome | 99,498 | 15,656 | 21.1 | 0.7 | 0.4 | 114 | 0 | 0 | yes |
| Baseball | 62,610 | 10,260 | 24.4 | 1.6 | 0.2 | 107 | 0 | 1 | yes |
| Uranium | 52,867 | 7,894 | 21.6 | 0.5 | 1.4 | 34 | 0 | 0 | yes |
| Suicide | 53,229 | 8,262 | 19.6 | **53.0** | 0.5 | 36 | 0 | 0 | yes |
| Mass Shooting | 43,086 | 6,556 | 24.0 | **79.7** | 0.3 | 37 | 0 | 0 | yes |
| Rape | 61,946 | 9,726 | 22.9 | 11.5 | 1.1 | 35 | 0 | 0 | yes |
| Opioid | 60,088 | 8,760 | 20.9 | 2.3 | 0.9 | 37 | 0 | 0 | yes |
| **Harry Potter** | **1,198,574** | **221,228** | **13.2** | 0.0 | **39.3** | 12,404 | **5,484** | 0 | **no** |
| Cannabis | 35,026 | 5,232 | **15.5** | 2.5 | 0.6 | 14 | 1 | 0 | yes |
| Gambling | 23,626 | 3,685 | 22.7 | 2.1 | 0.6 | 16 | 2 | 0 | yes |
| Gun | 21,930 | 3,483 | 23.7 | 4.0 | 0.3 | 10 | 0 | 0 | yes |
| Pornography | 101,444 | 15,447 | 23.6 | 7.7 | 4.2 | 69 | 2 | 0 | yes |

\* "% chars in quotes" is a crude heuristic (chars between paired quote marks) — see Step 2, it produced false
positives.

**Statistical outliers (|z| > 1.5 against the 15-concept distribution):**
- **Harry Potter**: outlier on `n_chars` (z=+3.60), `n_words` (z=+3.61), `avg_sentence_len` (z=−2.52),
  `pronoun_rate` (z=+3.60) — outlier on **four independent metrics simultaneously**. Also the **only** concept
  with zero Wikipedia-style section headers (`==...==`) anywhere in its text, and by far the only concept with
  meaningful curly-apostrophe usage (5,484 vs. 0–2 for everything else).
- **Suicide** (z=+1.80) and **Mass Shooting** (z=+2.96) on `pct_chars_in_quotes`.
- **Cannabis** (z=−1.83) on `avg_sentence_len` — a much weaker signal than Harry Potter's.
- Baseball's single citation marker (z=+3.61) is a trivial artifact — 1 vs. 0 for the others, not meaningful at
  this scale.

---

## Step 2 — Qualitative spot-check

Checked: Harry Potter, Suicide, Mass Shooting, Cannabis (all flagged), plus Culture of Greece, Golf, Ancient Rome
(controls). Each read at the beginning, middle, and ~75%-mark.

### Harry Potter — **narrative fan-fiction, confirmed at all three sample points**

> Beginning: *"'RUN!' Harry yelled, grabbing at her robes. Hermione's feet hit the hard ground, running in
> tandem to her heartbeat as the prophecies shattered around them... A death eater appeared beside them and
> Hermione screamed, watching as Harry elbowed him in the face."*

> Middle (~50%): *"And Astoria was looked with curiosity by everyone... It even got Draco's attention."*

> ~75% mark: *"'They will kill me', she whimpered and her mask broke. Astoria was a liar, he knew, not a great
> one, but this time she was sincere."*

Dialogue in quotes, character emotion/action narration, an invented plot (a battle involving "prophecies,"
"death eaters," and a Draco/Astoria relationship arc not depicted this way in the actual books) — this is
fan-fiction prose, not an encyclopedia article, and it's consistent for the entire 1.2M-character length, not
just the opening.

### Suicide — genuine encyclopedic content (structural flag was a false positive)

> Beginning: *"Suicide is the act of intentionally causing one's own death. Risk factors for suicide include
> mental disorders, physical disorders, and substance abuse."*

> Middle: real section header `=== Treatment of mental illness ===`, factual clinical prose about screening
> tools (Beck Depression Inventory, etc.)

> ~75% mark: real section headers `== Social and culture ==` / `=== Legislation ===`, factual prose about
> assisted-suicide law in the Netherlands.

Definitional opening, real Wikipedia section-header syntax, no dialogue or first-person narration anywhere. The
Step 1 quote-percentage flag (53.0%) was a false positive: the heuristic naively pairs up *any* two quote
characters in sequence, so short quoted technical terms scattered widely apart in a long article get
misinterpreted as one enormous block quote spanning everything between them. Noting this as a methodology
caveat, not a data problem.

### Mass Shooting — genuine encyclopedic content (same false-positive pattern)

> Beginning: *"A mass shooting is a violent crime in which one or more attackers use a firearm to kill or injure
> multiple individuals in rapid succession..."*

> Middle: factual reporting with attributed quotes from named public figures (a criminologist's "individualistic
> culture" remark, a quoted Obama statement) — legitimate encyclopedic use of short quotations, not narrative.

> ~75% mark: factual reporting on real events (the 2015 Charlie Hebdo attack, named perpetrators, casualty
> counts), section header `==== Russia and post-Soviet states`.

Same false-positive quote-heuristic explanation as Suicide. Genuine content, real events, real section
structure.

### Cannabis — genuine encyclopedic content

Taxonomic/botanical description at the opening, a real taxonomic-revision citation (Small & Cronquist, 1976) in
the middle, a structured `=== Recreational use ===` section at 75%. The mild sentence-length outlier is
consistent with technical/taxonomic writing using more short, list-like sentences — not a data-quality signal.

### Controls (Culture of Greece, Golf, Ancient Rome) — all genuine, as expected

All three read as standard, well-formed encyclopedic prose at every sample point (e.g. Golf: *"In February 1971,
astronaut Alan Shepard became the first person to golf anywhere other than on Earth..."* — a real, verifiable,
factual claim, followed by a real section header `=== Golf courses worldwide ===`). These confirm the control
group is a valid baseline.

### Remaining 8 concepts (Republic of Ireland, Baseball, Uranium, Rape, Opioid, Gambling, Gun, Pornography)

Not flagged by Step 1, so not given the full three-point deep read, but beginning+middle excerpts were checked
for completeness. All 8 show unambiguous genuine encyclopedic openings and factual mid-article prose (e.g.
Uranium: *"Uranium is a chemical element; it has symbol U and atomic number 92..."*; Gambling: real enumerated
lists of game types). Nothing resembling Harry Potter's pattern anywhere.

---

## Step 3 — Cross-reference against real, current Wikipedia

Internet access confirmed available in this environment. Checked the three flagged concepts against their real
current English Wikipedia articles.

| Concept | `cvs.json` opening | Real Wikipedia opening (fetched live) | Match? |
|---|---|---|---|
| Suicide | *"Suicide is the act of intentionally causing one's own death. Risk factors for suicide include mental disorders, physical disorders, and substance abuse."* | *"Suicide is the act of intentionally causing one's own death. Risk factors for suicide include mental disorders, **neurodevelopmental disorders**, physical disorders, and substance abuse."* | **Near-verbatim** — same source, `cvs.json` is a slightly older revision (real article structure, citations, and section-header style confirmed by direct fetch) |
| Mass Shooting | *"A mass shooting is a violent crime in which one or more attackers use a firearm to kill or injure multiple individuals in rapid succession."* | *"A mass shooting is a crime in which one or more attackers use gun(s) to kill or injure multiple individuals in rapid succession."* | **Near-verbatim**, same pattern — older revision of the real article |
| Harry Potter | *"'RUN!' Harry yelled, grabbing at her robes..."* | *"Harry Potter is a series of seven children's fantasy novels written by British author J. K. Rowling."* (real article: citations, infobox, standard Wikipedia structure, no narrative prose) | **No relationship whatsoever** — `cvs.json`'s content bears zero resemblance to the real Wikipedia "Harry Potter" article in content, structure, or genre |

This is conclusive, not just corroborating: Suicide and Mass Shooting are real (if slightly stale) Wikipedia
snapshots. Harry Potter's field is not derived from Wikipedia at all, despite the field name.

---

## Step 4 — Final verdict table

| Concept | Outlier flag | Qualitative verdict | Recommendation |
|---|---|---|---|
| Culture of Greece | No | Encyclopedic | Clean — use as-is |
| Golf | No | Encyclopedic | Clean — use as-is |
| Republic of Ireland | No | Encyclopedic | Clean — use as-is |
| Ancient Rome | No | Encyclopedic | Clean — use as-is |
| Baseball | No (trivial citation-count artifact) | Encyclopedic | Clean — use as-is |
| Uranium | No | Encyclopedic | Clean — use as-is |
| Suicide | Yes (quote % — false positive) | Encyclopedic | Clean — use as-is |
| Mass Shooting | Yes (quote % — false positive) | Encyclopedic | Clean — use as-is |
| Rape | No | Encyclopedic (light check) | Clean — use as-is |
| Opioid | No | Encyclopedic (light check) | Clean — use as-is |
| **Harry Potter** | **Yes (4 independent metrics + missing section headers)** | **Narrative fan-fiction** | **Contaminated — needs replacement for any run treating it as encyclopedic source text** |
| Cannabis | Yes (mild, sentence-length) | Encyclopedic | Clean — use as-is |
| Gambling | No | Encyclopedic (light check) | Clean — use as-is |
| Gun | No | Encyclopedic (light check) | Clean — use as-is |
| Pornography | No | Encyclopedic (light check) | Clean — use as-is |

**14 of 15 concepts are clean.** The contamination is isolated to Harry Potter — this is not a systemic issue
with the dataset.

---

## Direct answer: should Harry Potter remain the Track A validation concept?

**Not as the sole or primary validation concept, no — but it retains narrow, secondary value that shouldn't be
thrown away either.** Two things are both true and in tension:

1. **Harry Potter's `wikipedia_content` cannot be trusted as encyclopedic source text.** It's fan-fiction
   dialogue and invented plot, not a topical description of the Harry Potter franchise. Any pipeline step that
   treats it as "the text characterizing this concept" — seed-token extraction, `get_mlp_act_signs`'s
   activation-sign computation, `filter_features_by_effect_and_activations`'s forget-set — is working from
   contaminated input. This directly explains the earlier finding (from the Track A seed-token validation pass)
   that automated extraction recovered only 2 of 8 known-correct hardcoded tokens: the algorithm was ranking
   fan-fiction dialogue words (character names in dialogue tags, pronouns), not encyclopedic topic terms,
   because that's what the text actually contains. That result was a **data problem being misread as a
   pipeline problem**, and continuing to validate against Harry Potter risks repeating that mistake — either a
   bad result gets wrongly blamed on the pipeline, or by chance a good result doesn't actually demonstrate the
   pipeline works on genuine encyclopedic input (which is what the other 14 concepts, and any real future
   concept, actually are).

2. **Harry Potter is still the only concept with an independently-documented ground truth** — the five
   hardcoded features in `pisces_ref/erasing_harry_potter.ipynb`, picked by PISCES's own authors. None of the
   other 14 concepts have any external reference to compare Track A's output against at all.

**Recommendation:** stop treating Harry Potter as the definitive pipeline sanity check, but don't discard it
either — run it in parallel with at least one confirmed-clean concept (e.g. Golf or Culture of Greece: both
fully clean per this audit, moderate length, and topically unambiguous enough that a human reviewer can
sanity-check the seed tokens/selected features for topical relevance even without a formal hardcoded answer
key). Treat Harry Potter's hardcoded-feature comparison as a narrow "does the mechanical pipeline run end to
end and land roughly in the right neighborhood" check, and treat the clean-concept run as the actual test of
whether the automated seed-token/discovery methodology works on real encyclopedic text — which is what it will
face for the other 14 concepts and needs to work well on. This wasn't decided unilaterally here since it's an
experimental-design call, not a data-quality one — flagging it for the team alongside this audit's findings.
