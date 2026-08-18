"""
Step 2.5 validation: diff the selected feature set (FC) between an original-
settings discover.py run and a --reduced run for the same concept.

python compare_fc.py --original golf_original.parquet --reduced golf_reduced.parquet

Or, if both runs were uploaded with --push-to-hub (see hub_storage.py):

python compare_fc.py --from-hub --original golf__layers_6__original.parquet --reduced golf__layers_6__reduced.parquet

Reports exact matches, features that differ in selection status, and effect
scores for the differing ones -- per the task's own instruction ("report
exact matches, any differing features, intermediate effect scores for
borderline cases... if results diverge, report clearly rather than silently
keeping the reduction"), not just a pass/fail verdict.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd


def _key(row) -> tuple[int, int, bool]:
    return (int(row.layer), int(row.feature_id), bool(row.neg))


def load_candidates(path: str) -> dict[tuple[int, int, bool], dict]:
    df = pd.read_parquet(path)
    return {
        _key(row): {"selected": bool(row.selected), "effect_score": row.mass_ratio_or_effect_score}
        for row in df.itertuples()
    }


def compare(original_path: str, reduced_path: str) -> None:
    orig = load_candidates(original_path)
    reduced = load_candidates(reduced_path)

    orig_keys = set(orig)
    reduced_keys = set(reduced)
    common_keys = orig_keys & reduced_keys

    print(f"Candidate pool: original={len(orig_keys)}  reduced={len(reduced_keys)}  common={len(common_keys)}")

    only_in_original = orig_keys - reduced_keys
    only_in_reduced = reduced_keys - orig_keys
    if only_in_original:
        print(f"\n{len(only_in_original)} candidate(s) in original's pool but NOT in reduced's "
              f"(expected: --reduced's tighter minmatch drops low-overlap candidates before "
              f"effect measurement even runs on them):")
        for key in sorted(only_in_original):
            print(f"  {key}: selected={orig[key]['selected']} effect_score={orig[key]['effect_score']}")
    if only_in_reduced:
        print(f"\nWARNING: {len(only_in_reduced)} candidate(s) in reduced's pool but NOT in "
              f"original's -- this should not happen (minmatch=5's candidates should be a strict "
              f"subset of minmatch=1's); investigate before trusting this comparison:")
        for key in sorted(only_in_reduced):
            print(f"  {key}: selected={reduced[key]['selected']} effect_score={reduced[key]['effect_score']}")

    exact_matches = [k for k in common_keys if orig[k]["selected"] == reduced[k]["selected"]]
    differing = [k for k in common_keys if orig[k]["selected"] != reduced[k]["selected"]]

    n_selected_orig = sum(1 for k in common_keys if orig[k]["selected"])
    n_selected_reduced = sum(1 for k in common_keys if reduced[k]["selected"])
    print(f"\nAmong the {len(common_keys)} candidates present in BOTH pools:")
    print(f"  selected=True in original: {n_selected_orig}")
    print(f"  selected=True in reduced:  {n_selected_reduced}")
    print(f"  exact selection-status matches: {len(exact_matches)}/{len(common_keys)}")

    if differing:
        print(f"\n{len(differing)} candidate(s) DIFFER in selection status between the two runs:")
        for key in sorted(differing):
            o, r = orig[key], reduced[key]
            print(f"  {key}: original selected={o['selected']} (effect_score={o['effect_score']})  "
                  f"| reduced selected={r['selected']} (effect_score={r['effect_score']})")
        print("\n>>> The reductions did NOT reproduce the same selected FC for this concept/layer. "
              "Report this plainly rather than treating the reduction as validated. <<<")
    elif only_in_original:
        print("\nAll candidates common to both pools matched exactly, but --reduced's candidate "
              "search (VOCABPROJ_MINMATCH) dropped some candidates before they could even be "
              "measured -- whether that's acceptable depends on whether any of those dropped "
              "candidates would plausibly have been selected under the original's minmatch=1 "
              "(see their selected=True/effect_score above). Not an automatic pass.")
    else:
        print("\n>>> Exact match: identical candidate pools and identical selected FC. <<<")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Path (or, with --from-hub, filename) of the original-settings run's parquet")
    parser.add_argument("--reduced", required=True, help="Path (or, with --from-hub, filename) of the --reduced run's parquet")
    parser.add_argument(
        "--from-hub", action="store_true",
        help="Treat --original/--reduced as filenames on hub_storage.HF_REPO_ID (as uploaded by "
             "--push-to-hub) rather than local paths, and pull them first. Lets a comparison run "
             "on any machine with network access and an HF token, not just wherever the files "
             "happen to be downloaded to.",
    )
    args = parser.parse_args()

    if args.from_hub:
        from hub_storage import pull_run_output
        tmpdir = Path(tempfile.mkdtemp(prefix="compare_fc_"))
        original_path = str(pull_run_output(args.original, tmpdir))
        reduced_path = str(pull_run_output(args.reduced, tmpdir))
    else:
        original_path = args.original
        reduced_path = args.reduced

    compare(original_path, reduced_path)


if __name__ == "__main__":
    main()
