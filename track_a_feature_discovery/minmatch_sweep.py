"""
Fast standalone VOCABPROJ_MINMATCH sweep -- reports candidate counts across
several minmatch values for one concept/layer, WITHOUT running any of the
expensive stages (effect measurement, activation counting, MMLU). Those are
all downstream of search_features and cost hours; search_features itself is
a handful of cheap set-intersection checks over an already-built
vocabulary-projection lookup, so once the model and SAE are loaded, sweeping
5-10 minmatch values costs seconds, not hours.

Exists because Golf/layer-1 and Golf/layer-6 both collapsed to 0 candidates
at minmatch=5 (see track_a_feature_discovery/README.md's "Runtime
reductions" section) -- minmatch=1 gives 69 at layer 1, so there's a real
transition somewhere in between worth finding cheaply, rather than paying
for N full ~2-hour discover.py runs to find it one value at a time.

python minmatch_sweep.py --concept Golf --layers 6 --minmatch-values 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PISCES_REF = ROOT / "pisces_ref"
for p in (ROOT, PISCES_REF, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import MODEL_NAME  # noqa: E402
from feature_finder import search_features  # noqa: E402
from seed_tokens import derive_seed_tokens_for_concept  # noqa: E402
from vocab_projection import build_all_layer_lookups  # noqa: E402
from discover import load_cvs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Defaults to all 26 layers if omitted.")
    parser.add_argument("--minmatch-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cvs = load_cvs()

    from sae_lens import HookedSAETransformer

    # Same loading choices as discover.py's main() -- see that file's comments
    # for the full rationale (bf16 memory savings, from_pretrained_no_processing
    # avoiding a transient RAM spike). No forward passes happen in this script
    # at all, so those choices matter less here, but keeping them identical
    # avoids introducing a second, subtly-different model-loading code path.
    model = HookedSAETransformer.from_pretrained_no_processing(MODEL_NAME, device=args.device, dtype=torch.bfloat16)
    model.requires_grad_(False)

    def is_single_token(tok: str) -> bool:
        return len(model.to_tokens(tok, prepend_bos=False)[0]) == 1

    seed_tokens = derive_seed_tokens_for_concept(args.concept, cvs, is_single_token)
    if not seed_tokens:
        raise ValueError(f"No single-token seed words for {args.concept!r}.")
    print(f"[{args.concept}] seed_tokens={seed_tokens}")

    with torch.no_grad():
        saes_by_layer: dict[int, object] = {}
        lls = build_all_layer_lookups(model, MODEL_NAME, layers=args.layers, device=args.device, saes_out=saes_by_layer)

        print(f"\n{'minmatch':>10}  {'candidates':>10}")
        for minmatch in args.minmatch_values:
            candidates = search_features(model, lls, seed_tokens, minmatch=minmatch, layers=args.layers)
            print(f"{minmatch:>10}  {len(candidates):>10}")


if __name__ == "__main__":
    main()
