"""
Kaggle-specific Track A discovery entrypoint. Separate from discover.py
because Kaggle's T4 (Turing, no native bf16) and 12-hour session cap need
real accommodations that don't belong in the general-purpose script:

  - fp16 instead of bf16/fp32 (Turing has no bf16 tensor cores; fp16 risks
    NaN/overflow that bf16's wider exponent range doesn't -- watched for and
    logged explicitly, see `check_for_nonfinite`).
  - a batch-size sweep (--sweep-batch-sizes) to pick a T4-appropriate batch
    size empirically, since VRAM headroom isn't known in advance.
  - a minimal, script-scoped checkpoint/resume layer on top of what
    pisces_ref/feature_finder.py already provides (`checkpoint_path`/
    `checkpoint_dir` args, batch-level resume inside get_feature_effect and
    filter_features_by_mmlu) -- this script adds ONE more stage of resume,
    for the candidate-search step, which has no checkpointing of its own.
    This is NOT the full production checkpoint/resume system; it's scoped
    to exactly what a single `python run_kaggle.py --concept X` invocation
    needs to survive a session restart.
  - single concept per invocation (required --concept), so a 12-hour cap
    means "restart this concept", not "restart the whole 6-concept sweep".
  - explicit progress logging (layer/candidate counts/elapsed/ETA) and
    Hugging Face cache size/location logging (Kaggle disk caps at ~20GB).

IMPORTANT: this script has not been run against a real GPU (this development
machine has no CUDA device). Treat it as constructed-and-reviewed, not
validated, until it's actually run on Kaggle -- see track_a_feature_discovery
/README.md's "Runtime reductions" section for the same caveat applied to
--reduced in discover.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
PISCES_REF = ROOT / "pisces_ref"
for p in (ROOT, PISCES_REF, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import FeatureRecord, MODEL_NAME  # noqa: E402
from editor import Feature, get_mlp_act_signs  # noqa: E402
from feature_finder import (  # noqa: E402
    filter_features_by_effect_and_activations,
    filter_features_by_mmlu,
    search_features,
)
from seed_tokens import derive_seed_tokens_for_concept, get_neg_toks  # noqa: E402
from vocab_projection import build_all_layer_lookups  # noqa: E402
from reductions import (  # noqa: E402
    CASCADE_KEEP_FRACTION,
    CASCADE_PREFILTER_BATCHES,
    EARLY_EXIT_AFTER_BATCHES,
    EARLY_EXIT_MARGIN,
    MIDDLE_LAYERS,
    REDUCED_CONCEPTS,
    VOCABPROJ_MINMATCH,
)
from discover import cascade_filter_candidates, get_concept_data, load_cvs  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts" / "features"
_RUN_START = time.monotonic()


def _elapsed() -> str:
    s = time.monotonic() - _RUN_START
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.1f}s"


def log(msg: str) -> None:
    print(f"[{_elapsed()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Disk-space awareness -- Kaggle caps working disk at ~20GB, and the SAEs +
# Gemma-2-2B-it easily approach a meaningful fraction of that.
# ---------------------------------------------------------------------------

def hf_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def log_disk_usage(label: str, warn_gb: float = 18.0) -> None:
    cache = hf_cache_dir()
    size_gb = dir_size_bytes(cache) / (1024 ** 3)
    log(f"disk [{label}]: HF cache at {cache} = {size_gb:.2f}GB")
    if size_gb >= warn_gb:
        log(f"  WARNING: HF cache >= {warn_gb}GB -- approaching Kaggle's ~20GB disk cap")


def log_gpu_memory(label: str) -> None:
    """No-ops on non-CUDA devices (this development machine has none, so
    this path is untested against real numbers -- see README.md's
    'build_layer_lookup OOM fix' section). Uses max_memory_allocated, which
    is reset explicitly before the vocab-projection step so the peak it
    reports isolates that step rather than accumulating across the whole
    run."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    log(f"gpu [{label}]: allocated={allocated:.2f}GB peak={peak:.2f}GB")


# ---------------------------------------------------------------------------
# fp16 NaN/overflow watch. fp16 (unlike bf16) has a narrow exponent range and
# can overflow to inf on activations that bf16/fp32 handle fine -- checked
# explicitly after each expensive stage rather than trusting silent fp16
# behavior, since a silently-NaN'd effect score would look like "feature has
# zero effect" (a valid value) rather than an error.
# ---------------------------------------------------------------------------

def check_for_nonfinite(label: str, effects: dict) -> None:
    bad = []
    for key, vals in effects.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size and not np.all(np.isfinite(arr)):
            bad.append(key)
    if bad:
        log(f"  WARNING [{label}]: {len(bad)} feature(s) produced NaN/inf under fp16: {bad[:10]}{'...' if len(bad) > 10 else ''}")
    else:
        log(f"  [{label}]: all finite ({len(effects)} features checked)")


# ---------------------------------------------------------------------------
# Minimal script-scoped checkpoint/resume: ONE extra stage (candidate search)
# beyond what feature_finder.py already checkpoints internally (effect
# measurement, MMLU filtering both resume via checkpoint_dir/checkpoint_path
# already threaded through by discover_concept/filter_features_by_*). This is
# deliberately not a general system -- just enough that a Kaggle session
# restart doesn't have to redo a candidate search that already finished.
# ---------------------------------------------------------------------------

def load_candidates_checkpoint(path: Path) -> list[Feature] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [Feature(layer=r["layer"], id=r["id"], neg=r["neg"]) for r in rows]


def save_candidates_checkpoint(path: Path, candidates: list[Feature]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"layer": c.layer, "id": c.id, "neg": c.neg} for c in candidates], f)


def search_features_with_progress(model, lls, seed_tokens, minmatch, layers) -> list[Feature]:
    """Loops layers one at a time (search_features itself accepts a layers=
    list and would do this internally in one call) purely so each layer's
    progress/candidate-count can be logged -- pisces_ref's search_features
    has no per-layer callback of its own."""
    all_candidates: list[Feature] = []
    for idx, layer in enumerate(layers):
        layer_candidates = search_features(model, lls, seed_tokens, minmatch=minmatch, layers=[layer])
        all_candidates.extend(layer_candidates)
        log(f"  layer search {idx+1}/{len(layers)} (layer={layer}): "
            f"+{len(layer_candidates)} candidates (running total {len(all_candidates)})")
    return all_candidates


# ---------------------------------------------------------------------------
# Batch-size sweep: T4 VRAM headroom isn't known in advance (varies with
# what else is resident), so this times a handful of forward passes at each
# candidate batch size on a small probe slice and reports throughput, rather
# than assuming batch_size=3 (feature_finder.py's hardcoded default) is
# optimal for a T4.
# ---------------------------------------------------------------------------

def sweep_batch_sizes(model, forget_text: str, sizes: list[int], n_probe_batches: int = 3) -> None:
    lines = forget_text.splitlines()
    log(f"batch-size sweep over {sizes}, {n_probe_batches} probe batches each")
    for bs in sizes:
        probe_lines = lines[: bs * n_probe_batches]
        if not probe_lines:
            continue
        start = time.monotonic()
        with torch.no_grad():
            for i in range(0, len(probe_lines), bs):
                batch = probe_lines[i:i + bs]
                if not batch:
                    continue
                model(batch)
        elapsed = time.monotonic() - start
        n_batches = -(-len(probe_lines) // bs)
        log(f"  batch_size={bs}: {elapsed:.2f}s for {n_batches} batches "
            f"({elapsed/max(n_batches,1):.3f}s/batch, {elapsed/max(len(probe_lines),1):.3f}s/line)")


def discover_concept_kaggle(
    model, cvs: list[dict], concept: str, layers: list[int], device: str,
    effect_batch_size: int, checkpoint_dir: Path, debug_log_noop_edits: bool = False,
) -> pd.DataFrame:
    concept_data = get_concept_data(cvs, concept)

    def is_single_token(tok: str) -> bool:
        return len(model.to_tokens(tok, prepend_bos=False)[0]) == 1

    seed_tokens = derive_seed_tokens_for_concept(concept, cvs, is_single_token)
    if not seed_tokens:
        raise ValueError(f"No single-token seed words for {concept!r} -- check seed_tokens.py.")
    neg_toks = get_neg_toks(is_single_token)
    log(f"seed_tokens={seed_tokens} neg_toks={neg_toks}")

    saes_by_layer: dict[int, object] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log("building vocab-projection lookups (chunked -- see vocab_projection.py::FEATURE_CHUNK_SIZE)...")
    lls = build_all_layer_lookups(model, MODEL_NAME, layers=layers, device=device, saes_out=saes_by_layer)
    log_gpu_memory("post-vocab-projection")

    candidates_ckpt = checkpoint_dir / "candidates.json"
    candidates = load_candidates_checkpoint(candidates_ckpt)
    if candidates is not None:
        log(f"resumed {len(candidates)} candidates from {candidates_ckpt}")
    else:
        log(f"searching for candidate features across {len(layers)} layers (minmatch={VOCABPROJ_MINMATCH})...")
        candidates = search_features_with_progress(model, lls, seed_tokens, VOCABPROJ_MINMATCH, layers)
        save_candidates_checkpoint(candidates_ckpt, candidates)
        log(f"{len(candidates)} candidate features found, checkpointed to {candidates_ckpt}")

    forget_text = concept_data["wikipedia_content"]
    signs = get_mlp_act_signs(model, seed_tokens, forget_text.splitlines()[:1000])

    log(f"cascade prefilter ({CASCADE_PREFILTER_BATCHES} batches, keep_fraction={CASCADE_KEEP_FRACTION})...")
    effect_candidates = cascade_filter_candidates(
        model, candidates, forget_text, signs, seed_tokens, neg_toks,
        CASCADE_PREFILTER_BATCHES, CASCADE_KEEP_FRACTION, batch_size=effect_batch_size,
        debug_log_noop_edits=debug_log_noop_edits,
    )

    # Corpus-size reduction (EFFECT_MEASUREMENT_BATCHES) dropped -- see
    # reductions.py's comment: real-run validation on Golf showed it changes
    # the selected FC, not just its speed (two identical full-corpus runs
    # matched exactly; a 20-batch run against the same baseline did not).
    # effect_text is the full forget_text; 12-hour-session-cap pressure
    # should be handled by this script's checkpoint/resume across sessions,
    # not by measuring less data.
    effect_text = forget_text
    log(f"effect measurement: {len(effect_candidates)} candidates x full corpus "
        f"({len(forget_text.splitlines())} lines, corpus reduction dropped), "
        f"early_exit_after_batches={EARLY_EXIT_AFTER_BATCHES}")

    effect_filtered, (pos_effects, neg_effects, activations) = filter_features_by_effect_and_activations(
        model, effect_candidates, effect_text, signs, seed_tokens, neg_toks, filter_by_act=True,
        checkpoint_dir=str(checkpoint_dir),
        early_exit_after_batches=EARLY_EXIT_AFTER_BATCHES, early_exit_margin=EARLY_EXIT_MARGIN,
        batch_size=effect_batch_size, debug_log_noop_edits=debug_log_noop_edits,
    )
    check_for_nonfinite("effect measurement", pos_effects)
    check_for_nonfinite("effect measurement (neg)", neg_effects)
    log(f"{len(effect_filtered)} candidates survived effect/activation filtering")

    log("MMLU specificity filtering...")
    selected = filter_features_by_mmlu(
        model, effect_filtered, signs,
        checkpoint_path=str(checkpoint_dir / "mmlu.ckpt"),
        debug_log_noop_edits=debug_log_noop_edits,
    )
    selected_keys = {(f.layer, f.id, f.neg) for f in selected}
    log(f"{len(selected)} features survived filtering (selected=True)")

    rows = []
    for feature in candidates:
        sae = saes_by_layer[feature.layer]
        decoder_direction = sae.W_dec[feature.id].detach().float().cpu().tolist()
        if not all(np.isfinite(decoder_direction)):
            log(f"  WARNING: non-finite decoder_direction for feature {feature} -- check fp16 overflow")
        ll = lls[feature.layer]
        top_tokens = ll.t[feature.id]
        bottom_tokens = ll.b[feature.id]

        pos_effect_vals = pos_effects.get((feature.layer, feature.id))
        effect_score = float(sum(pos_effect_vals) / len(pos_effect_vals)) if pos_effect_vals else None

        rows.append(
            FeatureRecord(
                concept=concept,
                layer=feature.layer,
                feature_id=feature.id,
                neg=feature.neg,
                decoder_direction=decoder_direction,
                top_tokens=top_tokens,
                bottom_tokens=bottom_tokens,
                mass_ratio_or_effect_score=effect_score,
                selected=(feature.layer, feature.id, feature.neg) in selected_keys,
            ).to_dict()
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True, help=f"Single concept to run (e.g. one of {REDUCED_CONCEPTS}).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help=f"Defaults to MIDDLE_LAYERS ({MIDDLE_LAYERS[0]}-{MIDDLE_LAYERS[-1]}).")
    parser.add_argument("--batch-size", type=int, default=3, help="Effect-measurement batch size (feature_finder.py's own default is 3).")
    parser.add_argument("--sweep-batch-sizes", type=int, nargs="*", default=None,
                         help="If given, time these batch sizes on a small probe and exit (no discovery run).")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Must point at storage that survives a Kaggle session restart (e.g. a Kaggle Dataset "
                              "mounted read/write, or /kaggle/working saved as a dataset version) for resume to help "
                              "across the 12-hour cap. Defaults to artifacts/checkpoints/<concept>/, which is NOT "
                              "persistent on Kaggle unless you arrange that yourself.")
    parser.add_argument(
        "--debug-log-noop-edits",
        action="store_true",
        help=(
            "If a layer edit turns out to be a no-op, log which of two causes it was "
            "and continue, instead of crashing with pisces_ref/editor.py's 'No changes "
            "made to the model in layer X' assertion. Off by default here matches "
            "discover.py's default, but strongly recommended: a real discover.py run "
            "hit this exact crash at three separate call sites (get_feature_effect's "
            "own loop, cascade_filter_candidates, filter_features_by_mmlu) before all "
            "three were wired -- see track_a_feature_discovery/README.md."
        ),
    )
    args = parser.parse_args()

    layers = args.layers if args.layers is not None else MIDDLE_LAYERS
    cvs = load_cvs()
    concept_data = get_concept_data(cvs, args.concept)

    concept_slug = args.concept.lower().replace(" ", "_")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (ROOT / "artifacts" / "checkpoints" / concept_slug)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log(f"checkpoint_dir={checkpoint_dir}")
    if args.checkpoint_dir is None:
        log("  NOTE: using the default (non-persistent-on-Kaggle) checkpoint dir -- pass --checkpoint-dir "
            "pointing at persistent storage if you need resume to survive a session restart.")

    log_disk_usage("startup")

    from sae_lens import HookedSAETransformer

    # fp16, not bf16: T4 is Turing architecture, no native bf16 tensor cores --
    # bf16 would run but gain none of its usual speed benefit and still cost the
    # same memory as fp16. fp16's narrower exponent range (vs bf16) means real
    # overflow/NaN risk that bf16 on this project's other (Ampere+/CPU) targets
    # doesn't have -- watched for explicitly via check_for_nonfinite() after
    # every expensive stage, not assumed away.
    model_dtype = torch.float16
    log(f"loading {MODEL_NAME} (dtype={model_dtype}, device={args.device})...")
    model = HookedSAETransformer.from_pretrained_no_processing(MODEL_NAME, device=args.device, dtype=model_dtype)
    model.requires_grad_(False)
    log("model loaded")
    log_disk_usage("post-model-load")
    log_gpu_memory("post-model-load")

    if args.sweep_batch_sizes:
        with torch.no_grad():
            sweep_batch_sizes(model, concept_data["wikipedia_content"], args.sweep_batch_sizes)
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        df = discover_concept_kaggle(
            model, cvs, args.concept, layers=layers, device=args.device,
            effect_batch_size=args.batch_size, checkpoint_dir=checkpoint_dir,
            debug_log_noop_edits=args.debug_log_noop_edits,
        )

    out_path = ARTIFACTS_DIR / f"{concept_slug}.parquet"
    df.to_parquet(out_path, index=False)
    log(f"wrote {len(df)} candidate features to {out_path}")
    log_disk_usage("finish")
    log_gpu_memory("finish")


if __name__ == "__main__":
    main()
