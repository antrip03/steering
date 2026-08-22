"""
Modal entrypoint for Track A discovery -- runs the exact same discover.py CLI
already validated on Kaggle, just on Modal's A10G (24GB, Ampere, native bf16)
instead of Kaggle's T4 (~14.5GB usable, Turing, fp16-only), with no 12-hour
session cap and no manual "start a new session, re-run pip install" cycle
between experiments.

Deliberately a thin subprocess wrapper around discover.py, not a
reimplementation: every fix this investigation found (SAE caching,
names_filter restrictions on run_with_cache(_with_saes), checkpoint
layer-scoping, output-filename collisions, and the cascade/minmatch/
corpus-size reductions all tried-and-dropped after real-run validation)
lives in discover.py / pisces_ref already -- duplicating that logic here
would risk silently missing one of them.

One-time setup (run locally, not via Claude -- token material shouldn't
pass through an agent):
    modal secret create huggingface HF_TOKEN="$(cat ~/.cache/huggingface/token)"

Usage:
    modal run track_a_feature_discovery/modal_app.py --concept Uranium --layers 6
    modal run track_a_feature_discovery/modal_app.py --concept "Harry Potter" \\
        --layers 1,4,20 --cvs-path pisces_ref/data/cvs.json --candidates-only
    modal run track_a_feature_discovery/modal_app.py --concept "Harry Potter" \\
        --cvs-path pisces_ref/data/cvs.json --features 4:661:1,20:11104:1,1:13394:0

Multiple concepts: comma-separate --concept (e.g. --concept "Golf,Uranium").
--layers is comma-separated; omit for discover.py's own default (all layers,
or MIDDLE_LAYERS if --reduced). --features is comma-separated 'layer:id:neg'
specs (neg as 1/0) -- runs the full effect+MMLU pipeline on exactly those
features, skipping vocab-projection search and ignoring --layers (inferred
from the --features list instead). See discover.py's --features help.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

app = modal.App("pisces-track-a")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .add_local_dir(
        str(ROOT),
        remote_path="/root/steering",
        ignore=[".git", "__pycache__", "*.pyc", "artifacts", ".pytest_cache", ".claude", ".gitmodules"],
    )
)

# Two Volumes, two different lifetimes: checkpoints are per-run resume state
# (mirrors discover.py's own checkpoint_dir, scoped by concept+layers -- see
# build_layers_slug), while the HF cache holds the ~5GB Gemma-2-2b-it model
# and GemmaScope SAE weights, worth persisting across runs regardless of
# concept/layers so every invocation after the first skips re-downloading it.
checkpoints_volume = modal.Volume.from_name("pisces-track-a-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("pisces-track-a-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    # Bumped from 6h: single-layer runs (Golf/Uranium, ~85 batches) fit
    # comfortably in 6h, but the full MIDDLE_LAYERS (10 layers) production
    # scope multiplies candidate count roughly 10x per concept, and this
    # project has no reliable timing calibration at that scale yet -- better
    # to risk an over-generous timeout than have a legitimately-still-running
    # job silently killed hours in.
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        "/root/steering/artifacts/checkpoints": checkpoints_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_discover(
    concepts: list[str],
    layers: list[int] | None = None,
    reduced: bool = False,
    minmatch: int | None = None,
    enable_cascade: bool = False,
    cvs_path: str | None = None,
    candidates_only: bool = False,
    features: list[str] | None = None,
    corpus_batches: int | None = None,
    debug_log_noop_edits: bool = True,
    push_to_hub: bool = True,
) -> int:
    cmd = [sys.executable, "discover.py", "--device", "cuda"]
    for c in concepts:
        cmd += ["--concept", c]
    if layers is not None:
        cmd += ["--layers", *[str(l) for l in layers]]
    if reduced:
        cmd.append("--reduced")
    if minmatch is not None:
        cmd += ["--minmatch", str(minmatch)]
    if enable_cascade:
        cmd.append("--enable-cascade")
    if cvs_path is not None:
        cmd += ["--cvs-path", cvs_path]
    if candidates_only:
        cmd.append("--candidates-only")
    if features:
        cmd += ["--features", *features]
    if corpus_batches is not None:
        cmd += ["--corpus-batches", str(corpus_batches)]
    if debug_log_noop_edits:
        cmd.append("--debug-log-noop-edits")
    if push_to_hub:
        cmd.append("--push-to-hub")

    print(f"running: {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, cwd="/root/steering/track_a_feature_discovery")
        return result.returncode
    finally:
        # Commit even on failure/timeout -- a partial checkpoint from a run
        # that died mid-batch is exactly what lets the next invocation resume
        # instead of redoing already-measured batches.
        checkpoints_volume.commit()


@app.local_entrypoint()
def main(
    concept: str,
    layers: str = "",
    reduced: bool = False,
    minmatch: int | None = None,
    enable_cascade: bool = False,
    cvs_path: str = "",
    candidates_only: bool = False,
    features: str = "",
    corpus_batches: int | None = None,
    push_to_hub: bool = True,
    debug_log_noop_edits: bool = True,
):
    concepts = [c.strip() for c in concept.split(",") if c.strip()]
    layer_list = [int(x) for x in layers.split(",") if x.strip()] if layers else None
    feature_list = [f.strip() for f in features.split(",") if f.strip()] if features else None

    rc = run_discover.remote(
        concepts=concepts,
        layers=layer_list,
        reduced=reduced,
        minmatch=minmatch,
        enable_cascade=enable_cascade,
        cvs_path=cvs_path or None,
        candidates_only=candidates_only,
        features=feature_list,
        corpus_batches=corpus_batches,
        debug_log_noop_edits=debug_log_noop_edits,
        push_to_hub=push_to_hub,
    )
    print(f"discover.py exited with code {rc}")
    if rc != 0:
        sys.exit(rc)
