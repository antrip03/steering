"""
Modal entrypoint for Track C erasure evaluation -- mirrors
track_a_feature_discovery/modal_app.py's design exactly: a thin subprocess
wrapper around run_erasure_eval.py's own CLI, no reimplementation of any
evaluation logic, on Modal's A10G with no session cap.

One-time setup (run locally, not through an agent -- token material
shouldn't pass through one):
    modal secret create huggingface HF_TOKEN="$(cat ~/.cache/huggingface/token)"
    modal secret create gemini GEMINI_API_KEY="$GEMINI_API_KEY"

Usage:
    modal run --detach track_c_erasure_eval/modal_app.py --concept Golf
    modal run --detach track_c_erasure_eval/modal_app.py --concept "Golf,Uranium,Poison"

Multiple concepts: comma-separate --concept. Requires each concept's Track A
output to already exist at artifacts/features/<canonical filename> locally
in the image (baked in via add_local_dir at deploy/run time) -- run
track_a_feature_discovery/pull_production_results.py first if a concept's
result only exists on the hub so far, not in this local working tree.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

app = modal.App("pisces-track-c")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .add_local_dir(
        str(ROOT),
        remote_path="/root/steering",
        ignore=[".git", "__pycache__", "*.pyc", "artifacts/checkpoints", ".pytest_cache", ".claude", ".gitmodules"],
    )
)

# artifacts/erasure_eval (Track C's per-concept output) persisted the same
# way artifacts/checkpoints is for Track A -- lets a crashed/retried
# invocation not lose already-evaluated concepts, and lets --push-to-hub
# runs be inspected locally afterward via `modal volume get`.
results_volume = modal.Volume.from_name("pisces-track-c-results", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("pisces-track-a-hf-cache", create_if_missing=True)  # shared with Track A -- same model/SAE weights


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    retries=2,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("gemini")],
    volumes={
        "/root/steering/artifacts/erasure_eval": results_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_erasure_eval(
    concepts: list[str],
    mmlu_limit: int = 300,
    push_to_hub: bool = True,
) -> int:
    cmd = [sys.executable, "run_erasure_eval.py", "--device", "cuda", "--mmlu-limit", str(mmlu_limit)]
    for c in concepts:
        cmd += ["--concept", c]
    if push_to_hub:
        cmd.append("--push-to-hub")

    print(f"running: {' '.join(cmd)}", flush=True)
    # Popen + periodic commit, not subprocess.run + a single commit in
    # finally -- see track_a_feature_discovery/modal_app.py's identical
    # comment. A container killed outright (not a graceful crash) would
    # otherwise lose any concepts evaluated since the last commit.
    proc = subprocess.Popen(cmd, cwd="/root/steering/track_c_erasure_eval")
    try:
        while proc.poll() is None:
            time.sleep(120)
            results_volume.commit()
        return proc.returncode
    finally:
        results_volume.commit()


@app.local_entrypoint()
def main(
    concept: str,
    mmlu_limit: int = 300,
    push_to_hub: bool = True,
):
    concepts = [c.strip() for c in concept.split(",") if c.strip()]
    rc = run_erasure_eval.remote(concepts=concepts, mmlu_limit=mmlu_limit, push_to_hub=push_to_hub)
    print(f"run_erasure_eval.py exited with code {rc}")
    if rc != 0:
        sys.exit(rc)
