"""
Push/pull Track A run outputs (artifacts/features/*.parquet) to/from a
private HF Dataset repo. `artifacts/` is intentionally gitignored (see
.gitignore's "Generated data" comment) so these never belonged in the git
repo, but they still need a durable, shared home -- Kaggle sessions are
ephemeral, and every real comparison run in this project's validation
history so far required manually downloading a file from Kaggle's UI and
keeping track of local renames to avoid collisions (see build_run_tag() in
discover.py, added after that exact problem happened repeatedly).

Requires an HF token with repo.write scope for HF_REPO_ID's namespace.
huggingface_hub resolves this automatically from the HF_TOKEN environment
variable, then from its own cached login (~/.cache/huggingface/token) --
same resolution order used everywhere else in this project for read access
(HookedSAETransformer.from_pretrained, load_dataset, etc.), so no new
credential-handling pattern is introduced here.
"""
from __future__ import annotations

from pathlib import Path

HF_REPO_ID = "antrip03/pisces-track-a-runs"


def push_run_output(local_path: Path) -> str:
    """Uploads local_path to HF_REPO_ID under its own filename (already
    collision-free per build_run_tag()/build_layers_slug() -- see
    discover.py) and returns the file's HF URL. Raises on failure (e.g.
    insufficient token scope) rather than swallowing errors -- a silent
    failure here would be worse than no push at all, since the caller would
    believe the file is safely stored externally when it isn't."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/{local_path.name}"


def pull_run_output(filename: str, local_dir: Path) -> Path:
    """Downloads `filename` (as uploaded by push_run_output -- just the
    basename, e.g. golf__layers_6__reduced.parquet) from HF_REPO_ID into
    local_dir, returning the local path. Used both by callers on Kaggle
    (e.g. to compare against a baseline pushed from a previous session) and
    by compare_fc.py callers on any machine with network access and a
    (read-only is sufficient here) HF token."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename, local_dir=str(local_dir))
    return Path(path)


def list_run_outputs() -> list[str]:
    """Lists filenames currently stored in HF_REPO_ID -- lets a caller (or
    Claude, working from a machine with no direct access to Kaggle's UI)
    discover what's available to pull without already knowing the exact
    filename."""
    from huggingface_hub import HfApi

    api = HfApi()
    return [f for f in api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset") if f.endswith(".parquet")]
