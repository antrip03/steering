"""
Unit tests for hub_storage.py's argument-passing correctness -- mocks
huggingface_hub itself, so no network/credentials needed here. The actual
push/pull/list functions were separately verified end-to-end against the
real antrip03/pisces-track-a-runs repo (push a tiny file, list it, pull it
back, confirm byte-identical content, then delete it) -- that's not
repeatable in CI without credentials, so this test protects the plumbing
(right repo_id, right repo_type, right filename handling) against future
changes without needing a live network call every time.
"""
from __future__ import annotations

from pathlib import Path

import hub_storage


def test_push_run_output_uses_basename_and_dataset_repo_type(monkeypatch, tmp_path):
    captured = {}

    class FakeHfApi:
        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            captured["path_or_fileobj"] = path_or_fileobj
            captured["path_in_repo"] = path_in_repo
            captured["repo_id"] = repo_id
            captured["repo_type"] = repo_type

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    local_file = tmp_path / "golf__layers_6__reduced.parquet"
    local_file.write_bytes(b"fake parquet bytes")

    url = hub_storage.push_run_output(local_file)

    assert captured["path_or_fileobj"] == str(local_file)
    assert captured["path_in_repo"] == "golf__layers_6__reduced.parquet"  # basename only, not full path
    assert captured["repo_id"] == hub_storage.HF_REPO_ID
    assert captured["repo_type"] == "dataset"
    assert hub_storage.HF_REPO_ID in url
    assert "golf__layers_6__reduced.parquet" in url


def test_pull_run_output_passes_filename_and_local_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_hf_hub_download(repo_id, repo_type, filename, local_dir):
        captured.update(repo_id=repo_id, repo_type=repo_type, filename=filename, local_dir=local_dir)
        return str(tmp_path / filename)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    result = hub_storage.pull_run_output("golf__layers_1__original.parquet", tmp_path)

    assert captured["repo_id"] == hub_storage.HF_REPO_ID
    assert captured["repo_type"] == "dataset"
    assert captured["filename"] == "golf__layers_1__original.parquet"
    assert captured["local_dir"] == str(tmp_path)
    assert result == Path(tmp_path / "golf__layers_1__original.parquet")


def test_list_run_outputs_filters_to_parquet_only(monkeypatch):
    class FakeHfApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == hub_storage.HF_REPO_ID
            assert repo_type == "dataset"
            return ["golf__layers_1__original.parquet", "README.md", ".gitattributes", "golf__layers_6__reduced.parquet"]

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    result = hub_storage.list_run_outputs()

    assert result == ["golf__layers_1__original.parquet", "golf__layers_6__reduced.parquet"]
