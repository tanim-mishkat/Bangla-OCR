"""Publish data/processed/ + dataset card to the Hugging Face Hub as one atomic commit."""

import json
import subprocess
from pathlib import Path

import polars as pl
from huggingface_hub import CommitOperationAdd, HfApi

from bangla_ocr import card, checks, clean, split
from bangla_ocr.download import sha256_file

REPO_NAME = "bangla-ocr-bench"
RECORD_PATH = split.OUT_DIR / "published.json"


def code_revision() -> str:
    """Short git commit of the code, with '-dirty' if there are uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return git("rev-parse", "--short", "HEAD").strip() + (
        "-dirty" if git("status", "--porcelain") else ""
    )


def split_counts(out_dir: Path = split.OUT_DIR) -> dict[str, dict[str, int]]:
    return {
        name: dict(pl.scan_parquet(p).group_by("source").len().sort("source").collect().iter_rows())
        for name, p in checks.split_paths(out_dir).items()
    }


def publish(
    api: HfApi,
    repo_id: str,
    out_dir: Path = split.OUT_DIR,
    report_path: Path = clean.REPORT_PATH,
    private: bool = True,
) -> dict:
    """Upload splits + README.md in one commit. Returns the record Phase 5 pins to."""
    rev = code_revision()
    if rev.endswith("-dirty"):
        raise RuntimeError(
            "Commit your code first: the card must name the code that built the data"
        )
    paths = checks.split_paths(out_dir)
    sha256 = {name: sha256_file(p) for name, p in paths.items()}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    readme = card.render(split_counts(out_dir), report, sha256, rev)

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    ops = [CommitOperationAdd(f"data/{name}.parquet", str(p)) for name, p in paths.items()]
    ops.append(CommitOperationAdd("README.md", readme.encode("utf-8")))
    info = api.create_commit(
        repo_id, operations=ops, commit_message=f"Publish (code {rev})", repo_type="dataset"
    )
    return {"repo_id": repo_id, "revision": info.oid, "code_revision": rev, "sha256": sha256}
