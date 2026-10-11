import json
import re
from types import SimpleNamespace

import polars as pl
import pytest

from bangla_ocr import card, checks, publish


class FakeApi:
    """Records Hub calls instead of making them."""

    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def create_commit(self, repo_id, operations, **kwargs):
        self.calls.append(("create_commit", repo_id, operations, kwargs))
        return SimpleNamespace(oid="f" * 40)


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(card, "split_table", lambda counts: "TABLE")  # tested in test_card
    monkeypatch.setattr(checks, "run", lambda *a: {n: [] for n in checks.NAMES})  # all pass
    out = tmp_path / "processed"
    out.mkdir()
    for name, sources in {
        "train": ["synth"],
        "val": ["synth"],
        "test": ["reid2019", "synth"],
    }.items():
        pl.DataFrame({"source": sources}).write_parquet(out / f"{name}.parquet")
    report = tmp_path / "clean_report.json"
    report.write_text(json.dumps({"input": 5, "output": 4}), encoding="utf-8")
    return out, report


def test_publish_one_commit_with_splits_and_card(built, monkeypatch):
    monkeypatch.setattr(publish, "code_revision", lambda: "abc1234")
    out, report = built
    api = FakeApi()

    record = publish.publish(api, "me/bench", out, report)

    (_, repo, repo_kw), (_, repo2, ops, commit_kw) = api.calls
    assert repo == repo2 == "me/bench"
    assert repo_kw == {"repo_type": "dataset", "private": True, "exist_ok": True}
    assert commit_kw["repo_type"] == "dataset"
    assert [op.path_in_repo for op in ops] == [
        "data/train.parquet",
        "data/val.parquet",
        "data/test.parquet",
        "README.md",
    ]
    assert b"code revision `abc1234`" in ops[-1].path_or_fileobj
    assert record["revision"] == "f" * 40
    assert record["code_revision"] == "abc1234"
    assert set(record["sha256"]) == {"train", "val", "test"}


def test_publish_refuses_uncommitted_code(built, monkeypatch):
    monkeypatch.setattr(publish, "code_revision", lambda: "abc1234-dirty")
    api = FakeApi()
    with pytest.raises(RuntimeError, match="Commit your code first"):
        publish.publish(api, "me/bench", *built)
    assert api.calls == []  # nothing touched the Hub


def test_publish_refuses_failed_contract(built, monkeypatch):
    monkeypatch.setattr(publish, "code_revision", lambda: "abc1234")
    monkeypatch.setattr(
        checks, "run", lambda *a: {"ids_unique": ["synth/x/f"], "schema": [], "synth_shares": []}
    )
    api = FakeApi()
    with pytest.raises(RuntimeError, match=r"Data contract failed.*\['ids_unique'\]"):
        publish.publish(api, "me/bench", *built)
    assert api.calls == []


def test_verify_compares_hub_lfs_hashes():
    record = {"repo_id": "me/bench", "revision": "r1", "sha256": {"train": "aa", "val": "bb"}}

    class Api:
        def get_paths_info(self, repo_id, paths, revision, repo_type):
            assert (repo_id, revision, repo_type) == ("me/bench", "r1", "dataset")
            lfs = {"data/train.parquet": "aa", "data/val.parquet": "XX"}
            return [SimpleNamespace(path=p, lfs=SimpleNamespace(sha256=lfs[p])) for p in paths]

    assert publish.verify(Api(), record) == ["data/val.parquet: hub XX, local bb"]


def test_pinned_is_a_full_commit_with_every_split():
    assert re.fullmatch(r"[0-9a-f]{40}", publish.PINNED["revision"])  # a branch name would move
    assert set(publish.PINNED["sha256"]) == {"train", "val", "test"}


def test_split_counts(built):
    out, _ = built
    assert publish.split_counts(out) == {
        "train": {"synth": 1},
        "val": {"synth": 1},
        "test": {"reid2019": 1, "synth": 1},
    }
