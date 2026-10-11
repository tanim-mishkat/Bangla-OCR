"""Dagster pipeline: each asset wraps an existing build step and records metadata.

Run from the repo root (paths are relative): uv run dagster dev -m bangla_ocr.defs
"""

import json

import dagster as dg
import polars as pl
from huggingface_hub import HfApi

from bangla_ocr import checks, clean, corpus, download, ingest, publish, reid, split, synth
from bangla_ocr.sources import FONTS, REID2019, WIKI_BN

GROUP = "dataset"


def _rows(path) -> int:
    return pl.scan_parquet(path).select(pl.len()).collect().item()


@dg.asset(group_name=GROUP)
def raw_sources() -> dg.MaterializeResult:
    """Downloads, each checked against its pinned sha256."""
    paths = {src.name: str(download.fetch(src)) for src in (REID2019, WIKI_BN, *FONTS)}
    return dg.MaterializeResult(metadata={"files": dg.MetadataValue.json(paths)})


@dg.asset(group_name=GROUP, deps=[raw_sources])
def reid_crops() -> dg.MaterializeResult:
    """REID2019 PAGE XML -> one region crop per row (resumable per page)."""
    reid.build()
    return dg.MaterializeResult(metadata={"rows": _rows(reid.OUT_DIR / "*.parquet")})


@dg.asset(group_name=GROUP, deps=[raw_sources])
def text_corpus() -> dg.MaterializeResult:
    """3000 unique Bangla Wikipedia lines."""
    df = corpus.build_corpus()
    corpus.OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(corpus.OUT_PATH)
    return dg.MaterializeResult(metadata={"lines": df.height})


@dg.asset(group_name=GROUP, deps=[text_corpus, raw_sources])
def synth_images() -> dg.MaterializeResult:
    """Corpus x fonts -> degraded line images (resumable per chunk)."""
    skipped = synth.build()
    return dg.MaterializeResult(
        metadata={"rows": _rows(synth.OUT_DIR / "*.parquet"), "skipped_missing_glyphs": skipped}
    )


@dg.asset(group_name=GROUP, deps=[reid_crops, synth_images])
def ingested() -> dg.MaterializeResult:
    """Both sources in one schema; fails if any row breaks `check_row`."""
    ingest.combine()
    if bad := ingest.validate():
        raise dg.Failure(f"{len(bad)} bad rows, e.g. {list(bad.items())[:3]}")
    counts = pl.scan_parquet(ingest.OUT_PATH).group_by("source").len().collect()
    return dg.MaterializeResult(metadata=dict(counts.iter_rows()))


@dg.asset(group_name=GROUP, deps=[ingested])
def cleaned() -> dg.MaterializeResult:
    """Normalised labels, empties and duplicates dropped, one count per reason."""
    report = clean.clean()
    clean.REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dg.MaterializeResult(metadata=report)


@dg.asset(group_name=GROUP, deps=[cleaned])
def splits() -> dg.MaterializeResult:
    """train/val/test Parquet: sha256 split by text_id, REID test-only."""
    counts = split.split()
    meta = {name: sum(by_source.values()) for name, by_source in counts.items()}
    return dg.MaterializeResult(metadata={**meta, "by_source": dg.MetadataValue.json(counts)})


class HuggingFace(dg.ConfigurableResource):
    token: str
    repo_name: str = publish.REPO_NAME

    def api(self) -> HfApi:
        return HfApi(token=self.token)


@dg.asset(group_name="publish", deps=[splits])
def hf_dataset(hf: HuggingFace) -> dg.MaterializeResult:
    """Private HF dataset repo; runs only if every blocking check on `splits` passed."""
    api = hf.api()
    repo_id = f"{api.whoami()['name']}/{hf.repo_name}"
    record = publish.publish(api, repo_id)
    publish.RECORD_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    url = f"https://huggingface.co/datasets/{repo_id}/tree/{record['revision']}"
    return dg.MaterializeResult(
        metadata={
            "url": dg.MetadataValue.url(url),
            "revision": record["revision"],
            "code_revision": record["code_revision"],
        }
    )


@dg.multi_asset_check(
    specs=[dg.AssetCheckSpec(name, asset=splits, blocking=True) for name in checks.NAMES]
)
def splits_contract():
    """bangla_ocr.checks on the real files; a failure blocks everything downstream (publish)."""
    for name, problems in checks.run().items():
        yield dg.AssetCheckResult(
            asset_key=splits.key,
            check_name=name,
            passed=not problems,
            metadata={"problems": len(problems), "examples": dg.MetadataValue.json(problems[:5])},
        )


defs = dg.Definitions(
    assets=[
        raw_sources,
        reid_crops,
        text_corpus,
        synth_images,
        ingested,
        cleaned,
        splits,
        hf_dataset,
    ],
    asset_checks=[splits_contract],
    resources={"hf": HuggingFace(token=dg.EnvVar("HF_TOKEN"))},
)
