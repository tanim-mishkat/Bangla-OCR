import json

import polars as pl
import pytest

from bangla_ocr import predict, score
from bangla_ocr.download import sha256_file


def labels(*texts: str) -> pl.DataFrame:
    n = len(texts)
    return pl.DataFrame(
        {
            "id": [f"r{i}" for i in range(n)],
            "text_id": [f"t{i}" for i in range(n)],
            "source": ["synth"] * n,
            "level": ["clean"] * n,
            "font": ["F"] * n,
            "region_type": [None] * n,
            "text": list(texts),
        },
        schema_overrides={"region_type": pl.String},
    )


def preds(*hyps: str | None, error: str | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [f"r{i}" for i in range(len(hyps))],
            "pred": list(hyps),
            "seconds": [0.1] * len(hyps),
            "error": [error if h is None else None for h in hyps],
        },
        schema=predict.SCHEMA,
    )


def test_perfect_prediction_has_no_edits():
    r = score.score_rows(labels("আমি ভাত খাই"), preds("আমি ভাত খাই")).row(0, named=True)
    assert (r["char_edits"], r["chars"], r["word_edits"], r["words"]) == (0, 11, 0, 3)
    assert r["empty"] is False


@pytest.mark.parametrize("hyp", ["", None])  # empty output and an errored row score the same
def test_empty_or_error_costs_every_label_char(hyp):
    df = score.score_rows(labels("কর্ম"), preds(hyp, error="RuntimeError: x"))
    assert df["char_edits"][0] == df["chars"][0] == 4
    assert df["empty"][0]


def test_prediction_normalised_like_labels():
    # ZWJ + extra whitespace in the output are not OCR errors
    df = score.score_rows(labels("আমি ভাত"), preds("  আমি‍   ভাত\n"))
    assert df["char_edits"][0] == 0


def test_insertions_can_exceed_label_length():
    df = score.score_rows(labels("কখ"), preds("কখগঘঙ"))
    assert df["char_edits"][0] == 3 > df["chars"][0]  # row CER 1.5: not clipped


def test_missing_prediction_refuses_to_score():
    with pytest.raises(ValueError, match="1 ids have no prediction"):
        score.score_rows(labels("ক", "খ"), preds("ক"))


def test_summary_is_micro_averaged():
    rows = score.score_rows(labels("কখগঘঙচছজঝঞ", "কখ"), preds("কখগঘঙচছজঝঞ", ""))
    s = score.summarise(rows, []).row(0, named=True)
    assert s["cer"] == pytest.approx(2 / 12)  # not mean(0, 1) = 0.5
    assert (s["n"], s["empty_rate"], s["errors"]) == (2, 0.5, 0)


def test_score_refuses_predictions_on_other_data(tmp_path):
    split = tmp_path / "test.parquet"
    labels("ক").write_parquet(split)
    run = tmp_path / "run"
    run.mkdir()
    preds("ক").write_parquet(run / "part-00000.parquet")
    (run / "config.json").write_text(json.dumps({"data_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="different data"):
        score.score(run, split)
    (run / "config.json").write_text(json.dumps({"data_sha256": sha256_file(split)}))
    assert score.score(run, split)["char_edits"].to_list() == [0]
