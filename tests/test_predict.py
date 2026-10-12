import polars as pl
import pytest

from bangla_ocr import predict
from bangla_ocr.predict import Engine


@pytest.fixture
def split(tmp_path):
    path = tmp_path / "test.parquet"
    pl.DataFrame(
        {
            "id": [f"r{i}" for i in range(7)],
            "source": ["synth"] * 6 + ["reid2019"],
            "image": [f"img{i}".encode() for i in range(7)],
        }
    ).write_parquet(path)
    return path


def fake(calls: list, fail: set = frozenset(), config: dict | None = None) -> Engine:
    def read(row):
        calls.append(row["id"])
        if row["id"] in fail:
            raise RuntimeError("rate limited")
        return "" if row["id"] == "r0" else row["image"].decode().upper()

    return Engine("fake", config or {"v": 1}, read)


def test_predicts_every_row_and_keeps_empty_output(split, tmp_path):
    calls = []
    path = predict.run(fake(calls), split, tmp_path / "out", batch=3)
    df = predict.load(path)
    assert sorted(calls) == [f"r{i}" for i in range(7)]
    assert df.schema == predict.SCHEMA
    assert len(list(path.glob("part-*.parquet"))) == 3  # 7 rows in batches of 3
    preds = dict(zip(df["id"], df["pred"], strict=True))
    assert preds["r0"] == "" and preds["r1"] == "IMG1"  # "" is a prediction, not missing
    assert df["error"].null_count() == 7


def test_resume_runs_only_missing_ids(split, tmp_path):
    out = tmp_path / "out"
    predict.run(fake([]), split, out, limit=4)
    calls = []
    predict.run(fake(calls), split, out)
    assert calls == ["r4", "r5", "r6"]


def test_errors_are_recorded_then_retried(split, tmp_path):
    out = tmp_path / "out"
    path = predict.run(fake([], fail={"r2"}), split, out)
    first = predict.load(path).filter(pl.col("id") == "r2").row(0, named=True)
    assert first["pred"] is None and first["error"] == "RuntimeError: rate limited"

    calls = []
    predict.run(fake(calls), split, out)
    assert calls == ["r2"]
    df = predict.load(path)
    assert df.height == 7 and df["error"].null_count() == 7  # retry superseded the error


def test_new_config_never_reuses_old_predictions(split, tmp_path):
    out = tmp_path / "out"
    a = predict.run(fake([], config={"psm": 7}), split, out)
    calls = []
    b = predict.run(fake(calls, config={"psm": 6}), split, out)
    assert a != b and len(calls) == 7
    assert (b / "config.json").read_text(encoding="utf-8").count('"psm": 6') == 1


def test_config_id_ignores_key_order():
    assert predict.config_id({"a": 1, "b": 2}) == predict.config_id({"b": 2, "a": 1})


def test_load_missing_dir_is_empty(tmp_path):
    assert predict.load(tmp_path / "nothing").is_empty()


def test_changed_input_data_never_reuses_old_predictions(split, tmp_path):
    out = tmp_path / "out"
    a = predict.run(fake([]), split, out)
    pl.read_parquet(split).with_columns(image=pl.lit(b"new")).write_parquet(split)
    calls = []
    b = predict.run(fake(calls), split, out)
    assert a != b and len(calls) == 7
    assert b == predict.run_dir(fake([]), split, out)
