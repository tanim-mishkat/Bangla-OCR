from huggingface_hub import DatasetCard

from bangla_ocr.card import render, split_table

COUNTS = {"train": {"synth": 8}, "val": {"synth": 1}, "test": {"reid2019": 3, "synth": 1}}

# --- split_table (you write it) ---


def test_split_table():
    assert split_table(COUNTS) == "\n".join(
        [
            "| split | reid2019 | synth | total |",
            "|---|---:|---:|---:|",
            "| train | 0 | 8 | 8 |",
            "| val | 0 | 1 | 1 |",
            "| test | 3 | 1 | 4 |",
            "| **total** | 3 | 10 | 13 |",
        ]
    )


def test_split_table_keeps_split_order_and_sorts_sources():
    table = split_table({"test": {"b": 1, "a": 2}, "train": {"c": 5}})
    lines = table.split("\n")
    assert lines[0] == "| split | a | b | c | total |"
    assert [line.split(" | ")[0] for line in lines[2:]] == ["| test", "| train", "| **total**"]
    assert lines[-1] == "| **total** | 2 | 1 | 5 | 8 |"


def test_split_table_empty_split():
    assert split_table({"train": {"synth": 2}, "val": {}}).split("\n")[3] == "| val | 0 | 0 |"


# --- render ---


def card_text() -> str:
    report = {"input": 20, "empty text": 7, "output": 13}
    sha = {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64}
    return render(COUNTS, report, sha, "abc1234")


def test_card_metadata_is_valid_yaml_with_hf_split_names():
    data = DatasetCard(card_text()).data
    assert data.license == "cc-by-sa-3.0"
    assert data.language == ["bn"]
    files = data.configs[0]["data_files"]
    assert [(f["split"], f["path"]) for f in files] == [
        ("train", "data/train.parquet"),
        ("validation", "data/val.parquet"),
        ("test", "data/test.parquet"),
    ]


def test_card_body_has_pipeline_numbers():
    text = card_text()
    assert "code revision `abc1234`" in text
    assert "| **total** | 3 | 10 | 13 |" in text
    assert "| empty text | 7 |" in text
    assert f"| `data/train.parquet` | `{'a' * 64}` |" in text
