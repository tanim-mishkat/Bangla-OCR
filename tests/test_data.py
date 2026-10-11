"""Data tests: run the contract in bangla_ocr.checks against the REAL processed files.

Skipped if the data isn't built. The same checks run in Dagster as blocking asset checks.
"""

import pytest

from bangla_ocr import checks, publish
from bangla_ocr.download import sha256_file

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in checks.split_paths().values()), reason="processed data not built"
)


@pytest.fixture(scope="module")
def results() -> dict[str, list[str]]:
    return checks.run()


def test_every_check_ran(results):
    assert tuple(results) == checks.NAMES


@pytest.mark.parametrize("name", checks.NAMES)
def test_check_passes(results, name):
    assert results[name] == [], results[name][:5]


def test_local_splits_are_the_pinned_release():
    local = {name: sha256_file(p) for name, p in checks.split_paths().items()}
    assert local == publish.PINNED["sha256"], "rebuilt data differs from the pinned HF revision"
