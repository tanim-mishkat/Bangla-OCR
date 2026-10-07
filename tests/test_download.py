import hashlib

import pytest

from bangla_ocr.download import download

CONTENT = b"hello bangla"
CONTENT_SHA = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def src_url(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(CONTENT)
    return src.as_uri()  # file:// URL, so tests need no network


def test_download_returns_sha_and_writes_file(src_url, tmp_path):
    dest = tmp_path / "out" / "file.bin"
    assert download(src_url, dest) == CONTENT_SHA
    assert dest.read_bytes() == CONTENT


def test_download_mismatch_raises_and_leaves_nothing(src_url, tmp_path):
    dest = tmp_path / "file.bin"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(src_url, dest, sha256="0" * 64)
    assert list(tmp_path.glob("file.bin*")) == []


def test_download_skips_when_already_valid(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(CONTENT)
    assert download("file:///does/not/exist", dest, CONTENT_SHA) == CONTENT_SHA
