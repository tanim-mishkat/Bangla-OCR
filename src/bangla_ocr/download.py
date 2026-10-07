"""Download raw source files with checksum verification."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from bangla_ocr.sources import REID2019, Source

CHUNK = 1 << 20  # 1 MiB
RAW_DIR = Path("data/raw")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None) -> str:
    """Stream `url` to `dest` and return its SHA256.

    Skips the download if `dest` exists and matches `sha256`.
    Raises ValueError on checksum mismatch (and leaves no file behind).
    """
    if dest.exists() and sha256 and sha256_file(dest) == sha256:
        return sha256

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        for chunk in iter(lambda: resp.read(CHUNK), b""):
            h.update(chunk)
            f.write(chunk)

    digest = h.hexdigest()
    if sha256 and digest != sha256:
        tmp.unlink()
        raise ValueError(f"Checksum mismatch for {url}: expected {sha256}, got {digest}")
    tmp.replace(dest)
    return digest


def fetch(source: Source, raw_dir: Path = RAW_DIR) -> Path:
    """Download and unzip `source` into raw_dir/<name>/. Returns the extract dir."""
    archive = raw_dir / source.filename
    digest = download(source.url, archive, source.sha256)
    if source.sha256 is None:
        print(f"Not pinned yet. Set sha256 for {source.name} in sources.py:\n  {digest}")

    out = raw_dir / source.name
    if not out.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    return out


if __name__ == "__main__":
    print(f"Extracted to {fetch(REID2019)}")
