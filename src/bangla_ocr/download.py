"""Download raw source files with checksum verification."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from bangla_ocr.sources import FONTS, REID2019, WIKI_BN, Source

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
    """Download `source` into raw_dir. Zips are extracted to raw_dir/<name>/.

    Returns the extract dir for zips, else the downloaded file.
    """
    path = raw_dir / source.filename
    digest = download(source.url, path, source.sha256)
    if source.sha256 is None:
        print(f"Not pinned yet. Set sha256 for {source.name} in sources.py:\n  {digest}")
    if path.suffix != ".zip":
        return path

    out = raw_dir / source.name
    if not out.exists():
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
    return out


if __name__ == "__main__":
    for src in (REID2019, WIKI_BN, *FONTS):
        print(f"{src.name}: {fetch(src)}")
