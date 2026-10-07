# Fetch FriBiDi DLL so Pillow (raqm) can shape Bangla on Windows.
# Pinned version + checksum; installs into tools/fribidi (gitignored). FriBiDi is LGPL-2.1.
$ErrorActionPreference = "Stop"

$pkg = "mingw-w64-ucrt-x86_64-fribidi-1.0.17-1-any.pkg.tar.zst"
$sha256 = "3581EE03CF83A9BB0E98BF688C4F532A29553500E317532D90746DE295B0F8A8"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "tools\fribidi"
$tmp = Join-Path ([IO.Path]::GetTempPath()) "fribidi-$PID"

New-Item -ItemType Directory -Force $dest, $tmp | Out-Null
try {
    $archive = Join-Path $tmp $pkg
    Invoke-WebRequest -UseBasicParsing "https://repo.msys2.org/mingw/ucrt64/$pkg" -OutFile $archive
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $sha256) { throw "Checksum mismatch for $pkg" }
    tar -xf $archive -C $tmp ucrt64/bin/libfribidi-0.dll
    if ($LASTEXITCODE -ne 0) { throw "tar failed to extract $pkg" }
    Copy-Item (Join-Path $tmp "ucrt64\bin\libfribidi-0.dll") $dest -Force
    Write-Host "Installed libfribidi-0.dll -> $dest"
}
finally {
    Remove-Item -Recurse -Force $tmp
}
