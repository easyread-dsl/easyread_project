#!/usr/bin/env bash
set -euo pipefail

# Download + extract Mulberry Symbols into ../../../data/mulberry (relative to this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(cd "$SCRIPT_DIR/../../../data" && pwd)/mulberry"
URL="https://globalsymbols.com/symbolsets/mulberry/download?locale=en"

mkdir -p "$OUT_DIR"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "[info] Downloading Mulberry dataset..."
cd "$TMP_DIR"

# -L follows redirects
# -J/-O use the server-provided filename from Content-Disposition when available
curl --fail --location --retry 5 --retry-delay 2 \
  -A "Mozilla/5.0" \
  -J -O "$URL"

ARCHIVE="$(ls -1 | head -n 1)"
if [[ -z "${ARCHIVE:-}" ]]; then
  echo "[error] Download did not produce a file."
  exit 1
fi

echo "[info] Downloaded: $ARCHIVE"
echo "[info] Extracting to: $OUT_DIR"

# Handle common archive formats (Mulberry is typically a .zip, but redirects could change this)
case "$ARCHIVE" in
  *.zip)
    unzip -q "$ARCHIVE" -d "$OUT_DIR"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE" -C "$OUT_DIR"
    ;;
  *.tar.bz2|*.tbz2)
    tar -xjf "$ARCHIVE" -C "$OUT_DIR"
    ;;
  *.tar.xz|*.txz)
    tar -xJf "$ARCHIVE" -C "$OUT_DIR"
    ;;
  *)
    echo "[error] Unknown archive type: $ARCHIVE"
    echo "        Please open $URL in a browser and check what file type is provided."
    exit 1
    ;;
esac

echo "[done] Mulberry extracted into: $OUT_DIR"
