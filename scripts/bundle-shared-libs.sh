#!/bin/sh
# Copy any shared library the packaged binary loads from outside the package
# (MNN, if `ocr-rs` links it as a dylib rather than statically) to sit beside
# it, and repoint the binary at the copy. A package must run from wherever it
# is extracted — never from this build machine's `target/` path. A binary
# that needs no such library (the common case: MNN links in statically)
# exits cleanly with nothing to do.
set -eu

BIN="$1"
DIR=$(dirname "$BIN")

find_libs() {
  case "$(uname -s)" in
    Darwin) otool -L "$BIN" | tail -n +2 | awk '{print $1}' ;;
    Linux) ldd "$BIN" 2>/dev/null | awk '{print $3}' ;;
    *) : ;;
  esac
}

bundled=0
for lib in $(find_libs || true); do
  case "$lib" in
    */libMNN*|*/libmnn*|*/libonnxruntime*)
      base=$(basename "$lib")
      cp "$lib" "$DIR/$base"
      case "$(uname -s)" in
        Darwin)
          install_name_tool -change "$lib" "@executable_path/$base" "$BIN"
          ;;
        Linux)
          if command -v patchelf >/dev/null 2>&1; then
            patchelf --set-rpath '$ORIGIN' "$BIN"
          else
            echo "warning: patchelf not found — $base is bundled but $BIN's rpath was not adjusted" >&2
          fi
          ;;
      esac
      echo "bundled $base beside $(basename "$BIN")"
      bundled=1
      ;;
  esac
done

if [ "$bundled" = "0" ]; then
  echo "$(basename "$BIN") needs no bundled shared library (MNN is statically linked)"
fi
