#!/usr/bin/env bash
# Build ExigeScannerCore.xcframework from the shared C++.
# Requires Xcode on macOS. A Linux VM cannot run this.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/ios/build"

if [[ "$(uname -s)" != "Darwin" ]] || ! command -v xcodebuild >/dev/null 2>&1; then
  echo "iOS build needs Xcode (xcodebuild) on macOS." >&2
  echo "This machine did not compile the iOS library." >&2
  echo "The same C++ is $ROOT/src/scanner_core.cpp." >&2
  echo "Swift package: $ROOT/Package.swift (swift build, or open it in Xcode)." >&2
  echo "Toolchain: $ROOT/ios/ios.toolchain.cmake." >&2
  exit 2
fi

build_slice() {
  local name="$1"
  local sysroot="$2"
  local archs="$3"
  local build_dir="$OUT/$name"
  cmake -S "$ROOT" -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/ios/ios.toolchain.cmake" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sysroot" \
    -DCMAKE_OSX_ARCHITECTURES="$archs" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF >&2
  cmake --build "$build_dir" --config Release >&2
  local library
  library="$(find "$build_dir" -name 'libscanner_core.a' -print | head -n 1)"
  if [[ -z "$library" ]]; then
    echo "libscanner_core.a was not produced for $name" >&2
    exit 1
  fi
  echo "$library"
}

device="$(build_slice iphoneos iphoneos arm64)"
simulator="$(build_slice iphonesimulator iphonesimulator "arm64;x86_64")"
rm -rf "$OUT/ExigeScannerCore.xcframework"
xcodebuild -create-xcframework \
  -library "$device" -headers "$ROOT/src/include" \
  -library "$simulator" -headers "$ROOT/src/include" \
  -output "$OUT/ExigeScannerCore.xcframework"

echo "wrote $OUT/ExigeScannerCore.xcframework"
