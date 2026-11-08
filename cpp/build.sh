#!/bin/sh
# Builds drumgen (CLI) and example (library demo) with GCC or Clang (Linux, macOS,
# MinGW). No libraries needed.
#   ./build.sh            native build (AVX2+FMA fast path on x86-64 CPUs that have it)
#   ./build.sh portable   no SIMD intrinsics (same results, slower)
set -e
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
FLAGS="-O3 -std=c++17 -pthread"
if [ "$1" = "portable" ]; then
    FLAGS="$FLAGS -DD1D_NO_SIMD"
else
    case "$(uname -m)" in
        x86_64|amd64) FLAGS="$FLAGS -march=native" ;;
        *) FLAGS="$FLAGS -mcpu=native" ;;   # ARM (Apple Silicon, Raspberry Pi): portable path, compiler auto-vectorises
    esac
fi
$CXX $FLAGS drumgen.cpp -o drumgen
$CXX $FLAGS example.cpp -o example
echo "built $(pwd)/drumgen and $(pwd)/example"
