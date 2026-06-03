#!/usr/bin/env bash
# One-shot installer for Col-Bandit (source build).
#
# The vendored C kernel under native/numkong/ is compiled into
# colbandit/_kernel.*.so as part of the colbandit package itself —
# no separate numkong dependency, no PYTHONPATH tricks.
#
# Requirements:
#   - C toolchain (gcc/clang)
#   - macOS:  brew install libomp
#
# Most users should just run:  pip install colbandit
# Use this script only when building from source.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Building + installing colbandit (compiles vendored C kernel)…"
pip install "$HERE"

echo ">>> Done. Smoke test:"
echo "    python -c 'from colbandit import ColBandit; print(ColBandit.__doc__.splitlines()[0])'"
