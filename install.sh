#!/usr/bin/env bash
# One-shot installer for Col-Bandit.
# Builds the vendored numkong CB-NK kernel (Rust + C, AVX2 on x86, NEON on macOS-arm64)
# and installs the colbandit Python package — no PYTHONPATH tricks.
#
# Requirements:
#   - a C toolchain (gcc/clang)
#   - cargo (Rust)            curl https://sh.rustup.rs -sSf | sh
#   - macOS: libomp           brew install libomp
#
# Usage:  ./install.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> [1/2] Building numkong CB-NK kernel (Rust + C, ISA-probed)…"
pip install "$HERE/native/numkong"

echo ">>> [2/2] Installing colbandit"
pip install "$HERE"

echo ">>> Done. Smoke test:"
echo "    python -c 'from colbandit import ColBandit; print(ColBandit.__doc__.splitlines()[0])'"
