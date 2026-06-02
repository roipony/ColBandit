#!/usr/bin/env bash
# Full sanity job: fresh venv -> ./install.sh -> tests -> benchmark.
# CPU-only. Logs to LSF stdout/stderr (~/jobs_log/<JOBID>_log.txt).
set -uo pipefail
TAG="[cb-sanity]"
echo "$TAG === $(date) on $(hostname) ==="
echo "$TAG CPU: $(grep -c ^processor /proc/cpuinfo) logical cores"
lscpu | grep -E "Model name|Flags" | head -2

# Make sure cargo (rustup-installed under /u/pony) and the project python are visible.
export RUSTUP_HOME=/u/pony/.rustup
export CARGO_HOME=/u/pony/.cargo
export PATH="$CARGO_HOME/bin:/u/pony/miniforge3/envs/colbert_pac_prune/bin:$PATH"

echo "$TAG --- toolchain ---"
which python; python --version
which cargo; cargo --version
which gcc;   gcc --version | head -1

REPO=/dccstor/ocr-ai/pony/projects/colbandit
cd "$REPO"

# Fresh venv so we don't disturb the project env.
VENV=/tmp/cb_venv_$$
echo "$TAG creating venv at $VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools

echo
echo "$TAG ============ STEP 1: ./install.sh (vendored numkong + colbandit) ============"
t0=$SECONDS
bash install.sh; RC_INSTALL=$?
echo "$TAG install.sh wall-clock: $((SECONDS-t0))s, exit=$RC_INSTALL"

echo
echo "$TAG ============ STEP 2: smoke tests ============"
python tests/test_smoke.py; RC_TEST=$?
echo "$TAG smoke tests exit=$RC_TEST"

echo
echo "$TAG ============ STEP 3: small benchmark (N=5000, 1 thread) ============"
python examples/bench.py; RC_BENCH=$?
echo "$TAG bench (small) exit=$RC_BENCH"

echo
echo "$TAG ============ STEP 4: larger benchmark (N=50000, 8 threads) ============"
python examples/bench.py --N 50000 --threads 8; RC_BIG=$?
echo "$TAG bench (large) exit=$RC_BIG"

echo
echo "$TAG ============ SUMMARY ============"
printf "  install.sh : %s\n  smoke tests: %s\n  bench small: %s\n  bench large: %s\n" \
  "$([ $RC_INSTALL -eq 0 ] && echo OK || echo FAIL\(exit=$RC_INSTALL\))" \
  "$([ $RC_TEST    -eq 0 ] && echo OK || echo FAIL\(exit=$RC_TEST\))" \
  "$([ $RC_BENCH   -eq 0 ] && echo OK || echo FAIL\(exit=$RC_BENCH\))" \
  "$([ $RC_BIG     -eq 0 ] && echo OK || echo FAIL\(exit=$RC_BIG\))"

deactivate || true
rm -rf "$VENV"
