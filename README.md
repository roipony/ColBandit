# Col-Bandit

**Query-time top-K identification for late-interaction (ColBERT-style) retrieval.**

Col-Bandit accelerates multi-vector reranking by *adaptively* computing only the MaxSim
cells needed to identify the exhaustive top-K, instead of scoring the full `N × T`
query-token × document matrix. It is a drop-in scoring/reranking layer over precomputed
token embeddings — no retraining, no index changes — and runs on x86 (AVX2) and Apple
Silicon (NEON) via the vendored `numkong` kernel.

> Paper: *Col-Bandit: Query-Time Top-K Estimation for Late-Interaction Retrieval.*

## Install

```bash
pip install colbandit
```

That's it. Precompiled wheels are shipped for Linux x86\_64, macOS arm64,
macOS x86\_64, and Windows x86\_64 (Python 3.9–3.12). The native CB-NK kernel
is bundled inside the package — no separate dependency, no `PYTHONPATH`.

**Source builds** (when no wheel is available) need a C toolchain and Python ≥ 3.9.
macOS users also need `libomp` (`brew install libomp`).

```bash
git clone https://github.com/roipony/ColBandit colbandit && cd colbandit
./install.sh    # or: pip install .
```

## Quickstart

```python
import numpy as np
from colbandit import ColBandit

# Precomputed multi-vector embeddings: one [L_i, d] array per document.
docs  = [np.random.randn(60, 128).astype("float32") for _ in range(10_000)]
query = np.random.randn(32, 128).astype("float32")          # [T, d]

cb = ColBandit(alpha_ef=0.2, M=5, delta=0.01)               # paper-deployed defaults
cb.index(docs)                                              # one-time packing
ids, scores = cb.search(query, k=5)                         # top-5 document ids + scores
```

`index()` and `search()` accept NumPy arrays or PyTorch tensors. A single padded
`[N, L, d]` array is also accepted by `index()`.

## API

| | |
|---|---|
| `ColBandit(alpha_ef=0.2, M=5, delta=0.01, n_threads=1, rng_seed=42, exact_rescore=True)` | Construct. |
| `.index(doc_embeddings, doc_ids=None)` | Pack documents (list of `[L_i, d]`). |
| `.search(query_embedding, k=5) -> (ids, scores)` | Estimate the top-`k`. |
| `.save(path)` / `ColBandit.load(path)` | Persist / restore the packed index. |

**Knobs.** `alpha_ef ∈ (0,1]` is the single cost–fidelity knob: smaller prunes more
aggressively (faster, lower fidelity); `alpha_ef=1` is the conservative corner. `M` is the
rescore margin (keep `K+M` survivors, rescore exactly). `n_threads` parallelises *across
queries* (throughput, not single-query latency).

### Low-level kernel (advanced)

The high-level `ColBandit` wrapper is built on `colbandit._kernel`, also re-exported
on the top-level `colbandit` namespace as `colbandit_flat`, `topm_flat`, `full_maxsim`,
`maxsim_pack`, `extract_flat_from_packed`, and `total_tokens`. Three knobs on
`colbandit_flat` are worth knowing:

  - **`round_size` (default 4)**: tokens revealed per bandit round, `1..16`. Smaller
    means more aggressive elimination — lower coverage floor, but a lower max-coverage
    ceiling too. `B=2` unlocks ~6% min-coverage at 500K; `B=4` is the default sweet
    spot; `B=8+` works better for multimodal corpora.
  - **`docs_packed` (optional)**: pass the same packed-doc list `full_maxsim` takes
    and the `K+M` margin rescore goes through the bit-identical packed kernel
    (Fix-A). Without it the rescore uses the i8 flat tiled kernel, which diverges
    from `full_maxsim` by float-add-order noise; with it, `alpha_ef=1` cleanly
    reaches 99.5–100% overlap (true PAC corner).
  - **Per-round telemetry**: the returned `stats` dict includes
    `round_kernel_ms`, `round_elim_ms`, `round_n_survivors`, and `round_tokens`
    arrays for diagnostics.

## Sanity check

After `./install.sh`:

```bash
# 1) tests (top-K correctness + save/load roundtrip)
python tests/test_smoke.py

# 2) benchmark vs brute-force Full-MaxSim on a planted-winner corpus
python examples/bench.py                          # N=5000, K=5 — quick
python examples/bench.py --N 50000 --threads 4    # larger / multi-thread
```

Both should report **Overlap@5 = 1.00** and a meaningful speedup over brute force.

## Repo layout

```
colbandit/                Python package (ColBandit + _kernel C extension)
native/numkong/           vendored CB-NK kernel sources (C, ISA-probed)
examples/quickstart.py    minimal example
tests/test_smoke.py       smoke tests
setup.py + pyproject.toml standalone build (no separate numkong dep)
install.sh                one-shot source installer
```

## Acknowledgements

Col-Bandit bundles a vendored snapshot of [NumKong](https://github.com/ashvardanian/NumKong)
by Ash Vardanian (Apache-2.0). The progressive-elimination kernel in
[native/numkong/python/colbandit.c](native/numkong/python/colbandit.c) is built
on top of NumKong's SIMD MaxSim primitives. See [NOTICE](NOTICE) for the full
attribution.

## License

Apache-2.0 (matches the bundled CB-NK kernel).
