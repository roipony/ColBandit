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
git clone <this repo> colbandit && cd colbandit
./install.sh
```

That's it. `install.sh` builds the vendored numkong kernel (Rust + C, ISA-probed) and
installs the `colbandit` Python package — no `PYTHONPATH`, no library paths.

**Requirements:** a C toolchain, [Rust / cargo](https://rustup.rs), Python ≥ 3.9.
macOS users also need `libomp` (`brew install libomp`).

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
colbandit/                Python package
native/numkong/           vendored CB-NK kernel (Rust + C, ISA-probed)
examples/quickstart.py    minimal example
tests/test_smoke.py       smoke tests
install.sh                one-shot installer
```

## License

Apache-2.0 (matches the bundled `numkong` kernel).
