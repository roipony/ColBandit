# Col-Bandit

**Query-time top-K identification for late-interaction (ColBERT-style) retrieval.**

Col-Bandit accelerates multi-vector reranking by *adaptively* computing only the MaxSim
cells needed to identify the exhaustive top-K, instead of scoring the full `N × T`
query-token × document matrix. It is a drop-in scoring/reranking layer over precomputed
token embeddings — no retraining, no index changes — and runs on x86 (AVX2) and Apple
Silicon (NEON) via the bundled `numkong` kernel.

> Paper: *Col-Bandit: Query-Time Top-K Estimation for Late-Interaction Retrieval.*

## Install

```bash
pip install colbandit          # builds the vendored numkong kernel (AVX2 / NEON)
```

> **Status (v0.1.0):** the Python layer and API are complete and tested. The vendored
> `numkong` kernel build is being wired into `native/numkong` (see Roadmap). Until then,
> point `PYTHONPATH` at a prebuilt `numkong*.so`:
> ```bash
> PYTHONPATH=/path/to/numkong_lib python examples/quickstart.py
> ```

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
queries* (throughput).

## Roadmap

- [ ] Vendor the `numkong` kernel source under `native/` and wire `pip install` to compile it (AVX2 + NEON).
- [ ] macOS (Apple Silicon / NEON) wheel + perf validation.
- [ ] Optional `colbandit[encoder]` extra: encode raw text with a ColBERT model.

## License

Apache-2.0 (matches the bundled `numkong` kernel).
