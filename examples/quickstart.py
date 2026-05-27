"""Minimal Col-Bandit quickstart on synthetic embeddings.

Run (until the vendored numkong build lands, point PYTHONPATH at a prebuilt .so):
    PYTHONPATH=/path/to/numkong_lib:.. python examples/quickstart.py
"""
import numpy as np
from colbandit import ColBandit

rng = np.random.default_rng(0)
N, d, T = 500, 128, 32

# Precomputed multi-vector document embeddings: one [L_i, d] array per document.
docs = [rng.standard_normal((rng.integers(40, 80), d)).astype(np.float32) for _ in range(N)]
query = rng.standard_normal((T, d)).astype(np.float32)

cb = ColBandit(alpha_ef=0.2, M=5, delta=0.01, n_threads=1)
cb.index(docs)
ids, scores = cb.search(query, k=5)

print("top-5 doc ids:", ids)
print("top-5 scores :", [round(s, 4) for s in scores])
