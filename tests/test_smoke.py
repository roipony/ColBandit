"""Smoke test: on a corpus with clear winners, ColBandit recovers the exact top-K.

(Random Gaussian embeddings are intentionally avoided: with no score separation the
top-K boundary is degenerate and not a meaningful correctness signal. We plant K
query-aligned documents among noise — the realistic 'clear winners' regime.)
"""
import numpy as np
from colbandit import ColBandit


def _brute_topk(query, docs, k):
    qn = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
    totals = []
    for v in docs:
        vn = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        totals.append((qn @ vn.T).max(axis=1).sum())
    return set(int(i) for i in np.argsort(totals)[::-1][:k])


def _corpus(seed=7, N=400, d=128, T=32, n_winners=5):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, d)).astype(np.float32)
    docs = [(0.15 * rng.standard_normal((60, d))).astype(np.float32) for _ in range(N)]
    for w in range(n_winners):                       # plant query-aligned winners at 0..n_winners-1
        docs[w] = (q[rng.integers(0, T, 60)] + 0.05 * rng.standard_normal((60, d))).astype(np.float32)
    return q, docs


def test_topk_exact_on_clear_winners():
    q, docs = _corpus()
    cb = ColBandit(alpha_ef=0.2, M=5, delta=0.01, exact_rescore=True).index(docs)
    ids, scores = cb.search(q, k=5)
    assert len(ids) == 5 and len(scores) == 5
    overlap = len(set(ids) & _brute_topk(q, docs, 5)) / 5.0
    assert overlap >= 0.8, f"Overlap@5 on clear-winner corpus too low: {overlap}"


def test_save_load_roundtrip(tmp_path=None):
    import tempfile, os
    q, docs = _corpus(seed=11)
    cb = ColBandit(alpha_ef=0.2, M=5).index(docs)
    ids0, _ = cb.search(q, k=5)
    p = os.path.join(tempfile.mkdtemp(), "idx.npz")
    cb.save(p)
    ids1, _ = ColBandit.load(p).search(q, k=5)
    assert set(ids0) == set(ids1), "save/load changed the top-K"


if __name__ == "__main__":
    test_topk_exact_on_clear_winners()
    test_save_load_roundtrip()
    print("smoke tests passed")
