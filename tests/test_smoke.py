"""Smoke tests for Col-Bandit.

Two flavors live here:
  - the original high-level ColBandit clear-winners + save/load tests
  - low-level kernel tests that exercise the raw colbandit._kernel C entry points
    (Fix-A docs_packed kwarg + round_size kwarg + per-round telemetry).
"""
import numpy as np
import colbandit
from colbandit import ColBandit


# ---------------------------------------------------------------------------
# High-level wrapper tests (planted-winner corpus + save/load roundtrip)
# ---------------------------------------------------------------------------

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
    for w in range(n_winners):
        docs[w] = (q[rng.integers(0, T, 60)] + 0.05 * rng.standard_normal((60, d))).astype(np.float32)
    return q, docs


def test_topk_exact_on_clear_winners():
    q, docs = _corpus()
    cb = ColBandit(alpha_ef=0.2, M=5, delta=0.01, exact_rescore=True).index(docs)
    ids, scores = cb.search(q, k=5)
    assert len(ids) == 5 and len(scores) == 5
    overlap = len(set(ids) & _brute_topk(q, docs, 5)) / 5.0
    assert overlap >= 0.8, f"Overlap@5 on clear-winner corpus too low: {overlap}"


def test_save_load_roundtrip():
    import tempfile, os
    q, docs = _corpus(seed=11)
    cb = ColBandit(alpha_ef=0.2, M=5).index(docs)
    ids0, _ = cb.search(q, k=5)
    p = os.path.join(tempfile.mkdtemp(), "idx.npz")
    cb.save(p)
    ids1, _ = ColBandit.load(p).search(q, k=5)
    assert set(ids0) == set(ids1), "save/load changed the top-K"


# ---------------------------------------------------------------------------
# Low-level kernel tests (Fix-A docs_packed + round_size + per-round telemetry)
# ---------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _quantize_i8(v):
    abs_v = np.abs(v)
    k = min(6, v.shape[1])
    scale = np.maximum(np.partition(abs_v, -k, axis=1)[:, -k], 1e-10)[:, None]
    return np.clip(np.round(v * 79.0 / scale), -79, 79).astype(np.int8)


def _build_synth_corpus(N=100, T=32, d=128, seed=0):
    rng = np.random.default_rng(seed)
    q = _unit(rng.standard_normal((T, d)).astype(np.float32))
    docs_f32 = []
    for _ in range(N):
        L = int(rng.integers(40, 81))
        docs_f32.append(_unit(rng.standard_normal((L, d)).astype(np.float32)))
    return q, docs_f32


def test_raw_kernel_fixA_and_telemetry():
    """Round-trip pack -> extract_flat -> full_maxsim -> colbandit_flat(docs_packed=...) -> topm_flat."""
    q, docs_f32 = _build_synth_corpus(N=100, T=32, d=128, seed=0)

    # Step 1: pack each doc through the kernel.
    docs_packed = [colbandit.maxsim_pack(v, dtype="f32") for v in docs_f32]

    # Step 2: extract the flat arrays the bandit needs.
    i8 = np.concatenate([_quantize_i8(v) for v in docs_f32], axis=0)
    f32 = np.concatenate(docs_f32, axis=0)
    inv = np.ones(f32.shape[0], dtype=np.float32)
    offsets = np.zeros(len(docs_f32) + 1, dtype=np.int32)
    for i, v in enumerate(docs_f32):
        offsets[i + 1] = offsets[i] + v.shape[0]
    sum_i8 = (128 * i8.astype(np.int32).sum(axis=1)).astype(np.int32)

    # Step 3: full ground-truth via full_maxsim (the reference exact scorer).
    # full_maxsim returns (indices, scores, stats) for the top-K of its 'K' arg.
    gt_idx, _gt_scores, _gt_stats = colbandit.full_maxsim(q, docs_packed, K=10, n_threads=1)
    gt_top10 = set(int(i) for i in gt_idx)

    # Step 4: colbandit_flat with Fix-A (docs_packed = bit-identical rescore).
    idx, sc, stats = colbandit.colbandit_flat(
        q, i8, f32, inv, offsets, sum_i8,
        K=5, K_margin=5, alpha_ef=0.3, delta=0.01,
        n_threads=1, rng_seed=42,
        docs_packed=docs_packed,
        round_size=4,
    )
    assert len(idx) == 5 and len(sc) == 5
    pred = set(int(i) for i in idx)
    assert pred.issubset(gt_top10), f"colbandit_flat top-5 escaped GT top-10: {pred - gt_top10}"

    # Step 5: per-round telemetry sanity (the three new arrays from this patch).
    for k in ("round_kernel_ms", "round_elim_ms", "round_n_survivors", "round_tokens"):
        assert k in stats, f"per-round telemetry key '{k}' missing from stats"
        assert isinstance(stats[k], list) and len(stats[k]) >= 1, \
            f"stats['{k}'] should be a non-empty list, got {stats[k]!r}"

    # Step 6: topm_flat baseline path (also exercised by the public API).
    tm_idx, tm_sc, _tm_stats = colbandit.topm_flat(
        q, i8, f32, inv, offsets, sum_i8,
        K=5, M=10, n_warmup=8, n_threads=1, rng_seed=42,
    )
    assert len(tm_idx) == 5 and len(tm_sc) == 5
    assert all(0 <= int(i) < len(docs_f32) for i in tm_idx), "topm_flat returned out-of-range ids"


if __name__ == "__main__":
    test_topk_exact_on_clear_winners()
    test_save_load_roundtrip()
    test_raw_kernel_fixA_and_telemetry()
    print("smoke tests passed")
