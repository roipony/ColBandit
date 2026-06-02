"""Sanity benchmark: ColBandit vs brute-force Full-MaxSim on a planted-winner corpus.

Prints per-query wall-clock latency, speedup, and Overlap@K against brute force.
With clear winners the overlap should be 1.00 — if it isn't, something is wrong.

Usage:
    python examples/bench.py                       # N=5000, K=5
    python examples/bench.py --N 50000 --threads 4
"""
from __future__ import annotations
import argparse, time, numpy as np
from colbandit import ColBandit


def brute_maxsim_topk(query, docs, k):
    qn = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
    totals = np.empty(len(docs), dtype=np.float32)
    for i, v in enumerate(docs):
        vn = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        totals[i] = (qn @ vn.T).max(axis=1).sum()
    return list(np.argsort(-totals)[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--alpha-ef", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10, help="averaged search iterations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N, d, T, K = args.N, args.d, args.T, args.K
    print(f"Corpus: N={N}, T={T}, d={d}, K={K} (alpha_ef={args.alpha_ef}, threads={args.threads})")

    rng = np.random.default_rng(args.seed)
    docs = [(0.15 * rng.standard_normal((60, d))).astype(np.float32) for _ in range(N)]
    query = rng.standard_normal((T, d)).astype(np.float32)
    # plant K clear winners (query-aligned tokens)
    for w in range(K):
        docs[w] = (query[rng.integers(0, T, 60)] + 0.05 * rng.standard_normal((60, d))).astype(np.float32)

    # --- brute-force baseline (Python; slow for large N) -----------------
    t0 = time.time(); ref_ids = brute_maxsim_topk(query, docs, K); t_full = time.time() - t0
    print(f"\n[brute-force Full-MaxSim] {t_full*1000:8.1f} ms / query   top-{K} = {sorted(ref_ids)}")

    # --- ColBandit --------------------------------------------------------
    t0 = time.time()
    cb = ColBandit(alpha_ef=args.alpha_ef, M=5, delta=0.01,
                   n_threads=args.threads, exact_rescore=True).index(docs)
    t_index = time.time() - t0

    cb.search(query, k=K)  # warmup
    t0 = time.time()
    for _ in range(args.iters):
        ids, _ = cb.search(query, k=K)
    t_cb = (time.time() - t0) / args.iters
    overlap = len(set(ids) & set(ref_ids)) / K

    print(f"[Col-Bandit]              {t_cb*1000:8.2f} ms / query   top-{K} = {sorted(ids)}")
    print(f"[Col-Bandit] index build  {t_index*1000:8.1f} ms (one-time)")
    print()
    print(f"  speedup vs brute force = {t_full / t_cb:6.1f}x")
    print(f"  Overlap@{K} vs brute    = {overlap:.2f}  ({'OK' if overlap >= 0.8 else 'LOW'})")


if __name__ == "__main__":
    main()
