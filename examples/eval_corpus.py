"""Evaluate Col-Bandit on BEIR / REAL-MM-RAG bundles (Overlap@K vs Full-MaxSim, nDCG@K, Recall@K, MRR@K).

Uses the project's `utils.dataset_registry.load_bundle(...)` which returns
``{passage_embeddings, query_embeddings, qrels, passage_ids|doc_ids, query_ids}``
(BEIR text bundles, ColPali / GVE REAL-MM-RAG bundles, etc.).

Run from any shell — paths are absolute.

    PROJECT=/dccstor/ocr-ai/pony/projects/colbert_pac_prune
    NKLIB=$PROJECT/_numkong_build_lib
    COLB=/dccstor/ocr-ai/pony/projects/colbandit
    ENV=/u/pony/miniforge3/envs/colbert_pac_prune/bin/python

    PYTHONPATH=$NKLIB:$COLB:$PROJECT $ENV $COLB/examples/eval_corpus.py \\
        --dataset arguana --encoder lightonai_colbertv2 --k 5 --alpha-ef 0.2
"""
from __future__ import annotations
import argparse, math, sys, time
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

from colbandit import ColBandit


# --------------------------------------------------------------------- helpers
def _key(d: dict, *names: str):
    """Return the first key in *names that exists in dict d (or None)."""
    for n in names:
        if n in d:
            return n
    return None


def _to_np_f32(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


def _split_padded(docs_padded, doc_lengths):
    """[N, max_L, d] padded tensor + lengths -> list[ [L_i, d] ]."""
    out = []
    L = doc_lengths.tolist() if isinstance(doc_lengths, torch.Tensor) else list(doc_lengths)
    for i in range(docs_padded.shape[0]):
        out.append(_to_np_f32(docs_padded[i, : int(L[i])]))
    return out


def _bundle_to_arrays(bundle: dict):
    """Normalise the bundle into (docs: list[np[L,d]], queries: list[np[T,d]], doc_ids, query_ids, qrels)."""
    dkey = _key(bundle, "passage_embeddings", "doc_embeddings", "docs")
    qkey = _key(bundle, "query_embeddings", "queries")
    if dkey is None or qkey is None:
        print("[bundle keys]", list(bundle.keys()))
        raise KeyError("could not locate passage/doc + query embeddings in the bundle")

    docs_raw, qs_raw = bundle[dkey], bundle[qkey]
    dl = bundle.get("doc_lengths") or bundle.get("passage_lengths")
    ql = bundle.get("query_lengths")

    docs = (
        _split_padded(docs_raw, dl) if isinstance(docs_raw, torch.Tensor) and docs_raw.dim() == 3
        else [_to_np_f32(t) for t in docs_raw]
    )
    queries = (
        _split_padded(qs_raw, ql) if isinstance(qs_raw, torch.Tensor) and qs_raw.dim() == 3
        else [_to_np_f32(t) for t in qs_raw]
    )

    di_key = _key(bundle, "passage_ids", "doc_ids", "corpus_ids")
    qi_key = _key(bundle, "query_ids", "qids")
    doc_ids = [str(x) for x in bundle[di_key]] if di_key else [str(i) for i in range(len(docs))]
    query_ids = [str(x) for x in bundle[qi_key]] if qi_key else [str(i) for i in range(len(queries))]

    qrels = bundle.get("qrels") or bundle.get("qrels_dict") or {}
    # qrels can be {qid: {docid: rel}} or list of (qid, docid, rel)
    if isinstance(qrels, list):
        nested = defaultdict(dict)
        for qid, did, rel in qrels:
            nested[str(qid)][str(did)] = int(rel)
        qrels = dict(nested)
    else:
        qrels = {str(q): {str(d): int(r) for d, r in v.items()} for q, v in qrels.items()}

    return docs, queries, doc_ids, query_ids, qrels


# ------------------------------------------------------------------- metrics
def _dcg(rels: Iterable[int], k: int) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(list(rels)[:k]) if r > 0)


def _ndcg_at_k(retrieved_rels, query_qrels, k) -> float:
    idcg = _dcg(sorted(query_qrels.values(), reverse=True), k)
    return (_dcg(retrieved_rels, k) / idcg) if idcg > 0 else 0.0


def _recall_at_k(retrieved_rels, query_qrels, k) -> float:
    n_rel = sum(1 for r in query_qrels.values() if r > 0)
    return (sum(1 for r in retrieved_rels[:k] if r > 0) / n_rel) if n_rel else 0.0


def _mrr_at_k(retrieved_rels, k) -> float:
    for i, r in enumerate(retrieved_rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def _brute_full_maxsim(query_np, docs_np, k):
    """Reference Full-MaxSim ranking for Overlap@K (small N only)."""
    qn = query_np / np.maximum(np.linalg.norm(query_np, axis=-1, keepdims=True), 1e-12)
    totals = np.empty(len(docs_np), dtype=np.float32)
    for i, v in enumerate(docs_np):
        vn = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        totals[i] = (qn @ vn.T).max(axis=1).sum()
    return list(np.argsort(-totals)[:k])


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. arguana, scidocs, hotpotqa, nq, quora, finslides, finreport, techslides, techreport")
    ap.add_argument("--encoder", default="lightonai_colbertv2",
                    help="e.g. lightonai_colbertv2, jina_colbertv2_128, jina_colbertv2_64, colpali, gve")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha-ef", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--project", default="/dccstor/ocr-ai/pony/projects/colbert_pac_prune",
                    help="absolute path to colbert_pac_prune (for utils.dataset_registry)")
    ap.add_argument("--full-maxsim-ref", action="store_true",
                    help="also compute brute-force Full-MaxSim and report Overlap@K (slow for large N)")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all; cap for quick smoke runs")
    args = ap.parse_args()

    if args.project not in sys.path:
        sys.path.insert(0, args.project)
    from utils.dataset_registry import load_bundle  # type: ignore

    print(f"[load] {args.dataset} / {args.encoder}", flush=True)
    bundle = load_bundle(args.dataset, encoder=args.encoder)
    docs, queries, doc_ids, query_ids, qrels = _bundle_to_arrays(bundle)
    N, d = len(docs), docs[0].shape[1]
    print(f"       N={N}  Q={len(queries)}  d={d}  qrels-queries={len(qrels)}", flush=True)

    eval_qs = [(i, qid) for i, qid in enumerate(query_ids)
               if qid in qrels and any(r > 0 for r in qrels[qid].values())]
    if args.max_queries:
        eval_qs = eval_qs[: args.max_queries]
    print(f"       {len(eval_qs)} queries evaluable", flush=True)

    print(f"[index] alpha_ef={args.alpha_ef} M={args.M} delta={args.delta} threads={args.threads}", flush=True)
    t0 = time.time()
    cb = ColBandit(alpha_ef=args.alpha_ef, M=args.M, delta=args.delta,
                   n_threads=args.threads, exact_rescore=True).index(docs, doc_ids=doc_ids)
    print(f"       index built in {time.time()-t0:.1f}s", flush=True)

    K = args.k
    ndcg = rec = mrr = ov = 0.0
    t0 = time.time()
    for qi, qid in eval_qs:
        ids, _ = cb.search(queries[qi], k=K)
        rels = [qrels[qid].get(str(did), 0) for did in ids]
        ndcg += _ndcg_at_k(rels, qrels[qid], K)
        rec  += _recall_at_k(rels, qrels[qid], K)
        mrr  += _mrr_at_k(rels, K)
        if args.full_maxsim_ref:
            ref_pos = _brute_full_maxsim(queries[qi], docs, K)
            ref_ids = {doc_ids[p] for p in ref_pos}
            ov += len(ref_ids & set(ids)) / K
    n = len(eval_qs)
    elapsed = time.time() - t0
    print()
    print(f"=== {args.dataset} / {args.encoder}  @K={K}  (alpha_ef={args.alpha_ef}) ===")
    print(f"  nDCG@{K}   = {ndcg/n:.4f}")
    print(f"  Recall@{K} = {rec/n:.4f}")
    print(f"  MRR@{K}    = {mrr/n:.4f}")
    if args.full_maxsim_ref:
        print(f"  Overlap@{K} vs Full-MaxSim = {ov/n:.4f}")
    print(f"  queries evaluated: {n}   search wall-clock: {elapsed:.1f}s   ({elapsed/n*1000:.1f} ms/query)")


if __name__ == "__main__":
    main()
