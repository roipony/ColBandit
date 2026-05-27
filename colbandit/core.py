"""Col-Bandit: query-time top-K identification for late-interaction (ColBERT-style) retrieval.

A thin, dependency-light wrapper over the ``numkong`` CB-NK kernel. It operates on
*precomputed* multi-vector token embeddings (it is a scoring/reranking layer, not an
encoder) and exposes two methods:

    cb = ColBandit(alpha_ef=0.2, M=5, delta=0.01)
    cb.index(doc_embeddings)                 # list of [L_i, d] float32 (one array per document)
    ids, scores = cb.search(query_emb, k=5)  # query [T, d] float32 -> top-k document ids

Runs on x86 (AVX2) and Apple Silicon (NEON) via the numkong kernel.
"""

from __future__ import annotations

import numpy as np

try:
    import numkong as _nk
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "colbandit requires the 'numkong' kernel. Build/install it (vendored under "
        "native/numkong), or set PYTHONPATH to a prebuilt numkong*.so."
    ) from e

__all__ = ["ColBandit"]


def _to_f32(x) -> np.ndarray:
    """Accept a numpy array or torch tensor -> contiguous float32 numpy array."""
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _quantize_i8(v: np.ndarray) -> np.ndarray:
    """p95-clip int8 quantization matching numkong's pack (6th-largest |value| as scale)."""
    abs_v = np.abs(v)
    k = min(6, v.shape[1])
    scale = np.partition(abs_v, -k, axis=1)[:, -k]            # k-th largest |value| per row
    scale = np.maximum(scale, 1e-10)[:, None]
    return np.clip(np.round(v * 79.0 / scale), -79, 79).astype(np.int8)


class ColBandit:
    """Query-time top-K estimator over an indexed set of multi-vector documents.

    Parameters
    ----------
    alpha_ef : float, default 0.2
        Relaxation knob in (0, 1]. Smaller prunes more aggressively (lower fidelity,
        higher speed); ``alpha_ef=1`` is the conservative (PAC) corner.
    M : int, default 5
        Rescore margin: keep K+M survivors and rescore them exactly.
    delta : float, default 0.01
        Per-document confidence parameter.
    n_threads : int, default 1
        OpenMP threads (parallelism is across queries).
    rng_seed : int, default 42
        Seed for the shared token-reveal permutation.
    exact_rescore : bool, default True
        If True, also build the packed representation so the K+M rescore uses the same
        kernel as exhaustive MaxSim (top-K bit-identical to Full-MaxSim). Uses more memory.
    """

    def __init__(self, alpha_ef: float = 0.2, M: int = 5, delta: float = 0.01,
                 n_threads: int = 1, rng_seed: int = 42, exact_rescore: bool = True):
        self.alpha_ef = float(alpha_ef)
        self.M = int(M)
        self.delta = float(delta)
        self.n_threads = int(n_threads)
        self.rng_seed = int(rng_seed)
        self.exact_rescore = bool(exact_rescore)
        self._reset()

    def _reset(self):
        self._i8 = self._f32 = self._inv = self._offsets = self._sum_i8 = None
        self._packed = None
        self._doc_ids = None
        self.depth = None
        self.n_docs = 0

    # ------------------------------------------------------------------ index
    def index(self, doc_embeddings, doc_ids=None) -> "ColBandit":
        """Build the packed index from per-document token embeddings.

        Parameters
        ----------
        doc_embeddings : sequence of arrays, each [L_i, d]
            One float32 (or torch) array per document. ``L_i`` may vary per document;
            ``d`` (embedding dim) must be constant. A single [N, L, d] array is also accepted.
        doc_ids : optional sequence
            External ids returned by ``search``; defaults to 0..N-1.
        """
        if isinstance(doc_embeddings, np.ndarray) and doc_embeddings.ndim == 3:
            doc_embeddings = [doc_embeddings[i] for i in range(doc_embeddings.shape[0])]
        docs = list(doc_embeddings)
        if not docs:
            raise ValueError("doc_embeddings is empty")

        i8_parts, f32_parts, inv_parts, offsets = [], [], [], [0]
        packed = [] if self.exact_rescore else None
        depth = None
        for d_emb in docs:
            v = _l2_normalize(_to_f32(d_emb))
            if v.ndim != 2:
                raise ValueError(f"each document must be 2D [L, d], got shape {v.shape}")
            if depth is None:
                depth = v.shape[1]
            elif v.shape[1] != depth:
                raise ValueError(f"inconsistent embedding dim: {v.shape[1]} vs {depth}")
            i8_parts.append(_quantize_i8(v))
            f32_parts.append(v)
            inv_parts.append(np.ones(v.shape[0], dtype=np.float32))  # rows are unit-norm
            offsets.append(offsets[-1] + v.shape[0])
            if packed is not None:
                packed.append(_nk.maxsim_pack(v, dtype="f32"))

        self._i8 = np.concatenate(i8_parts, axis=0)
        self._f32 = np.concatenate(f32_parts, axis=0)
        self._inv = np.concatenate(inv_parts, axis=0)
        self._offsets = np.asarray(offsets, dtype=np.int32)
        self._sum_i8 = (128 * self._i8.astype(np.int32).sum(axis=1)).astype(np.int32)  # bias term
        self._packed = packed
        self.depth = depth
        self.n_docs = len(docs)
        self._doc_ids = list(doc_ids) if doc_ids is not None else list(range(self.n_docs))
        if len(self._doc_ids) != self.n_docs:
            raise ValueError("doc_ids length must match number of documents")
        return self

    # ----------------------------------------------------------------- search
    def search(self, query_embedding, k: int = 5):
        """Return (ids, scores): the estimated top-k document ids and their scores.

        ``query_embedding`` is a [T, d] float32 (or torch) array of query token embeddings.
        Scores are MaxSim similarities (higher = better); ids are mapped through ``doc_ids``.
        """
        if self._i8 is None:
            raise RuntimeError("call index(...) before search(...)")
        q = _l2_normalize(_to_f32(query_embedding))
        idx, scores, _stats = _nk.colbandit_flat(
            q, self._i8, self._f32, self._inv, self._offsets, self._sum_i8,
            K=int(k), K_margin=self.M, alpha_ef=self.alpha_ef, delta=self.delta,
            n_threads=self.n_threads, rng_seed=self.rng_seed,
            docs_packed=self._packed,
        )
        ids = [self._doc_ids[i] for i in idx]
        return ids, list(scores)

    # -------------------------------------------------------------- persistence
    def save(self, path: str):
        """Persist the flat index to a .npz file (packed-rescore buffers are rebuilt on load)."""
        if self._i8 is None:
            raise RuntimeError("nothing to save; call index(...) first")
        np.savez(
            path, i8=self._i8, f32=self._f32, inv=self._inv, offsets=self._offsets,
            doc_ids=np.asarray(self._doc_ids), depth=self.depth,
            knobs=np.asarray([self.alpha_ef, self.M, self.delta, self.rng_seed], dtype=np.float64),
            exact_rescore=self.exact_rescore,
        )

    @classmethod
    def load(cls, path: str) -> "ColBandit":
        z = np.load(path, allow_pickle=False)
        knobs = z["knobs"]
        obj = cls(alpha_ef=float(knobs[0]), M=int(knobs[1]), delta=float(knobs[2]),
                  rng_seed=int(knobs[3]), exact_rescore=bool(z["exact_rescore"]))
        obj._i8, obj._f32, obj._inv = z["i8"], z["f32"], z["offsets"] if False else z["inv"]
        obj._offsets = z["offsets"]
        obj._sum_i8 = (128 * obj._i8.astype(np.int32).sum(axis=1)).astype(np.int32)
        obj.depth = int(z["depth"])
        obj._doc_ids = list(z["doc_ids"])
        obj.n_docs = len(obj._doc_ids)
        if obj.exact_rescore:
            # rebuild packed reps from the stored f32 rows
            obj._packed = [
                _nk.maxsim_pack(obj._f32[obj._offsets[i]:obj._offsets[i + 1]], dtype="f32")
                for i in range(obj.n_docs)
            ]
        return obj
