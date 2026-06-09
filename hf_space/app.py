"""Col-Bandit live demo — Gradio app.

Three methods race on a real query against a precomputed SciDocs corpus:
  * CB-NK (Col-Bandit + numkong)   — progressive elimination
  * Full-MaxSim (packed kernel)    — exhaustive
  * maxsim-cpu (Rust AVX2)         — alternative exhaustive baseline

Renders:
  (1) Survivors waterfall — one vertical bar per round, height = # docs alive,
      bar segmented by the round in which each surviving doc later got eliminated.
      Round 0 starts at N, the final round ends at K. Coloured rings on top
      mark which round each elimination cohort fell in.
  (2) Wall-clock comparison.
  (3) Metrics: speedup, Overlap@K vs exhaustive, % of MaxSim matrix revealed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Must be set before first maxsim_cpu call.
os.environ.setdefault("RAYON_NUM_THREADS", str(min(os.cpu_count() or 4, 16)))

import numpy as np
import gradio as gr
import plotly.graph_objects as go

from colbandit import (
    colbandit_flat, full_maxsim, maxsim_pack,
    extract_flat_from_packed, total_tokens,
)
try:
    import maxsim_cpu
    HAVE_MAXSIM_CPU = True
except ImportError:
    HAVE_MAXSIM_CPU = False


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_NPZ = HERE / "demo_data" / "scidocs_demo.npz"


def _load_real_corpus(max_docs: int = 1000):
    z = np.load(DATA_NPZ, allow_pickle=False)
    doc_emb = z["doc_emb_fp16"].astype(np.float32)
    q_emb   = z["query_emb_fp16"].astype(np.float32)
    doc_off, q_off = z["doc_offsets"], z["query_offsets"]
    doc_off = doc_off[: max_docs + 1]
    docs    = [doc_emb[doc_off[i]:doc_off[i + 1]] for i in range(len(doc_off) - 1)]
    queries = [q_emb[q_off[i]:q_off[i + 1]]       for i in range(len(q_off) - 1)]
    return docs, queries, "SciDocs (cached)"


def _load_synthetic_corpus(N: int = 800, T: int = 32, d: int = 128, K_winners: int = 5):
    rng = np.random.default_rng(0)
    docs = [(0.15 * rng.standard_normal((40 + (i % 25), d))).astype(np.float32)
            for i in range(N)]
    queries = []
    for qi in range(20):
        q = rng.standard_normal((T, d)).astype(np.float32)
        # plant a different set of winners per query so it's not always the same docs
        winners = rng.choice(N, size=K_winners, replace=False)
        for w_idx, w in enumerate(winners):
            docs[w] = (q[rng.integers(0, T, 40)] + 0.05 * rng.standard_normal((40, d))).astype(np.float32)
        queries.append(q)
    return docs, queries, "synthetic (no SciDocs file found)"


def load_corpus():
    if DATA_NPZ.exists():
        try:
            return _load_real_corpus()
        except Exception as e:
            print(f"[demo] failed to load {DATA_NPZ}: {e}; using synthetic fallback")
    return _load_synthetic_corpus()


# ---------------------------------------------------------------------------
# Index packing (one-time)
# ---------------------------------------------------------------------------

print(">>> loading corpus …")
DOCS, QUERIES, CORPUS_NAME = load_corpus()
N_DOCS = len(DOCS)
N_QUERIES = len(QUERIES)
print(f">>> {CORPUS_NAME}: {N_DOCS} docs, {N_QUERIES} queries")

print(">>> packing index …")
DOCS_PACKED = [maxsim_pack(d, dtype="f32") for d in DOCS]
_total = total_tokens(DOCS_PACKED)
DIM = DOCS[0].shape[-1]
I8  = np.empty((_total, DIM), np.int8)
F32 = np.empty((_total, DIM), np.float32)
INV = np.empty(_total, np.float32)
SM  = np.empty(_total, np.int32)
OFF = np.zeros(N_DOCS + 1, np.int32)
extract_flat_from_packed(DOCS_PACKED, I8, F32, INV, SM, OFF)
DOCS_LIST_F32 = [np.ascontiguousarray(d, dtype=np.float32) for d in DOCS]  # for maxsim-cpu

# warmup
_ = colbandit_flat(QUERIES[0].astype(np.float32), I8, F32, INV, OFF, SM,
                  K=5, K_margin=5, alpha_ef=0.2, delta=0.01, n_threads=1,
                  docs_packed=DOCS_PACKED, round_size=4)
if HAVE_MAXSIM_CPU:
    _ = maxsim_cpu.maxsim_scores_variable(QUERIES[0].astype(np.float32), DOCS_LIST_F32)
print(">>> ready")

ROUND_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


# ---------------------------------------------------------------------------
# Backend: one query × three methods
# ---------------------------------------------------------------------------

def run_one_query(query_idx: int, K: int, alpha_ef: float, n_threads: int):
    q = QUERIES[int(query_idx)].astype(np.float32)
    out: dict[str, Any] = {"corpus": CORPUS_NAME, "N": N_DOCS, "T": int(q.shape[0]),
                           "K": int(K), "alpha_ef": float(alpha_ef), "n_threads": int(n_threads)}

    # CB-NK (real wall-clock)
    t0 = time.perf_counter()
    idx_cb, _, st = colbandit_flat(
        q, I8, F32, INV, OFF, SM,
        K=int(K), K_margin=5, alpha_ef=float(alpha_ef), delta=0.01,
        n_threads=int(n_threads), docs_packed=DOCS_PACKED, round_size=4,
    )
    out["t_cb_ms"]  = (time.perf_counter() - t0) * 1000.0
    out["idx_cb"]   = [int(x) for x in idx_cb]
    # Real per-round telemetry straight from the C kernel.
    out["coverage"]  = float(st["coverage"])
    out["survived"]  = int(st.get("survived", N_DOCS))
    survivors = [N_DOCS] + [int(x) for x in st.get("round_n_survivors", [])]
    if survivors[-1] > int(K):
        survivors.append(int(K))    # final exact-rescore phase drops to K
    out["survivors"]      = survivors
    out["round_tokens"]   = [int(x) for x in st.get("round_tokens", [])]
    out["round_kernel_ms"] = [float(x) for x in st.get("round_kernel_ms", [])]
    out["round_elim_ms"]   = [float(x) for x in st.get("round_elim_ms", [])]

    # Full-MaxSim (packed)
    t0 = time.perf_counter()
    idx_ex, _, _ = full_maxsim(q, DOCS_PACKED, K=int(K), n_threads=int(n_threads))
    out["t_full_ms"] = (time.perf_counter() - t0) * 1000.0
    out["idx_full"]  = [int(x) for x in idx_ex]

    # maxsim-cpu (optional)
    if HAVE_MAXSIM_CPU:
        q_c = np.ascontiguousarray(q, dtype=np.float32)
        t0 = time.perf_counter()
        scores = maxsim_cpu.maxsim_scores_variable(q_c, DOCS_LIST_F32)
        out["t_mscpu_ms"] = (time.perf_counter() - t0) * 1000.0
        top = np.argpartition(-scores, int(K))[: int(K)]
        out["idx_mscpu"] = [int(x) for x in top[np.argsort(-scores[top])]]
    else:
        out["t_mscpu_ms"] = None
        out["idx_mscpu"] = []

    ref = set(out["idx_full"])
    out["ov_cb"]    = len(set(out["idx_cb"])    & ref) / max(1, int(K))
    out["ov_mscpu"] = len(set(out["idx_mscpu"]) & ref) / max(1, int(K)) if out["idx_mscpu"] else None
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def waterfall_figure(survivors: list[int], K: int):
    """Vertical bars over rounds; height = #survivors at that round.

    Each bar is segmented bottom-up by which round those docs were eliminated in:
    the bottom segment is "still alive at the end" (== K), the next segment is
    "eliminated in the final round", etc. That way the eye reads the population
    cascade as a stack of coloured cohorts.
    """
    if not survivors:
        return go.Figure()
    R = len(survivors)
    fig = go.Figure()

    # Bottom segment of every bar = the K survivors that lived to the end.
    survivors = list(survivors)
    survivors[-1] = max(survivors[-1], K)  # safety
    bottom_height = K

    # For each round bar r (x-axis position r), the *segments* are:
    #   - K winners (bottom, white/yellow)
    #   - For each later round r' > r: number of docs eliminated in r'
    #   - The "still alive at round r" total equals sum of segments
    fig.add_trace(go.Bar(
        name=f"Top-{K} (winners)",
        x=list(range(R)),
        y=[bottom_height] * R,
        marker=dict(color="#fde725", line=dict(width=0)),
        hovertemplate="Round %{x}<br>%{y} eventual winners<extra></extra>",
    ))
    # Eliminations per round: alive[r] - alive[r+1]
    for elim_round in range(R - 1, 0, -1):
        eliminated_here = survivors[elim_round - 1] - survivors[elim_round]
        if eliminated_here <= 0:
            continue
        # Present on bars 0 .. elim_round-1 (these are the rounds where these docs were still alive)
        y_segment = [eliminated_here if r < elim_round else 0 for r in range(R)]
        fig.add_trace(go.Bar(
            name=f"Eliminated in round {elim_round}",
            x=list(range(R)),
            y=y_segment,
            marker=dict(color=ROUND_COLORS[elim_round % len(ROUND_COLORS)], line=dict(width=0)),
            hovertemplate=f"Round %{{x}}<br>{eliminated_here} docs (eliminated in round {elim_round})<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=f"Survivors per round — N={survivors[0]} → K={K}",
        xaxis=dict(title="Round", tickmode="linear", dtick=1),
        yaxis=dict(title="# documents still alive"),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=40, r=20, t=60, b=80),
        height=440,
    )
    # Annotate the survivor counts on top of each stacked bar
    for r in range(R):
        fig.add_annotation(x=r, y=survivors[r], text=f"<b>{survivors[r]}</b>",
                          showarrow=False, yshift=8, font=dict(size=12))
    return fig


def round_timing_figure(kernel_ms: list[float], elim_ms: list[float], survivors: list[int]):
    """Per-round wall-clock: kernel-time + elim-time stacked, with #survivors annotated.

    Tells the "doing less work each round" story directly — as the active set
    shrinks the kernel-time per round drops, often by an order of magnitude
    between the first and last rounds.
    """
    if not kernel_ms:
        return go.Figure()
    R = len(kernel_ms)
    elim = (elim_ms + [0.0] * R)[:R]  # pad in case kernel has one extra round
    rounds = list(range(R))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="kernel (SIMD MaxSim on survivors)",
        x=rounds, y=kernel_ms, marker=dict(color="#7e57c2"),
        hovertemplate="round %{x}<br>kernel %{y:.2f} ms<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="elimination (LCB/UCB sweep)",
        x=rounds, y=elim, marker=dict(color="#ffb74d"),
        hovertemplate="round %{x}<br>elim %{y:.2f} ms<extra></extra>",
    ))
    # Annotate #survivors above each bar
    totals = [kernel_ms[r] + elim[r] for r in range(R)]
    surv_after = (survivors[1:] if len(survivors) > R else survivors)[:R]
    for r in range(R):
        if r < len(surv_after):
            fig.add_annotation(x=r, y=totals[r], text=f"{surv_after[r]} alive",
                              showarrow=False, yshift=10, font=dict(size=10, color="#555"))
    fig.update_layout(
        barmode="stack",
        title="Per-round wall-clock — kernel work shrinks as the active set crashes",
        xaxis=dict(title="Round", tickmode="linear", dtick=1),
        yaxis=dict(title="ms"),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=60, b=60),
        height=320,
    )
    return fig


def timing_figure(t_cb: float, t_full: float, t_mscpu: float | None):
    names, vals, colors = [], [], []
    names.append("CB-NK<br>(Col-Bandit)"); vals.append(t_cb);     colors.append("#7e57c2")
    names.append("Full-MaxSim<br>(packed)"); vals.append(t_full); colors.append("#ef5350")
    if t_mscpu is not None:
        names.append("maxsim-cpu<br>(Rust AVX2)"); vals.append(t_mscpu); colors.append("#66bb6a")
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker=dict(color=colors),
        text=[f"{v:.1f} ms" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="Wall-clock per query",
        yaxis=dict(title="ms (lower is better)"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        height=320,
    )
    return fig


def metrics_table(out: dict[str, Any]):
    K = out["K"]
    spd_cb    = out["t_full_ms"] / max(out["t_cb_ms"], 1e-6)
    rows = [
        ["CB-NK",       f'{out["t_cb_ms"]:.2f} ms',  f"{spd_cb:.2f}×",
         f'{out["ov_cb"]:.2f}', f'{out["coverage"]:.1f} %'],
        ["Full-MaxSim", f'{out["t_full_ms"]:.2f} ms', "1.00×", "1.00",   "100.0 %"],
    ]
    if out["t_mscpu_ms"] is not None:
        spd_ms = out["t_full_ms"] / max(out["t_mscpu_ms"], 1e-6)
        rows.append([
            "maxsim-cpu", f'{out["t_mscpu_ms"]:.2f} ms', f"{spd_ms:.2f}×",
            f'{out["ov_mscpu"]:.2f}' if out["ov_mscpu"] is not None else "—",
            "100.0 %",
        ])
    return rows


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def on_run(query_idx, K, alpha_ef, n_threads):
    out = run_one_query(query_idx, K, alpha_ef, n_threads)
    wf   = waterfall_figure(out["survivors"], int(K))
    rt   = round_timing_figure(out["round_kernel_ms"], out["round_elim_ms"], out["survivors"])
    tm   = timing_figure(out["t_cb_ms"], out["t_full_ms"], out["t_mscpu_ms"])
    tbl  = metrics_table(out)
    info = (
        f"**Corpus:** {out['corpus']} — **N={out['N']}** docs, **T={out['T']}** query tokens, "
        f"**K={out['K']}**, **α_ef={out['alpha_ef']}**, threads={out['n_threads']}\n\n"
        f"**Top-K (Full-MaxSim ref):** {out['idx_full']}\n\n"
        f"**Top-K (CB-NK):** {out['idx_cb']}"
    )
    return wf, rt, tm, tbl, info


def build_ui():
    with gr.Blocks(title="Col-Bandit live demo",
                   theme=gr.themes.Soft(primary_hue="purple")) as ui:
        gr.Markdown(
            f"""# Col-Bandit live demo
Watch query-time top-K identification cut MaxSim FLOPs by **~5×** on real ColBERT
embeddings. Pick a query, hit **Run**, and watch the candidate set crash from
**{N_DOCS}** documents down to **K** survivors across a handful of progressive-elimination rounds.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                query_idx = gr.Slider(0, N_QUERIES - 1, value=0, step=1, label="Query #")
                K         = gr.Slider(1, 20, value=10, step=1, label="K (top-K)")
                alpha_ef  = gr.Slider(0.05, 1.0, value=0.2, step=0.05,
                                      label="α_ef (cost↔fidelity)",
                                      info="Smaller prunes more aggressively. 1.0 = δ-PAC certificate.")
                threads   = gr.Slider(1, max(1, os.cpu_count() or 1), value=min(8, os.cpu_count() or 1),
                                      step=1, label="threads")
                btn       = gr.Button("Run", variant="primary")
            with gr.Column(scale=3):
                waterfall = gr.Plot(label="Survivors per round (population crash)")
        with gr.Row():
            round_timing = gr.Plot(label="Per-round timing (kernel + elim)")
        with gr.Row():
            timing = gr.Plot(label="End-to-end wall-clock")
            table  = gr.Dataframe(
                headers=["method", "latency", "speedup vs Full", f"Ov@K", "MaxSim coverage"],
                datatype=["str", "str", "str", "str", "str"],
                label="Per-method metrics",
                interactive=False,
            )
        info = gr.Markdown()

        outputs = [waterfall, round_timing, timing, table, info]
        btn.click(on_run, [query_idx, K, alpha_ef, threads], outputs)
        ui.load(on_run, [query_idx, K, alpha_ef, threads], outputs)
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
