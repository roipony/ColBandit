---
title: Col-Bandit Live Demo
emoji: 🪄
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Query-time top-K id cuts MaxSim FLOPs ~5x on ColBERT
---

# Col-Bandit Live Demo

Interactive demo of [Col-Bandit](https://github.com/roipony/ColBandit) — query-time
top-K identification for ColBERT-style late-interaction retrieval. Run a real query
against a precomputed SciDocs index and watch the candidate population crash from
~1000 documents down to the top-K survivors across a handful of elimination rounds.

Three methods race side by side on the same query:

| Method | What it does |
|---|---|
| **CB-NK (Col-Bandit + numkong)** | Progressive elimination, reveals only the cells it needs |
| **Full-MaxSim (packed)** | Exhaustive: scores every (doc, query token) cell |
| **maxsim-cpu (Rust AVX2)** | Optimized exhaustive baseline |

Metrics reported: wall-clock latency, speedup, MaxSim-matrix coverage (% of cells
revealed), and `Overlap@K` against the exhaustive top-K.

## Deploying this Space

```bash
# 1) clone the Space repo from huggingface.co/spaces/<you>/colbandit-demo
git clone https://huggingface.co/spaces/<you>/colbandit-demo
cd colbandit-demo

# 2) copy the contents of this directory
cp -R /path/to/colbandit/hf_space/* .

# 3) drop your scidocs_demo.npz into demo_data/ (optional —
#    app.py falls back to a synthetic fixture if the file is absent)
cp /your/path/scidocs_demo.npz demo_data/

# 4) push
git lfs install
git lfs track "*.npz"
git add . && git commit -m "deploy Col-Bandit demo" && git push
```

Pick **Persistent CPU (free)** in Space hardware settings to avoid cold-start
spinups. **ZeroGPU** is overkill unless you also add live ColBERT encoding.

## Local preview

```bash
pip install -r requirements.txt
python app.py
# -> http://localhost:7860
```
