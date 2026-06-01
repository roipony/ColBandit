# Build & Run CB-NK on macOS M1

## 1. Clone from server

```bash
# On your Mac — clone via ssh from the Linux server
git clone ssh://YOUR_SERVER:/u/pony/numkong-src ~/numkong-src
cd ~/numkong-src
git checkout cb-nk-top6-clip95
```

Or if you prefer rsync:
```bash
rsync -avz YOUR_SERVER:/u/pony/numkong-src/ ~/numkong-src/ --exclude build/ --exclude '*.so' --exclude __pycache__
```

## 2. Install dependencies

```bash
# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# OpenMP for multi-threading (optional but recommended)
brew install libomp

# Python env
conda create -n cbnk python=3.11 numpy scipy torch -c pytorch -y
conda activate cbnk
```

## 3. Build

```bash
cd ~/numkong-src
rm -rf build/
python setup.py build_ext --inplace
```

The setup.py auto-detects macOS and:
- Uses ARM NEON kernels (neonsdot) instead of AVX2
- Finds Homebrew libomp at `/opt/homebrew/opt/libomp/`
- Falls back to `NK_NO_OPENMP` if libomp not found

## 4. Quick benchmark

```bash
# Need test data — copy a small dataset from server
rsync -avz YOUR_SERVER:/dccstor/ocr-ai/pony/projects/colbert_pac_prune/colbandit_demo/data/scidocs_embeddings_25K_colbert_v2_prepared.pt ~/numkong-data/

# Edit bench_quick.py paths (or use this one-liner)
python -c "
import numpy as np, torch, torch.nn.functional as F, time, sys
sys.path.insert(0, '.')
import numkong as nk

data = torch.load('$HOME/numkong-data/scidocs_embeddings_25K_colbert_v2_prepared.pt', map_location='cpu', weights_only=False)
de, dl = data['doc_embeddings'], data['doc_lengths'].numpy()
qe, ql = data['query_embeddings'], data['query_lengths'].numpy()
N = de.shape[0]; K = 5
docs = [nk.maxsim_pack(F.normalize(de[i,:dl[i]].float(),dim=-1).numpy(), dtype='f32') for i in range(N)]
print(f'Packed {N} docs')
for qi in range(5):
    Q = F.normalize(qe[qi,:ql[qi]].float(),dim=-1).numpy()
    t0 = time.perf_counter()
    nk_sc = np.array([nk.maxsim_packed(nk.maxsim_pack(Q,dtype='f32'), d) for d in docs])
    t_nk = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter()
    idx, _, st = nk.colbandit_maxsim(Q, docs, K=K, warmup_ratio=1.0, alpha_ef=0.3, delta=0.01)
    t_cb = (time.perf_counter()-t0)*1000
    print(f'Q{qi}: NK={t_nk:.0f}ms CB={t_cb:.0f}ms {t_nk/t_cb:.1f}x cov={st[\"coverage\"]:.0f}%')
"
```

## 5. Expected M1 performance

M1 uses ARM NEON (128-bit SIMD) vs Haswell AVX2 (256-bit). Expect:
- ~2× slower per-doc than Haswell for the i8 coarse screening
- f32 dot products similar (NEON FMA is efficient)
- Overall CB-NK speedup ratio (vs NK full) should be similar (~4-5×)

## Notes
- The `.so` file is platform-specific — don't copy from Linux, always rebuild
- `rm -rf build/` before rebuild after changing `.h` files (setup.py doesn't track headers)
- If OpenMP not found, CB-NK works but n_threads>1 is ignored
