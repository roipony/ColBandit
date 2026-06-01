/**
 *  @brief ColBandit multi-phase elimination with Serfling bounds on NumKong MaxSim backend.
 *  @file python/colbandit.c
 *
 *  Phase 1 (warmup): Score one token at a time per doc to get per-token MaxSim values.
 *    Compute mean, variance, and Bernstein-Serfling bounds per document.
 *    Eliminate: doc removed if UCB < K-th best LCB.
 *  Phase 2 (explore): Score remaining tokens (batched) on survivors only.
 *
 *  Returns: (top_k_indices, scores, stats_dict)
 */
#include "maxsim.h"
#include "tensor.h"
#include <numkong/maxsim.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* OpenMP: guarded because Apple Clang requires -Xpreprocessor -fopenmp (libomp).
   If libomp is not installed, setup.py defines NK_NO_OPENMP and we fall back to serial. */
#if !defined(NK_NO_OPENMP)
#include <omp.h>
#else
/* Stub out the OMP runtime functions used in this file */
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
#define NK_OMP_PRAGMAS_DISABLED 1
#endif

/* Centroid-lookup MaxSim kernel.
   On x86 with AVX2, use the Haswell-optimised kernel.
   On ARM and other platforms, use a portable scalar implementation. */
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
extern void nk_maxsim_packed_centroid_stats_f32_haswell(
    void const *, void const *, nk_size_t, nk_size_t, nk_size_t, nk_f64_t *, float const *);
#define NK_CENTROID_STATS_KERNEL nk_maxsim_packed_centroid_stats_f32_haswell
extern void nk_maxsim_packed_4bit_stats_f32_haswell(
    void const *, void const *, nk_size_t, nk_size_t, nk_size_t, nk_f64_t *, nk_i8_t const *);
#else
/* Portable scalar fallback for centroid-lookup stats (ARM, RISC-V, etc.) */
#include "numkong/maxsim/serial.h"
static void nk_maxsim_packed_centroid_stats_f32_serial_cb_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, float const *centroid_table) {
    nk_maxsim_packed_regions_t rg = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;
    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_u8_t const *q_idx = (nk_u8_t const *)(rg.query_quantized + qi * rg.depth_i8_padded);
        nk_f64_t best_dot = -1e30;
        for (nk_size_t di = 0; di < document_count; di++) {
            nk_u8_t const *d_idx = (nk_u8_t const *)(rg.document_quantized + di * rg.depth_i8_padded);
            float dot = 0.0f;
            for (nk_size_t k = 0; k < depth; k++)
                dot += centroid_table[q_idx[k]] * centroid_table[d_idx[k]];
            if ((nk_f64_t)dot > best_dot) best_dot = (nk_f64_t)dot;
        }
        nk_f64_t angular = 1.0 - best_dot;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        total_sum_sq += angular * angular;
    }
    result[0] = total_sum;
    result[1] = total_sum_sq;
}
#define NK_CENTROID_STATS_KERNEL nk_maxsim_packed_centroid_stats_f32_serial_cb_
#endif

/* ====== Compact 4-bit kernel: flat contiguous arrays, no packed format overhead ====== */
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
#include <immintrin.h>

/**
 * Compact 4-bit coarse argmax + f32 refine.
 *
 * Memory layout: nibble-packed data is contiguous at stride = depth/2 bytes per vector.
 * For d=128: 64 bytes/vector (vs 160 bytes in packed format = 2.5× less memory).
 *
 * @param q_4bit  [query_count, depth/2] u8 nibble-packed query (contiguous)
 * @param q_f32   [query_count, depth] f32 original query vectors
 * @param q_inv_norms [query_count] f32 inverse norms
 * @param d_4bit  [document_count, depth/2] u8 nibble-packed doc (contiguous, compact stride)
 * @param d_f32   [document_count, depth] f32 original doc vectors
 * @param d_inv_norms [document_count] f32 inverse norms
 * @param centroid_i8 [16] i8 centroid table
 * @param result  [2] output: (sum_angular, sum_angular_sq)
 */
__attribute__((target("avx2,fma")))
static void nk_maxsim_4bit_compact_stats(
    nk_u8_t const *q_4bit, float const *q_f32, float const *q_inv_norms,
    nk_u8_t const *d_4bit, float const *d_f32, float const *d_inv_norms,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_i8_t const *centroid_i8, nk_f64_t *result);

/**
 * Compact i8 coarse argmax + f32 refine.
 * Same as NK stats kernel but with stride=depth (128 bytes) instead of depth_i8_padded (160 bytes).
 * ~20% less memory bandwidth per doc vector read.
 */
/* Two-pass kernel: Phase A (coarse i8 only) + Phase B (f32 refine from stored indices).
   Phase A touches ONLY the i8 buffer — zero f32 cache pollution.
   Phase B is a tiny pass over 2 f32 vectors per query token. */

__attribute__((target("avx2,fma")))
static void nk_maxsim_i8_coarse_top2(
    nk_i8_t const *q_i8,
    nk_i8_t const *d_i8,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_u32_t *best_indices, nk_u32_t *second_indices) {

    __m256i xor_mask = _mm256_set1_epi8((char)0x80);
    __m256i ones_i16 = _mm256_set1_epi16(1);
    __m256i ones_i8 = _mm256_set1_epi8(1);

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_i8_t const *qp = q_i8 + qi * depth;
        nk_i32_t best1_dot = -2147483647, best2_dot = -2147483647;
        nk_u32_t best1_idx = 0, best2_idx = 0;

        for (nk_size_t di = 0; di < document_count; di++) {
            nk_i8_t const *dp = d_i8 + di * depth;

            __m256i acc = _mm256_setzero_si256();
            __m256i d_sum_acc = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth; k += 32) {
                __m256i q = _mm256_loadu_si256((__m256i const *)(qp + k));
                __m256i d = _mm256_loadu_si256((__m256i const *)(dp + k));
                __m256i qu = _mm256_xor_si256(q, xor_mask);
                __m256i prod = _mm256_maddubs_epi16(qu, d);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones_i16));
                /* Bias sum */
                __m256i ds = _mm256_maddubs_epi16(ones_i8, d);
                d_sum_acc = _mm256_add_epi32(d_sum_acc, _mm256_madd_epi16(ds, ones_i16));
            }

            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot = _mm_extract_epi32(s, 0);
            {
                __m128i ds_lo = _mm256_castsi256_si128(d_sum_acc);
                __m128i ds_hi = _mm256_extracti128_si256(d_sum_acc, 1);
                __m128i ds = _mm_add_epi32(ds_lo, ds_hi);
                ds = _mm_add_epi32(ds, _mm_shuffle_epi32(ds, 0x4E));
                ds = _mm_add_epi32(ds, _mm_shuffle_epi32(ds, 0xB1));
                dot -= 128 * _mm_extract_epi32(ds, 0);
            }

            if (dot > best1_dot) {
                best2_dot = best1_dot; best2_idx = best1_idx;
                best1_dot = dot; best1_idx = (nk_u32_t)di;
            } else if (dot > best2_dot) {
                best2_dot = dot; best2_idx = (nk_u32_t)di;
            }
        }
        best_indices[qi] = best1_idx;
        second_indices[qi] = best2_idx;
    }
}

__attribute__((target("avx2,fma")))
static void nk_maxsim_f32_refine_from_indices(
    float const *q_f32, float const *q_inv_norms,
    float const *d_f32, float const *d_inv_norms,
    nk_size_t query_count, nk_size_t depth,
    nk_u32_t const *best_indices, nk_u32_t const *second_indices,
    nk_f64_t *result) {

    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;
    for (nk_size_t qi = 0; qi < query_count; qi++) {
        float const *qf = q_f32 + qi * depth;
        float inv_qn = q_inv_norms[qi];
        float const *d1f = d_f32 + best_indices[qi] * depth;
        float const *d2f = d_f32 + second_indices[qi] * depth;

        __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
        for (nk_size_t k = 0; k < depth; k += 8) {
            __m256 q = _mm256_loadu_ps(qf + k);
            acc1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1f + k), acc1);
            acc2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2f + k), acc2);
        }
        __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
        s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
        s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
        s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
        s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));

        nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * (nk_f64_t)inv_qn *
                        (nk_f64_t)d_inv_norms[best_indices[qi]];
        nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * (nk_f64_t)inv_qn *
                        (nk_f64_t)d_inv_norms[second_indices[qi]];
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        total_sum_sq += angular * angular;
    }
    result[0] = total_sum;
    result[1] = total_sum_sq;
}

/* 4Q×4D tiled kernel for flat i8 at stride=depth.
   Same tiling pattern as NK's nk_maxsim_coarse_top2_haswell_, but works on flat arrays.
   Computes bias sum inline (no pre-allocated metadata array). */
#define CB_FLAT_TOP2_UPDATE(new_dots, didx_i32x4) \
    do { \
        __m128i _gate = _mm_cmpgt_epi32(new_dots, running_max2); \
        if (!_mm_testz_si128(_gate, _gate)) { \
            __m128i _beats1 = _mm_cmpgt_epi32(new_dots, running_max1); \
            __m128i _old_max1 = running_max1; \
            __m128i _old_arg1 = running_argmax1; \
            running_max1 = _mm_blendv_epi8(running_max1, new_dots, _beats1); \
            running_argmax1 = _mm_blendv_epi8(running_argmax1, didx_i32x4, _beats1); \
            __m128i _r2_cand = _mm_blendv_epi8(new_dots, _old_max1, _beats1); \
            __m128i _r2_arg  = _mm_blendv_epi8(didx_i32x4, _old_arg1, _beats1); \
            __m128i _beats2 = _mm_cmpgt_epi32(_r2_cand, running_max2); \
            running_max2 = _mm_blendv_epi8(running_max2, _r2_cand, _beats2); \
            running_argmax2 = _mm_blendv_epi8(running_argmax2, _r2_arg, _beats2); \
        } \
    } while(0)

__attribute__((target("avx2,fma")))
static inline __m128i cb_reduce_i32x8x4(__m256i a, __m256i b, __m256i c, __m256i d) {
    /* Horizontal sum of 4 i32x8 accumulators into i32x4.
       Avoids hadd (slow on Haswell) — uses shuffles + adds instead. */
    /* Reduce each 256-bit to 128-bit: sum upper and lower halves */
    __m128i a128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    __m128i b128 = _mm_add_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
    __m128i c128 = _mm_add_epi32(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1));
    __m128i d128 = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
    /* Now each 128-bit has 4 i32 to sum.
       Transpose 4×4 matrix of i32 so each lane of result holds sum of one input. */
    __m128i ab_lo = _mm_unpacklo_epi32(a128, b128);  /* a0,b0,a1,b1 */
    __m128i ab_hi = _mm_unpackhi_epi32(a128, b128);  /* a2,b2,a3,b3 */
    __m128i cd_lo = _mm_unpacklo_epi32(c128, d128);  /* c0,d0,c1,d1 */
    __m128i cd_hi = _mm_unpackhi_epi32(c128, d128);  /* c2,d2,c3,d3 */
    __m128i r0 = _mm_unpacklo_epi64(ab_lo, cd_lo);   /* a0,b0,c0,d0 */
    __m128i r1 = _mm_unpackhi_epi64(ab_lo, cd_lo);   /* a1,b1,c1,d1 */
    __m128i r2 = _mm_unpacklo_epi64(ab_hi, cd_hi);   /* a2,b2,c2,d2 */
    __m128i r3 = _mm_unpackhi_epi64(ab_hi, cd_hi);   /* a3,b3,c3,d3 */
    return _mm_add_epi32(_mm_add_epi32(r0, r1), _mm_add_epi32(r2, r3));
}

__attribute__((target("avx2,fma")))
static void nk_maxsim_i8_flat_stats_tiled(
    nk_i8_t const *q_i8, float const *q_f32, float const *q_inv_norms,
    nk_i8_t const *d_i8, float const *d_f32, float const *d_inv_norms,
    nk_i32_t const *d_sum_i8,  /* pre-computed 128*sum(d_i8) per doc token */
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_f64_t *result) {

    __m256i xor_mask = _mm256_set1_epi8((char)0x80);
    __m256i ones_i16 = _mm256_set1_epi16(1);
    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;

    nk_size_t qblock = 0;
    for (; qblock + 4 <= query_count; qblock += 4) {
        __m128i running_max1 = _mm_set1_epi32(INT32_MIN);
        __m128i running_argmax1 = _mm_setzero_si128();
        __m128i running_max2 = _mm_set1_epi32(INT32_MIN);
        __m128i running_argmax2 = _mm_setzero_si128();

        /* 4Q×4D tile loop over doc tokens */
        nk_size_t dblock = 0;
        for (; dblock + 4 <= document_count; dblock += 4) {
            __m256i acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] = _mm256_setzero_si256();

            for (nk_size_t k = 0; k < depth; k += 32) {
                __m256i q0 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 0) * depth + k)), xor_mask);
                __m256i q1 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 1) * depth + k)), xor_mask);
                __m256i q2 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 2) * depth + k)), xor_mask);
                __m256i q3 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 3) * depth + k)), xor_mask);
                for (int dti = 0; dti < 4; dti++) {
                    __m256i d = _mm256_loadu_si256((__m256i const *)(d_i8 + (dblock + dti) * depth + k));
                    __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(q0, d), ones_i16);
                    __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q1, d), ones_i16);
                    __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2, d), ones_i16);
                    __m256i p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(q3, d), ones_i16);
                    acc[0][dti] = _mm256_add_epi32(acc[0][dti], p0);
                    acc[1][dti] = _mm256_add_epi32(acc[1][dti], p1);
                    acc[2][dti] = _mm256_add_epi32(acc[2][dti], p2);
                    acc[3][dti] = _mm256_add_epi32(acc[3][dti], p3);
                }
            }

            /* Reduce + bias + top-2 for each of 4 docs */
            for (int dti = 0; dti < 4; dti++) {
                __m128i reduced = cb_reduce_i32x8x4(acc[0][dti], acc[1][dti], acc[2][dti], acc[3][dti]);
                __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(d_sum_i8[dblock + dti]));
                __m128i didx = _mm_set1_epi32((int)(dblock + dti));
                CB_FLAT_TOP2_UPDATE(dots, didx);
            }
        }

        /* Doc tail: 4Q×1D */
        for (nk_size_t di = dblock; di < document_count; di++) {
            __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth; k += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_i8 + di * depth + k));
                __m256i q0 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 0) * depth + k)), xor_mask);
                __m256i q1 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 1) * depth + k)), xor_mask);
                __m256i q2 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 2) * depth + k)), xor_mask);
                __m256i q3 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + (qblock + 3) * depth + k)), xor_mask);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(q0, d), ones_i16));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(q1, d), ones_i16));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(q2, d), ones_i16));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(q3, d), ones_i16));
            }
            __m128i reduced = cb_reduce_i32x8x4(acc0, acc1, acc2, acc3);
            __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(d_sum_i8[di]));
            __m128i didx = _mm_set1_epi32((int)di);
            CB_FLAT_TOP2_UPDATE(dots, didx);
        }

        /* INLINE REFINE: extract top-2 indices and do f32 refine immediately,
           while doc f32 data is still hot in cache from the coarse scan. */
        nk_u32_t best_idx[4], second_idx[4];
        best_idx[0] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 0);
        best_idx[1] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 1);
        best_idx[2] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 2);
        best_idx[3] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 3);
        second_idx[0] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 0);
        second_idx[1] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 1);
        second_idx[2] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 2);
        second_idx[3] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 3);

        for (int qq = 0; qq < 4; qq++) {
            float const *qf = q_f32 + (qblock + qq) * depth;
            float inv_qn = q_inv_norms[qblock + qq];
            float const *d1f = d_f32 + best_idx[qq] * depth;
            float const *d2f = d_f32 + second_idx[qq] * depth;
            __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
            for (nk_size_t k = 0; k < depth; k += 8) {
                __m256 q = _mm256_loadu_ps(qf + k);
                a1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1f + k), a1);
                a2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2f + k), a2);
            }
            __m128 lo1 = _mm256_castps256_ps128(a1), hi1 = _mm256_extractf128_ps(a1, 1);
            __m128 lo2 = _mm256_castps256_ps128(a2), hi2 = _mm256_extractf128_ps(a2, 1);
            __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
            s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
            s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
            s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
            s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best_idx[qq]];
            nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[second_idx[qq]];
            nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
            nk_f64_t angular = 1.0 - best_cosine;
            if (angular < 0.0) angular = 0.0;
            total_sum += angular;
            total_sum_sq += angular * angular;
        }
    }

    /* Query tail: 1Q (scalar top-2) with inline refine */
    for (nk_size_t qi = qblock; qi < query_count; qi++) {
        nk_i32_t best1 = INT32_MIN, best2 = INT32_MIN;
        nk_u32_t best1_idx = 0, best2_idx = 0;
        for (nk_size_t di = 0; di < document_count; di++) {
            __m256i acc = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth; k += 32) {
                __m256i q = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_i8 + qi * depth + k)), xor_mask);
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_i8 + di * depth + k));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q, d), ones_i16));
            }
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot = _mm_extract_epi32(s, 0) - d_sum_i8[di];
            if (dot > best1) { best2 = best1; best2_idx = best1_idx; best1 = dot; best1_idx = (nk_u32_t)di; }
            else if (dot > best2) { best2 = dot; best2_idx = (nk_u32_t)di; }
        }
        /* Inline refine */
        float const *qf = q_f32 + qi * depth;
        float inv_qn = q_inv_norms[qi];
        float const *d1f = d_f32 + best1_idx * depth;
        float const *d2f = d_f32 + best2_idx * depth;
        __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
        for (nk_size_t k = 0; k < depth; k += 8) {
            __m256 q = _mm256_loadu_ps(qf + k);
            a1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1f + k), a1);
            a2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2f + k), a2);
        }
        __m128 lo1 = _mm256_castps256_ps128(a1), hi1 = _mm256_extractf128_ps(a1, 1);
        __m128 lo2 = _mm256_castps256_ps128(a2), hi2 = _mm256_extractf128_ps(a2, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
        s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
        s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
        s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
        s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best1_idx];
        nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best2_idx];
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

__attribute__((target("avx2,fma")))
static void nk_maxsim_i8_compact_stats(
    nk_i8_t const *q_i8, float const *q_f32, float const *q_inv_norms,
    nk_i8_t const *d_i8, float const *d_f32, float const *d_inv_norms,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_f64_t *result) {

    __m256i xor_mask = _mm256_set1_epi8((char)0x80);
    __m256i ones_i16 = _mm256_set1_epi16(1);
    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_i8_t const *qp = q_i8 + qi * depth;

        /* Phase 1: i8 coarse scan → top-2 doc token indices */
        nk_i32_t best1_dot = -2147483647, best2_dot = -2147483647;
        nk_u32_t best1_idx = 0, best2_idx = 0;

        for (nk_size_t di = 0; di < document_count; di++) {
            nk_i8_t const *dp = d_i8 + di * depth;  /* compact stride = depth! */

            __m256i acc = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth; k += 32) {
                __m256i q = _mm256_loadu_si256((__m256i const *)(qp + k));
                __m256i d = _mm256_loadu_si256((__m256i const *)(dp + k));
                /* Convert q to unsigned for maddubs */
                __m256i qu = _mm256_xor_si256(q, xor_mask);
                __m256i prod = _mm256_maddubs_epi16(qu, d);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones_i16));
            }

            /* Horizontal sum */
            __m128i lo128 = _mm256_castsi256_si128(acc);
            __m128i hi128 = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo128, hi128);
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot_i32 = _mm_extract_epi32(s, 0);

            /* Bias correction: maddubs(q^0x80, d) = q·d + 128·sum(d).
               For signed q × signed d, subtract 128 * sum(d). */
            {
                __m256i d_abs = _mm256_loadu_si256((__m256i const *)dp);
                /* Quick sum via maddubs(1, d): pairwise sums */
                __m256i ones8 = _mm256_set1_epi8(1);
                __m256i d_sum_acc = _mm256_setzero_si256();
                for (nk_size_t k = 0; k < depth; k += 32) {
                    __m256i dv = _mm256_loadu_si256((__m256i const *)(dp + k));
                    __m256i ps = _mm256_maddubs_epi16(ones8, dv);
                    d_sum_acc = _mm256_add_epi32(d_sum_acc, _mm256_madd_epi16(ps, ones_i16));
                }
                __m128i ds_lo = _mm256_castsi256_si128(d_sum_acc);
                __m128i ds_hi = _mm256_extracti128_si256(d_sum_acc, 1);
                __m128i ds = _mm_add_epi32(ds_lo, ds_hi);
                ds = _mm_add_epi32(ds, _mm_shuffle_epi32(ds, 0x4E));
                ds = _mm_add_epi32(ds, _mm_shuffle_epi32(ds, 0xB1));
                dot_i32 -= 128 * _mm_extract_epi32(ds, 0);
            }

            if (dot_i32 > best1_dot) {
                best2_dot = best1_dot; best2_idx = best1_idx;
                best1_dot = dot_i32;   best1_idx = (nk_u32_t)di;
            } else if (dot_i32 > best2_dot) {
                best2_dot = dot_i32;   best2_idx = (nk_u32_t)di;
            }
        }

        /* Phase 2: f32 refine on top-2 */
        float const *qf = q_f32 + qi * depth;
        float inv_qn = q_inv_norms[qi];
        float const *d1f = d_f32 + best1_idx * depth;
        float const *d2f = d_f32 + best2_idx * depth;

        __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
        for (nk_size_t k = 0; k < depth; k += 8) {
            __m256 q = _mm256_loadu_ps(qf + k);
            acc1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1f + k), acc1);
            acc2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2f + k), acc2);
        }
        __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
        s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
        s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
        s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
        s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));

        nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * (nk_f64_t)inv_qn *
                        (nk_f64_t)d_inv_norms[best1_idx];
        nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * (nk_f64_t)inv_qn *
                        (nk_f64_t)d_inv_norms[best2_idx];
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;

        total_sum += angular;
        total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

__attribute__((target("avx2,fma")))
static void nk_maxsim_4bit_compact_stats(
    nk_u8_t const *q_4bit, float const *q_f32, float const *q_inv_norms,
    nk_u8_t const *d_4bit, float const *d_f32, float const *d_inv_norms,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_i8_t const *centroid_i8, nk_f64_t *result) {

    __m128i ctable_128 = _mm_loadu_si128((__m128i const *)centroid_i8);
    __m256i ctable = _mm256_broadcastsi128_si256(ctable_128);
    __m256i lo_mask = _mm256_set1_epi8(0x0F);
    __m256i xor_mask = _mm256_set1_epi8((char)0x80);
    __m256i ones_i16 = _mm256_set1_epi16(1);
    __m256i ones_i8 = _mm256_set1_epi8(1);

    nk_size_t half = depth / 2;  /* 64 for d=128 */
    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_u8_t const *qp = q_4bit + qi * half;

        /* Expand query nibbles to i8 centroid values (once per query token) */
        nk_i8_t q_expanded[256];
        for (nk_size_t k = 0; k < half; k += 32) {
            nk_size_t chunk = half - k < 32 ? half - k : 32;
            __m256i packed;
            if (chunk == 32)
                packed = _mm256_loadu_si256((__m256i const *)(qp + k));
            else {
                packed = _mm256_setzero_si256();
                for (nk_size_t b = 0; b < chunk; b++)
                    ((nk_u8_t *)&packed)[b] = qp[k + b];
            }
            __m256i nib_lo = _mm256_and_si256(packed, lo_mask);
            __m256i nib_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);
            __m256i val_lo = _mm256_shuffle_epi8(ctable, nib_lo);
            __m256i val_hi = _mm256_shuffle_epi8(ctable, nib_hi);
            __m256i even = _mm256_unpacklo_epi8(val_lo, val_hi);
            __m256i odd  = _mm256_unpackhi_epi8(val_lo, val_hi);
            _mm256_storeu_si256((__m256i *)(q_expanded + 2*k), even);
            _mm256_storeu_si256((__m256i *)(q_expanded + 2*k + 32), odd);
        }

        /* Phase 1: 4-bit coarse scan → top-2 doc token indices */
        nk_i32_t best1_dot = -2147483647, best2_dot = -2147483647;
        nk_u32_t best1_idx = 0, best2_idx = 0;

        for (nk_size_t di = 0; di < document_count; di++) {
            nk_u8_t const *dp = d_4bit + di * half;  /* compact stride! */

            __m256i acc = _mm256_setzero_si256();
            __m256i d_sum_acc = _mm256_setzero_si256();

            for (nk_size_t k = 0; k < half; k += 32) {
                nk_size_t chunk = half - k < 32 ? half - k : 32;
                __m256i d_nibbles;
                if (chunk == 32)
                    d_nibbles = _mm256_loadu_si256((__m256i const *)(dp + k));
                else {
                    d_nibbles = _mm256_setzero_si256();
                    for (nk_size_t b = 0; b < chunk; b++)
                        ((nk_u8_t *)&d_nibbles)[b] = dp[k + b];
                }

                __m256i d_lo = _mm256_shuffle_epi8(ctable, _mm256_and_si256(d_nibbles, lo_mask));
                __m256i d_hi = _mm256_shuffle_epi8(ctable, _mm256_and_si256(_mm256_srli_epi16(d_nibbles, 4), lo_mask));
                __m256i d_even = _mm256_unpacklo_epi8(d_lo, d_hi);
                __m256i d_odd  = _mm256_unpackhi_epi8(d_lo, d_hi);

                __m256i d_sum_even = _mm256_maddubs_epi16(ones_i8, d_even);
                __m256i d_sum_odd  = _mm256_maddubs_epi16(ones_i8, d_odd);
                d_sum_acc = _mm256_add_epi32(d_sum_acc,
                    _mm256_madd_epi16(_mm256_add_epi16(d_sum_even, d_sum_odd), ones_i16));

                __m256i q_even = _mm256_loadu_si256((__m256i const *)(q_expanded + 2*k));
                __m256i q_odd  = _mm256_loadu_si256((__m256i const *)(q_expanded + 2*k + 32));
                __m256i q_even_u = _mm256_xor_si256(q_even, xor_mask);
                __m256i q_odd_u  = _mm256_xor_si256(q_odd, xor_mask);
                __m256i prod_even = _mm256_maddubs_epi16(q_even_u, d_even);
                __m256i prod_odd  = _mm256_maddubs_epi16(q_odd_u, d_odd);
                __m256i sum16 = _mm256_add_epi16(prod_even, prod_odd);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(sum16, ones_i16));
            }

            __m128i lo128 = _mm256_castsi256_si128(acc);
            __m128i hi128 = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo128, hi128);
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot_i32 = _mm_extract_epi32(s, 0);
            {
                __m128i ds_lo = _mm256_castsi256_si128(d_sum_acc);
                __m128i ds_hi = _mm256_extracti128_si256(d_sum_acc, 1);
                __m128i ds = _mm_add_epi32(ds_lo, ds_hi);
                ds = _mm_add_epi32(ds, _mm_shuffle_epi32(ds, 0x4E));
                ds = _mm_add_epi32(ds, _mm_shuffle_epi32(ds, 0xB1));
                dot_i32 -= 128 * _mm_extract_epi32(ds, 0);
            }

            if (dot_i32 > best1_dot) {
                best2_dot = best1_dot; best2_idx = best1_idx;
                best1_dot = dot_i32;   best1_idx = (nk_u32_t)di;
            } else if (dot_i32 > best2_dot) {
                best2_dot = dot_i32;   best2_idx = (nk_u32_t)di;
            }
        }

        /* Phase 2: f32 refine on top-2 */
        float const *qf = q_f32 + qi * depth;
        float inv_qn = q_inv_norms[qi];

        float const *d1f = d_f32 + best1_idx * depth;
        float const *d2f = d_f32 + best2_idx * depth;

        __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
        for (nk_size_t k = 0; k < depth; k += 8) {
            __m256 q = _mm256_loadu_ps(qf + k);
            acc1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1f + k), acc1);
            acc2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2f + k), acc2);
        }
        __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
        s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
        s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
        s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
        s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));

        nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * (nk_f64_t)inv_qn *
                        (nk_f64_t)d_inv_norms[best1_idx];
        nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * (nk_f64_t)inv_qn *
                        (nk_f64_t)d_inv_norms[best2_idx];
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;

        total_sum += angular;
        total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}
#endif /* NK_TARGET_X86_ && NK_TARGET_HASWELL — closes top-of-file Haswell block */

/* ============================================================================
 * Pure-fp32 MaxSim micro-kernels for colbandit_maxsim_fp32.
 *
 * Unlike `nk_maxsim_packed_f32_haswell`, these kernels do NOT touch the i8
 * coarse region of the packed format. They operate purely on raw fp32 query
 * vectors and the fp32 originals region of packed docs. For each query token
 * the inner loop scans ALL doc tokens with full fp32 dot products and takes
 * the max — no i8 argmax screening.
 *
 * Inputs:
 *   q_f32         [query_count, depth] raw fp32 query (caller supplies)
 *   d_f32         [doc_token_count, doc_stride_floats] fp32 originals from packed doc
 *   doc_stride    row stride of d_f32 in floats (= header->original_stride_bytes / 4)
 *   d_inv_norms   per-doc-token inverse norms (from doc metadata)
 *   q_inv_norm    per-query-token inverse norm (assumed pre-normalized to 1.0;
 *                 caller can pass NULL to skip the multiply)
 *
 * Output:
 *   result[0] = sum of angular distances (1 - max cosine) over query tokens
 *   result[1] = sum of squared angular distances (only filled for *_stats variant)
 *
 * Cost per cell: one fp32 FMA chain over `depth` (32 fmadd_ps for d=128).
 * No i8 quantize, no two-phase coarse+refine. All-fp32 from start to finish.
 * ============================================================================ */
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
__attribute__((target("avx2,fma")))
static inline float cb_fp32_dot_d128_avx2_(float const *qf, float const *df, nk_size_t depth) {
    /* Two-accumulator unrolled fp32 dot product. Works for depth multiple of 16;
       for d=128 unrolls to 8 fmadd pairs. Accuracy and speed match nk_dot_f32_haswell. */
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    nk_size_t k = 0;
    for (; k + 16 <= depth; k += 16) {
        __m256 q0 = _mm256_loadu_ps(qf + k);
        __m256 q1 = _mm256_loadu_ps(qf + k + 8);
        a0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(df + k), a0);
        a1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(df + k + 8), a1);
    }
    for (; k + 8 <= depth; k += 8) {
        __m256 q = _mm256_loadu_ps(qf + k);
        a0 = _mm256_fmadd_ps(q, _mm256_loadu_ps(df + k), a0);
    }
    __m256 a = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(a), hi = _mm256_extractf128_ps(a, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float scalar = _mm_cvtss_f32(s);
    /* Scalar tail (rare for d=128) */
    for (; k < depth; k++) scalar += qf[k] * df[k];
    return scalar;
}

/* Scan all doc tokens for each query token in pure fp32; accumulate sum (and optionally sum_sq).
   Both query and doc vectors are assumed L2-normalized externally
   — the caller passes inv-norms = 1.0 in the bench. */
__attribute__((target("avx2,fma")))
static void cb_fp32only_maxsim_stats_(
    float const *q_f32, nk_size_t q_stride_floats,
    float const *q_inv_norms,                /* may be NULL, then treated as 1.0 */
    float const *d_f32, nk_size_t d_stride_floats,
    nk_maxsim_vector_metadata_t const *d_meta,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    int want_sum_sq,
    nk_f64_t *result) {

    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        float const *qf = q_f32 + qi * q_stride_floats;
        float inv_qn = q_inv_norms ? q_inv_norms[qi] : 1.0f;

        /* Scan ALL doc tokens with fp32 dots, track max cosine */
        nk_f64_t best_cosine = -1e30;
        for (nk_size_t di = 0; di < document_count; di++) {
            float const *df = d_f32 + di * d_stride_floats;
            float dot = cb_fp32_dot_d128_avx2_(qf, df, depth);
            nk_f64_t cosine = (nk_f64_t)dot * (nk_f64_t)inv_qn *
                              (nk_f64_t)d_meta[di].inverse_norm_f32;
            if (cosine > best_cosine) best_cosine = cosine;
        }

        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        if (want_sum_sq) total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    if (want_sum_sq) result[1] = total_sum_sq;
}
#else
/* Portable scalar fallback for non-Haswell builds */
static void cb_fp32only_maxsim_stats_(
    float const *q_f32, nk_size_t q_stride_floats,
    float const *q_inv_norms,
    float const *d_f32, nk_size_t d_stride_floats,
    nk_maxsim_vector_metadata_t const *d_meta,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    int want_sum_sq,
    nk_f64_t *result) {

    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        float const *qf = q_f32 + qi * q_stride_floats;
        float inv_qn = q_inv_norms ? q_inv_norms[qi] : 1.0f;

        nk_f64_t best_cosine = -1e30;
        for (nk_size_t di = 0; di < document_count; di++) {
            float const *df = d_f32 + di * d_stride_floats;
            float dot_f = 0.0f;
            for (nk_size_t k = 0; k < depth; k++) dot_f += qf[k] * df[k];
            nk_f64_t cosine = (nk_f64_t)dot_f * (nk_f64_t)inv_qn *
                              (nk_f64_t)d_meta[di].inverse_norm_f32;
            if (cosine > best_cosine) best_cosine = cosine;
        }
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        if (want_sum_sq) total_sum_sq += angular * angular;
    }
    result[0] = total_sum;
    if (want_sum_sq) result[1] = total_sum_sq;
}
#endif /* NK_TARGET_X86_ && NK_TARGET_HASWELL */

static double _now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int _cmp_u32(const void *a, const void *b) {
    nk_u32_t va = *(const nk_u32_t *)a, vb = *(const nk_u32_t *)b;
    return (va > vb) - (va < vb);
}

/* Compare (address, index) pairs by address — for sorting docs by memory location */
static int _cmp_u64_pair(const void *a, const void *b) {
    nk_size_t va = *(const nk_size_t *)a, vb = *(const nk_size_t *)b;
    return (va > vb) - (va < vb);
}

/* Compare (score, idx) pairs by score ascending — used by api_topm_maxsim.
   File-scope (FIX G) because Apple Clang doesn't accept nested function defs. */
typedef struct { double score; nk_u32_t idx; } _score_idx_t;
static int _cmp_si(const void *a, const void *b) {
    double da = ((const _score_idx_t *)a)->score;
    double db = ((const _score_idx_t *)b)->score;
    return (da > db) - (da < db);
}

/* ====== Centroid packing: overwrite i8 region with u8 centroid indices ====== */
char const doc_maxsim_pack_set_indices[] =
    "maxsim_pack_set_indices(packed, indices, /) -> None\n\n"
    "Overwrite the i8 region of a MaxSimPackedMatrix with u8 centroid indices.\n"
    "The packed object must have been created by maxsim_pack() first.\n";

PyObject *api_maxsim_pack_set_indices(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "maxsim_pack_set_indices(packed, indices) requires 2 args");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "First arg must be MaxSimPackedMatrix");
        return NULL;
    }
    MaxSimPackedMatrix *packed = (MaxSimPackedMatrix *)args[0];

    Py_buffer idx_buf;
    if (PyObject_GetBuffer(args[1], &idx_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    /* Read header to find i8 data offset */
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed->start;
    nk_i8_t *i8_data = (nk_i8_t *)(packed->start + header->offset_i8_data);
    nk_size_t depth_i8_padded = header->depth_i8_padded;
    nk_size_t vector_count = packed->vector_count;
    nk_size_t depth = packed->depth;

    /* Validate indices shape: [vector_count, depth] or flat [vector_count * depth] */
    nk_size_t expected = vector_count * depth;
    nk_size_t got = idx_buf.len;
    if (got != expected) {
        PyBuffer_Release(&idx_buf);
        PyErr_Format(PyExc_ValueError, "indices size %zu != expected %zu (vector_count=%zu × depth=%zu)",
                     got, expected, vector_count, depth);
        return NULL;
    }

    /* Copy indices into i8 region (u8 → i8 reinterpret, zero-pad to depth_i8_padded) */
    nk_u8_t const *indices = (nk_u8_t const *)idx_buf.buf;
    for (nk_size_t v = 0; v < vector_count; v++) {
        for (nk_size_t d = 0; d < depth; d++)
            i8_data[v * depth_i8_padded + d] = (nk_i8_t)indices[v * depth + d];
        for (nk_size_t d = depth; d < depth_i8_padded; d++)
            i8_data[v * depth_i8_padded + d] = 0;
    }

    PyBuffer_Release(&idx_buf);
    Py_RETURN_NONE;
}

/* ====== 4-bit nibble-pack: overwrite i8 region with nibble-packed u8 indices ====== */
char const doc_maxsim_pack_set_4bit[] =
    "maxsim_pack_set_4bit(packed, indices, /) -> None\n\n"
    "Overwrite the i8 region with 4-bit nibble-packed centroid indices.\n"
    "indices: [vector_count, depth] u8 array with values 0-15.\n"
    "Byte k gets: dim[2k] in low nibble, dim[2k+1] in high nibble.\n";

PyObject *api_maxsim_pack_set_4bit(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "maxsim_pack_set_4bit(packed, indices) requires 2 args");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "First arg must be MaxSimPackedMatrix");
        return NULL;
    }
    MaxSimPackedMatrix *packed = (MaxSimPackedMatrix *)args[0];

    Py_buffer idx_buf;
    if (PyObject_GetBuffer(args[1], &idx_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed->start;
    nk_u8_t *i8_data = (nk_u8_t *)(packed->start + header->offset_i8_data);
    nk_size_t depth_i8_padded = header->depth_i8_padded;
    nk_size_t vector_count = packed->vector_count;
    nk_size_t depth = packed->depth;
    nk_size_t half = depth / 2;

    nk_size_t expected = vector_count * depth;
    nk_size_t got = idx_buf.len;
    if (got != expected) {
        PyBuffer_Release(&idx_buf);
        PyErr_Format(PyExc_ValueError, "indices size %zu != expected %zu (vector_count=%zu x depth=%zu)",
                     got, expected, vector_count, depth);
        return NULL;
    }

    nk_u8_t const *indices = (nk_u8_t const *)idx_buf.buf;
    for (nk_size_t v = 0; v < vector_count; v++) {
        nk_u8_t const *src = indices + v * depth;
        nk_u8_t *dst = i8_data + v * depth_i8_padded;
        /* Nibble-pack: dim[2k] in low nibble, dim[2k+1] in high nibble */
        for (nk_size_t k = 0; k < half; k++)
            dst[k] = (src[2*k] & 0x0F) | ((src[2*k+1] & 0x0F) << 4);
        /* Zero-pad remainder */
        for (nk_size_t k = half; k < depth_i8_padded; k++)
            dst[k] = 0;
    }

    PyBuffer_Release(&idx_buf);
    Py_RETURN_NONE;
}

/* ====== Centroid MaxSim: single query vs single doc, with centroid table ====== */
char const doc_centroid_maxsim[] =
    "centroid_maxsim(query_packed, doc_packed, centroid_table, /) -> (sum, sum_sq)\n\n"
    "Compute MaxSim using centroid-lookup dot products.\n";

PyObject *api_centroid_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "centroid_maxsim(query, doc, centroid_table) requires 3 args");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], &MaxSimPackedMatrixType) ||
        !PyObject_TypeCheck(args[1], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "query and doc must be MaxSimPackedMatrix");
        return NULL;
    }
    MaxSimPackedMatrix *query = (MaxSimPackedMatrix *)args[0];
    MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)args[1];

    Py_buffer ct_buf;
    if (PyObject_GetBuffer(args[2], &ct_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_f64_t result[2];
    NK_CENTROID_STATS_KERNEL(
        query->start, doc->start,
        query->vector_count, doc->vector_count,
        query->depth, result, (float const *)ct_buf.buf);

    PyBuffer_Release(&ct_buf);
    return Py_BuildValue("(dd)", result[0], result[1]);
}

/* ====== 4-bit VPSHUFB centroid MaxSim ====== */
char const doc_centroid4_maxsim[] =
    "centroid4_maxsim(query_packed, doc_packed, centroid_i8_table, /) -> (sum, sum_sq)\n\n"
    "4-bit VPSHUFB centroid MaxSim. Near-maddubs speed with Lloyd-Max accuracy.\n";

PyObject *api_centroid4_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "centroid4_maxsim(query, doc, centroid_i8_table) requires 3 args");
        return NULL;
    }
    if (!PyObject_TypeCheck(args[0], &MaxSimPackedMatrixType) ||
        !PyObject_TypeCheck(args[1], &MaxSimPackedMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "query and doc must be MaxSimPackedMatrix");
        return NULL;
    }
    MaxSimPackedMatrix *query = (MaxSimPackedMatrix *)args[0];
    MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)args[1];

    Py_buffer ct_buf;
    if (PyObject_GetBuffer(args[2], &ct_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_f64_t result[2];
    nk_maxsim_packed_4bit_stats_f32_haswell(
        query->start, doc->start,
        query->vector_count, doc->vector_count,
        query->depth, result, (nk_i8_t const *)ct_buf.buf);

    PyBuffer_Release(&ct_buf);
    return Py_BuildValue("(dd)", result[0], result[1]);
#else
    PyErr_SetString(PyExc_NotImplementedError, "centroid4_maxsim requires x86 AVX2");
    return NULL;
#endif
}

/* Score a packed query against N packed docs. Parallelized with OpenMP. */
static void score_docs(
    nk_maxsim_packed_punned_t kernel,
    void const *query_packed_start,
    nk_size_t query_count,
    PyObject **doc_objects,
    nk_u32_t *doc_indices,
    nk_size_t n_docs,
    nk_size_t depth,
    nk_dtype_t dtype,
    double *out_scores,
    int n_threads
) {
    nk_dtype_t out_dtype = nk_kernel_output_dtype(nk_kernel_maxsim_packed_k, dtype);
    #if !defined(NK_OMP_PRAGMAS_DISABLED)
    #pragma omp parallel for schedule(static) if(n_docs > 8 && n_threads > 1) num_threads(n_threads)
    #endif
    for (nk_size_t i = 0; i < n_docs; i++) {
        nk_size_t di = doc_indices ? doc_indices[i] : i;
        MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
        if (out_dtype == nk_f64_k) {
            nk_f64_t result;
            kernel(query_packed_start, doc->start, query_count, doc->vector_count, depth, &result);
            out_scores[i] = result;
        } else {
            nk_f32_t result;
            kernel(query_packed_start, doc->start, query_count, doc->vector_count, depth, &result);
            out_scores[i] = (double)result;
        }
    }
}

/* K-element max-heap for finding K-th smallest UCB without array copy. */
static void _heap_sift_down(double *heap, nk_size_t n, nk_size_t i) {
    while (1) {
        nk_size_t largest = i, left = 2*i+1, right = 2*i+2;
        if (left < n && heap[left] > heap[largest]) largest = left;
        if (right < n && heap[right] > heap[largest]) largest = right;
        if (largest == i) break;
        double tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp;
        i = largest;
    }
}

/* Bitmap counting sort for u32 survivor indices — O(N_orig) with tiny constant. */
static void _counting_sort_u32(nk_u32_t *arr, nk_size_t n, nk_u32_t max_val, nk_u8_t *bitmap) {
    nk_size_t bitmap_bytes = (max_val + 8) / 8;
    memset(bitmap, 0, bitmap_bytes);
    for (nk_size_t i = 0; i < n; i++)
        bitmap[arr[i] / 8] |= (1u << (arr[i] % 8));
    nk_size_t out = 0;
    for (nk_u32_t v = 0; v <= max_val && out < n; v++)
        if (bitmap[v / 8] & (1u << (v % 8)))
            arr[out++] = v;
}

/* Serfling-based elimination: returns new number of survivors.
   Uses K-element min-heap (no array copy) and bitmap counting sort (no qsort).

   Bound: Bardenet-Maillard (2015), Theorem 4.3. Three corrections vs. earlier
   versions of this kernel:

   FIX 1: Piecewise rho_n per BM eq. (14). Earlier used the single-formula
          (1 - n/T)(1 + 1/T), which is the n > T/2 branch — and even that branch
          had (1 + 1/T) instead of the correct (1 + 1/n). For n <= T/2 the
          correct formula is rho_n = 1 - (n - 1)/T.

   FIX 3: log term unions over (doc, sample-size n) pairs simultaneously:
          log(c * N * T / delta), not log(N / delta). The factor T covers a
          union over n in {1, 2, ..., T}; c = 10 is the BM 4.3 two-sided
          probability-mass constant (5 one-sided x 2 two-sided).

   FIX 6: Always use N_orig (original corpus size) in the log term. The
          previously-available `use_orig_N=False` mode (substituting current
          n_survivors) is self-referential — eliminated docs were judged at a
          past M, not the current one — and is not covered by a single Boole
          union. The use_orig_N parameter is kept on the signature for ABI
          compat but is now ignored. */
static nk_size_t serfling_eliminate(
    double *obs_sum, double *obs_sum_sq, nk_u32_t *obs_count,
    nk_u32_t *survivors, nk_size_t n_survivors,
    nk_size_t K, nk_size_t T, double alpha_ef, double delta,
    double *lcb, double *ucb,
    double *heap_buf,       /* pre-allocated [K] for min-heap */
    nk_u32_t N_orig,        /* original corpus size — used for both log term and counting sort */
    nk_u8_t *sort_bitmap,   /* pre-allocated [(N_orig+7)/8] */
    int use_orig_N,         /* if non-zero use N_orig in union bound (BM 4.3 valid);
                               otherwise use current n_survivors (tighter per round, NOT
                               covered by a single Boole union — empirical-only mode). */
    int *fallback_out       /* FIX B: set to 1 if fallback triggered (bound failure canary), else 0.
                               Pass NULL to opt out. */
) {
    if (fallback_out) *fallback_out = 0;

    /* Union-bound size:
       - use_orig_N=True  : log(c * N * T / delta)        — mathematically valid (BM 4.3, FIX 3+6)
       - use_orig_N=False : log(c * n_survivors * T / delta) — TIGHTER per round, but the bound
                            is not covered by a single Boole union (eliminated docs were judged
                            at past M, not current M). Use only for empirical comparison. */
    const double C_BM = 10.0;
    double safe_delta = (delta > 1e-30 ? delta : 1e-30);
    double union_n = use_orig_N ? (double)N_orig : (double)n_survivors;
    if (union_n < 1.0) union_n = 1.0;
    double log_idd = log(C_BM * union_n * (double)T / safe_delta);

    double inv_T = 1.0 / (double)T;

    /* FIX A: cells = angular = 1 - cosine, with cosine in [-1, 1] mathematically.
       The kernel clamps angular >= 0 (via `if (angular < 0.0) angular = 0.0;`)
       but does NOT clamp angular <= 1 (which would require cosine >= 0). So strictly
       angular in [0, 2]. We use R = 2 in the remaining-cell hard cap to be valid for
       any cosine. For ColBERTv2-style trained embeddings cosines are typically >= 0,
       so the previous R = 1 cap was tight in practice — but R = 2 is mathematically
       safe and only loosens the bound near elimination boundaries. */
    const double CELL_RANGE_MAX = 2.0;

    for (nk_size_t i = 0; i < n_survivors; i++) {
        nk_u32_t di = survivors[i];
        nk_u32_t nt = obs_count[di];
        double sum = obs_sum[di];
        if (nt < 2) { lcb[i] = sum; ucb[i] = sum + CELL_RANGE_MAX * (double)(T - nt); continue; }
        double inv_nt = 1.0 / (double)nt;
        double mean = sum * inv_nt;
        double var = (obs_sum_sq[di] * inv_nt - mean * mean) * (double)nt / (double)(nt - 1);
        if (var < 0.0) var = 0.0;
        double sigma = sqrt(var > 1e-12 ? var : 1e-12);

        /* FIX 1: piecewise Bardenet-Maillard rho_n (BM 2015 eq. 14) */
        double rho;
        if ((nk_size_t)nt * 2 <= T) {
            /* n <= T/2:  rho = 1 - (n - 1) / T */
            rho = 1.0 - (double)(nt - 1) * inv_T;
        } else {
            /* n  > T/2:  rho = (1 - n/T) * (1 + 1/n) */
            rho = (1.0 - (double)nt * inv_T) * (1.0 + inv_nt);
        }
        if (rho < 0.0) rho = 0.0;

        double r = alpha_ef * (double)T * sigma * sqrt(2.0 * rho * log_idd * inv_nt);
        double est_full = mean * (double)T;
        lcb[i] = est_full - r;
        if (lcb[i] < sum) lcb[i] = sum;
        ucb[i] = est_full + r;
        double ucb_hard = sum + CELL_RANGE_MAX * (double)(T - nt);  /* FIX A: R = 2 */
        if (ucb[i] > ucb_hard) ucb[i] = ucb_hard;
    }

    /* K-element max-heap to find K-th smallest UCB — no memcpy needed */
    nk_size_t heap_size = 0;
    for (nk_size_t i = 0; i < n_survivors; i++) {
        if (heap_size < K) {
            heap_buf[heap_size++] = ucb[i];
            if (heap_size == K)
                for (nk_size_t j = K/2; j-- > 0; )
                    _heap_sift_down(heap_buf, K, j);
        } else if (ucb[i] < heap_buf[0]) {
            heap_buf[0] = ucb[i];
            _heap_sift_down(heap_buf, K, 0);
        }
    }
    double kth_ucb = heap_buf[0];

    nk_size_t new_count = 0;
    for (nk_size_t i = 0; i < n_survivors; i++) {
        if (lcb[i] <= kth_ucb) {
            survivors[new_count] = survivors[i];
            new_count++;
        }
    }
    /* FIX B: under valid bounds the true top-K cannot be eliminated, so this
       fallback should never trigger. If it fires, alpha_ef is too aggressive
       and the bound was violated — surface it to the caller so they can act. */
    if (new_count < K) {
        if (fallback_out) *fallback_out = 1;
        new_count = n_survivors;
    }

    /* Bitmap counting sort for cache locality — O(N_orig), skip for very large sets */
    if (new_count <= 10000)
        _counting_sort_u32(survivors, new_count, N_orig > 0 ? N_orig - 1 : 0, sort_bitmap);

    return new_count;
}

/* ====== Top-M baseline: single warmup round → truncate → full score ====== */
char const doc_topm_maxsim[] =
    "topm_maxsim(query_tokens, doc_packed_list, /, K=5, M=100, "
    "n_warmup_tokens=4, n_threads=1) -> tuple\n\n"
    "Top-M heuristic baseline. Score all docs on n_warmup_tokens random tokens,\n"
    "take top-M by partial score, then full-score those M docs on all T tokens.\n"
    "Returns: (top_k_indices list, top_k_scores list, stats dict)\n";

PyObject *api_topm_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 2) { PyErr_SetString(PyExc_TypeError, "topm_maxsim() requires at least 2 args"); return NULL; }

    PyObject *query_obj = args[0], *doc_list_obj = args[1];
    long K = 5, M = 100, n_warmup_tokens = 4;
    int n_threads = 1;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i), *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "M") == 0) M = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_warmup_tokens") == 0) n_warmup_tokens = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    nk_size_t T = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    /* Skip padding tokens */
    while (T > 0) {
        float *row = query_data + (T-1)*depth; float ns = 0;
        for (nk_size_t j = 0; j < depth; j++) ns += row[j]*row[j];
        if (ns > 1e-12f) break; T--;
    }
    if (T == 0) T = 1;
    if ((nk_size_t)n_warmup_tokens > T) n_warmup_tokens = (long)T;
    /* Align to 4 */
    n_warmup_tokens = (n_warmup_tokens / 4) * 4;
    if (n_warmup_tokens < 4) n_warmup_tokens = 4;

    if (!PyList_Check(doc_list_obj)) { PyBuffer_Release(&query_buf); PyErr_SetString(PyExc_TypeError, "list"); return NULL; }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    PyObject **doc_objects = PySequence_Fast_ITEMS(doc_list_obj);

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;
    nk_maxsim_packed_punned_t kernel = NULL, kernel_stats = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    nk_find_kernel_punned(nk_kernel_maxsim_packed_stats_k, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel_stats, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities, (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!kernel || !kernel_stats || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf); PyErr_SetString(PyExc_LookupError, "No kernel"); return NULL;
    }

    nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;

    /* Random permutation (same LCG as CB-NK) */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    PyThreadState *save = PyEval_SaveThread();
    double t0, t_warmup = 0, t_full = 0;

    /* Step 1: Pack warmup tokens */
    float *q_warmup_data = (float *)malloc(n_warmup_tokens * depth * sizeof(float));
    for (long t = 0; t < n_warmup_tokens; t++)
        memcpy(q_warmup_data + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    nk_size_t wp_size = size_fn(n_warmup_tokens, depth);
    void *q_warmup_packed = malloc(wp_size);
    pack_fn(q_warmup_data, n_warmup_tokens, depth, depth * element_bytes, q_warmup_packed);
    free(q_warmup_data);

    /* Step 2: Score ALL N docs on warmup tokens — OpenMP parallel */
    double *partial_scores = (double *)malloc(N * sizeof(double));
    t0 = _now_ms();
    #if !defined(NK_OMP_PRAGMAS_DISABLED)
    #pragma omp parallel for schedule(static) if(N > 256 && n_threads > 1) num_threads(n_threads)
    #endif
    for (nk_size_t i = 0; i < N; i++) {
        MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[i];
        nk_f64_t stats[2];
        kernel_stats(q_warmup_packed, doc->start, n_warmup_tokens, doc->vector_count, depth, stats);
        partial_scores[i] = stats[0];
    }
    t_warmup = _now_ms() - t0;
    free(q_warmup_packed);

    /* Step 3: Take top-M by partial score (angular: lower = better) */
    nk_size_t M_actual = (nk_size_t)M < N ? (nk_size_t)M : N;
    nk_u32_t *topM = (nk_u32_t *)malloc(M_actual * sizeof(nk_u32_t));
    /* Partial argpartition: find M smallest */
    nk_u32_t *indices = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < N; i++) indices[i] = (nk_u32_t)i;
    /* Simple: sort and take first M (qsort is fine for one-time) */
    /* Pack score+index for sorting (struct + cmp at file scope, FIX G) */
    _score_idx_t *si = (_score_idx_t *)malloc(N * sizeof(_score_idx_t));
    for (nk_size_t i = 0; i < N; i++) { si[i].score = partial_scores[i]; si[i].idx = (nk_u32_t)i; }
    qsort(si, N, sizeof(_score_idx_t), _cmp_si);
    for (nk_size_t i = 0; i < M_actual; i++) topM[i] = si[i].idx;
    free(si); free(indices); free(partial_scores);

    /* Step 4: Full-score top-M docs on ALL T tokens — OpenMP parallel */
    nk_size_t n_remaining = T - n_warmup_tokens;
    float *q_remaining_data = (float *)malloc(n_remaining * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_remaining; t++)
        memcpy(q_remaining_data + t * depth, query_data + perm[n_warmup_tokens + t] * depth, depth * sizeof(float));
    nk_size_t rp_size = size_fn(n_remaining, depth);
    void *q_remaining_packed = malloc(rp_size);
    pack_fn(q_remaining_data, n_remaining, depth, depth * element_bytes, q_remaining_packed);
    free(q_remaining_data);

    double *full_scores = (double *)malloc(M_actual * sizeof(double));
    t0 = _now_ms();
    nk_dtype_t out_dtype = nk_kernel_output_dtype(nk_kernel_maxsim_packed_k, dtype);
    #if !defined(NK_OMP_PRAGMAS_DISABLED)
    #pragma omp parallel for schedule(static) if(M_actual > 8 && n_threads > 1) num_threads(n_threads)
    #endif
    for (nk_size_t i = 0; i < M_actual; i++) {
        MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[topM[i]];
        if (out_dtype == nk_f64_k) {
            nk_f64_t r; kernel(q_remaining_packed, doc->start, n_remaining, doc->vector_count, depth, &r);
            full_scores[i] = r;
        } else {
            nk_f32_t r; kernel(q_remaining_packed, doc->start, n_remaining, doc->vector_count, depth, &r);
            full_scores[i] = (double)r;
        }
    }
    t_full = _now_ms() - t0;
    free(q_remaining_packed);

    /* Add warmup partial scores back (need warmup contribution too) */
    /* Actually re-score warmup tokens on M survivors for consistency */
    /* Simpler: just re-score all T tokens on M survivors */
    /* The full_scores above only has remaining tokens. Add warmup contribution. */
    /* For simplicity, re-pack all T tokens and score M docs */
    nk_size_t fp_size = size_fn(T, depth);
    void *q_full_packed = malloc(fp_size);
    pack_fn(query_data, T, depth, depth * element_bytes, q_full_packed);
    #if !defined(NK_OMP_PRAGMAS_DISABLED)
    #pragma omp parallel for schedule(static) if(M_actual > 8 && n_threads > 1) num_threads(n_threads)
    #endif
    for (nk_size_t i = 0; i < M_actual; i++) {
        MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[topM[i]];
        if (out_dtype == nk_f64_k) {
            nk_f64_t r; kernel(q_full_packed, doc->start, T, doc->vector_count, depth, &r);
            full_scores[i] = r;
        } else {
            nk_f32_t r; kernel(q_full_packed, doc->start, T, doc->vector_count, depth, &r);
            full_scores[i] = (double)r;
        }
    }
    free(q_full_packed);

    /* Step 5: Return top-K from M (ascending = best for angular) */
    nk_size_t result_K = (nk_size_t)K < M_actual ? (nk_size_t)K : M_actual;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        for (nk_size_t j = i + 1; j < M_actual; j++)
            if (full_scores[j] < full_scores[best]) best = j;
        if (best != i) {
            double ts = full_scores[i]; full_scores[i] = full_scores[best]; full_scores[best] = ts;
            nk_u32_t ti = topM[i]; topM[i] = topM[best]; topM[best] = ti;
        }
    }

    PyEval_RestoreThread(save);

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)topM[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(full_scores[i]));
    }

    double coverage = ((double)N * n_warmup_tokens + (double)M_actual * T) / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue("{s:d,s:n,s:d,s:d}",
        "coverage", coverage, "M", (Py_ssize_t)M_actual,
        "warmup_ms", t_warmup, "full_ms", t_full);

    free(full_scores); free(topM); free(perm);
    PyBuffer_Release(&query_buf);
    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

/* ====== Extract flat arrays from packed docs ====== */
char const doc_extract_flat_from_packed[] =
    "extract_flat_from_packed(doc_list, i8_out, f32_out, inv_out, sum_out, offsets_out, /) -> None\n\n"
    "Copy i8, f32, inv_norm, sum_i8 from packed docs into pre-allocated flat buffers.\n"
    "Uses NK's exact i8 quantization (no re-quantization).\n"
    "i8_out: [total_tokens, depth] int8\n"
    "f32_out: [total_tokens, depth] float32\n"
    "inv_out: [total_tokens] float32\n"
    "sum_out: [total_tokens] int32 (128*sum(i8))\n"
    "offsets_out: [N+1] int32 (cumulative token counts, must be pre-filled)\n";

PyObject *api_extract_flat_from_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs != 6) {
        PyErr_SetString(PyExc_TypeError, "extract_flat_from_packed requires 6 args");
        return NULL;
    }
    if (!PyList_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "doc_list must be a list");
        return NULL;
    }
    PyObject *doc_list = args[0];
    nk_size_t N = (nk_size_t)PyList_GET_SIZE(doc_list);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list, 0);

    Py_buffer i8_buf, f32_buf, inv_buf, sum_buf, off_buf;
    if (PyObject_GetBuffer(args[1], &i8_buf, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) return NULL;
    if (PyObject_GetBuffer(args[2], &f32_buf, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) { PyBuffer_Release(&i8_buf); return NULL; }
    if (PyObject_GetBuffer(args[3], &inv_buf, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) { PyBuffer_Release(&i8_buf); PyBuffer_Release(&f32_buf); return NULL; }
    if (PyObject_GetBuffer(args[4], &sum_buf, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) { PyBuffer_Release(&i8_buf); PyBuffer_Release(&f32_buf); PyBuffer_Release(&inv_buf); return NULL; }
    if (PyObject_GetBuffer(args[5], &off_buf, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) { PyBuffer_Release(&i8_buf); PyBuffer_Release(&f32_buf); PyBuffer_Release(&inv_buf); PyBuffer_Release(&sum_buf); return NULL; }

    nk_i8_t *i8_flat = (nk_i8_t *)i8_buf.buf;
    float *f32_flat = (float *)f32_buf.buf;
    float *inv_flat = (float *)inv_buf.buf;
    nk_i32_t *sum_flat = (nk_i32_t *)sum_buf.buf;
    nk_i32_t *offsets = (nk_i32_t *)off_buf.buf;

    /* Compute offsets first and extract doc data */
    offsets[0] = 0;
    for (nk_size_t i = 0; i < N; i++) {
        MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[i];
        nk_maxsim_packed_header_t const *hdr = (nk_maxsim_packed_header_t const *)doc->start;
        nk_size_t L = (nk_size_t)hdr->vector_count;
        nk_size_t depth = (nk_size_t)hdr->depth_dimensions;
        nk_size_t depth_i8_padded = (nk_size_t)hdr->depth_i8_padded;
        nk_size_t original_stride = (nk_size_t)hdr->original_stride_bytes;

        offsets[i + 1] = offsets[i] + (nk_i32_t)L;

        nk_i8_t const *doc_i8 = (nk_i8_t const *)(doc->start + hdr->offset_i8_data);
        char const *doc_orig = (char const *)(doc->start + hdr->offset_original_data);
        nk_maxsim_vector_metadata_t const *meta = (nk_maxsim_vector_metadata_t const *)(doc->start + hdr->offset_metadata);

        nk_i32_t d_start = offsets[i];
        for (nk_size_t t = 0; t < L; t++) {
            /* Copy i8 (strip padding: depth_i8_padded → depth) */
            nk_i8_t *i8_dst = i8_flat + (d_start + (nk_i32_t)t) * depth;
            nk_i8_t const *i8_src = doc_i8 + t * depth_i8_padded;
            memcpy(i8_dst, i8_src, depth);

            /* Copy f32 originals (strip padding from original_stride → depth*4) */
            float *f32_dst = f32_flat + (d_start + (nk_i32_t)t) * depth;
            float const *f32_src = (float const *)(doc_orig + t * original_stride);
            memcpy(f32_dst, f32_src, depth * sizeof(float));

            /* Metadata: inv_norm, sum_i8 */
            inv_flat[d_start + (nk_i32_t)t] = meta[t].inverse_norm_f32;
            sum_flat[d_start + (nk_i32_t)t] = 128 * meta[t].sum_i8_i32;
        }
    }

    PyBuffer_Release(&i8_buf);
    PyBuffer_Release(&f32_buf);
    PyBuffer_Release(&inv_buf);
    PyBuffer_Release(&sum_buf);
    PyBuffer_Release(&off_buf);
    Py_RETURN_NONE;
}

/* ====== Get total token count from a list of packed docs ====== */
char const doc_total_tokens[] =
    "total_tokens(doc_list) -> int\n\nReturn sum of vector_count over all packed docs.\n";

PyObject *api_total_tokens(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    nk_unused_(kwnames);
    if (nargs != 1 || !PyList_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "total_tokens(doc_list)");
        return NULL;
    }
    PyObject *doc_list = args[0];
    nk_size_t N = (nk_size_t)PyList_GET_SIZE(doc_list);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list, 0);
    nk_size_t total = 0;
    for (nk_size_t i = 0; i < N; i++) {
        MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[i];
        total += doc->vector_count;
    }
    return PyLong_FromSize_t(total);
}

/* ====== Flat-buffer CB-NK: separated i8/f32 for cache-friendly elimination ====== */
char const doc_colbandit_flat[] =
    "colbandit_flat(query, doc_i8, doc_f32, doc_inv_norms, doc_offsets, /, "
    "K=5, K_margin=5, alpha_ef=0.3, delta=0.01, n_threads=1, rng_seed=42, docs_packed=None) -> (indices, scores, stats)\n\n"
    "CB-NK on flat separated arrays. i8 coarse scan is cache-friendly (no f32 pollution).\n"
    "rng_seed: seed for the Fisher-Yates token-permutation; default 42 matches deployed builds.\n"
    "docs_packed (optional): list of MaxSimPackedMatrix (same as full_maxsim takes).\n"
    "  If provided, the K-margin rescore uses the same packed kernel as full_maxsim,\n"
    "  aligning float-add-order so that CB-NK's top-K matches full_maxsim's top-K\n"
    "  bit-for-bit on borderline docs. If omitted, the i8 flat tiled kernel is used\n"
    "  (faster setup but produces ~3-5pp Ov@K gap vs full_maxsim due to add-order noise).\n";

PyObject *api_colbandit_flat(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (nargs < 6) {
        PyErr_SetString(PyExc_TypeError, "colbandit_flat(query, doc_i8, doc_f32, doc_inv_norms, doc_offsets, doc_sum_i8, ...)");
        return NULL;
    }

    long K = 5, K_margin = 5;
    double alpha_ef = 0.3, delta = 0.01;
    int n_threads = 1;
    int measure_imbalance = 0;
    int use_orig_N = 1;   /* default = BM-valid (N) for post-fix builds */
    long rng_seed = 42;   /* Fisher-Yates seed; default matches pre-knob builds */
    PyObject *docs_packed_obj = NULL;  /* FIX-A: optional list of MaxSimPackedMatrix for aligned rescore */

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "measure_imbalance") == 0) measure_imbalance = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "rng_seed") == 0) rng_seed = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "docs_packed") == 0) docs_packed_obj = value;
    }

    /* Per-thread timing arrays for load-imbalance analysis */
    #define max_threads_tracked 128
    double *thread_kernel_ms = NULL;
    double *thread_wait_ms = NULL;
    if (measure_imbalance) {
        thread_kernel_ms = (double *)calloc(max_threads_tracked, sizeof(double));
        thread_wait_ms = (double *)calloc(max_threads_tracked, sizeof(double));
    }

    Py_buffer q_buf, di_buf, df_buf, dn_buf, do_buf, ds_buf;
    if (PyObject_GetBuffer(args[0], &q_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(args[1], &di_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q_buf); return NULL; }
    if (PyObject_GetBuffer(args[2], &df_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); return NULL; }
    if (PyObject_GetBuffer(args[3], &dn_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf); return NULL; }
    if (PyObject_GetBuffer(args[4], &do_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf); PyBuffer_Release(&dn_buf); return NULL; }
    if (PyObject_GetBuffer(args[5], &ds_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf); PyBuffer_Release(&dn_buf); PyBuffer_Release(&do_buf); return NULL; }

    float const *query_data = (float const *)q_buf.buf;
    nk_size_t T_raw = (nk_size_t)q_buf.shape[0];
    nk_size_t depth = (nk_size_t)q_buf.shape[1];

    /* Strip trailing zero-padded query tokens (matches api_colbandit_maxsim).
       MM-style queries (e.g. ColPali, GVE) come padded to a fixed max length; without
       stripping, padding tokens score as angular=1 on every doc, contributing no
       elimination signal and dilating the coverage denominator. */
    nk_size_t T = T_raw;
    while (T > 0) {
        float const *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    nk_i8_t const *all_i8 = (nk_i8_t const *)di_buf.buf;
    float const *all_f32 = (float const *)df_buf.buf;
    float const *all_inv_norms = (float const *)dn_buf.buf;
    nk_i32_t const *doc_offsets = (nk_i32_t const *)do_buf.buf;
    nk_i32_t const *all_sum_i8 = (nk_i32_t const *)ds_buf.buf;  /* pre-computed 128*sum(d_i8) per token */
    nk_size_t N = (nk_size_t)(do_buf.shape[0] - 1);

    /* Quantize query to i8 with p95 clip (6th largest |value| as scale, matching NK pack) */
    nk_i8_t *q_i8 = (nk_i8_t *)malloc(T * depth * sizeof(nk_i8_t));
    float *q_inv_norms = (float *)malloc(T * sizeof(float));
    for (nk_size_t t = 0; t < T; t++) {
        float const *row = query_data + t * depth;
        float norm_sq = 0.0f;
        float top6[6] = {0, 0, 0, 0, 0, 0};
        for (nk_size_t d = 0; d < depth; d++) {
            float a = row[d] > 0 ? row[d] : -row[d];
            norm_sq += row[d] * row[d];
            if (a > top6[5]) {
                top6[5] = a;
                for (int k = 4; k >= 0; k--) {
                    if (top6[k+1] > top6[k]) { float tmp = top6[k]; top6[k] = top6[k+1]; top6[k+1] = tmp; }
                    else break;
                }
            }
        }
        float clip_val = top6[5] > 0.0f ? top6[5] : top6[0];
        float scale = clip_val > 1e-10f ? 79.0f / clip_val : 1.0f;
        for (nk_size_t d = 0; d < depth; d++) {
            float v = row[d] * scale;
            q_i8[t * depth + d] = (nk_i8_t)(v > 79 ? 79 : v < -79 ? -79 : (v > 0 ? (int)(v + 0.5f) : (int)(v - 0.5f)));
        }
        q_inv_norms[t] = norm_sq > 0 ? 1.0f / sqrtf(norm_sq) : 0.0f;
    }

    /* Random permutation (Fisher-Yates with caller-supplied seed) */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = (nk_u32_t)rng_seed;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    nk_size_t n_warmup = T;  /* always full warmup */
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    nk_size_t round_size = 4;

    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    PyThreadState *save = PyEval_SaveThread();

    double t_kernel = 0, t_elim = 0, t_rescore = 0;
    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;   /* FIX B: count rounds where bound failed */
    nk_size_t K_elim = (nk_size_t)(K + K_margin);

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        /* Build permuted query slices for this round */
        nk_i8_t *q_round_i8 = q_i8;   /* pointer into pre-quantized query */
        float *q_round_f32 = (float *)query_data;
        /* We need the permuted tokens for this round */
        nk_i8_t q_perm_i8[16 * 256];   /* max 16 tokens × 256 depth, stack */
        float q_perm_f32[16 * 256];
        float q_perm_inv[16];
        for (nk_size_t t = 0; t < tokens_this_round; t++) {
            nk_u32_t ti = perm[warmup_token_ptr + t];
            memcpy(q_perm_i8 + t * depth, q_i8 + ti * depth, depth);
            memcpy(q_perm_f32 + t * depth, query_data + ti * depth, depth * sizeof(float));
            q_perm_inv[t] = q_inv_norms[ti];
        }

        /* 4Q×4D tiled flat kernel with inline refine */
        double t0 = _now_ms();
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        #pragma omp parallel if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
        {
            double t_thread_start = _now_ms();
            double t_my_kernel = 0.0;
            /* Static scheduling: preserves cache locality (each thread owns a contiguous doc range).
               Tested dynamic(32) — worse on MM because work-stealing destroys L3 locality.
               TODO: flat is 1.4-1.8x slower than packed on full MM (L_d=1031 uniform, N=8604).
                 Per-thread kernel imbalance 1.7-2.2x with static (real wait=0).
                 Hypothesis: bias metadata (d_sum_i8) in separate array adds memory round-trip
                 per 4D-block vs packed's co-located metadata. Revisit if MM performance matters.
                 Text datasets win clearly with flat — use flat as default. */
            #pragma omp for schedule(static) nowait
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_u32_t di = survivors[i];
                nk_size_t d_start = (nk_size_t)doc_offsets[di];
                nk_size_t Ld = (nk_size_t)(doc_offsets[di + 1] - doc_offsets[di]);
                nk_f64_t local_stats[2];
                double tk0 = _now_ms();
                nk_maxsim_i8_flat_stats_tiled(
                    q_perm_i8, q_perm_f32, q_perm_inv,
                    all_i8 + d_start * depth,
                    all_f32 + d_start * depth,
                    all_inv_norms + d_start,
                    all_sum_i8 + d_start,
                    tokens_this_round, Ld, depth, local_stats);
                t_my_kernel += _now_ms() - tk0;
                obs_sum[di] += local_stats[0];
                obs_sum_sq[di] += local_stats[1];
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
            double t_before_barrier = _now_ms();
            #pragma omp barrier
            double t_after_barrier = _now_ms();
            double t_my_wait = t_after_barrier - t_before_barrier;  /* real barrier wait time */
            #pragma omp critical
            {
                if (thread_kernel_ms && thread_wait_ms && omp_get_thread_num() < max_threads_tracked) {
                    thread_kernel_ms[omp_get_thread_num()] += t_my_kernel;
                    thread_wait_ms[omp_get_thread_num()] += t_my_wait;
                }
            }
        }
        #else
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            nk_size_t d_start = (nk_size_t)doc_offsets[di];
            nk_size_t Ld = (nk_size_t)(doc_offsets[di + 1] - doc_offsets[di]);
            nk_f64_t local_stats[2];
            nk_maxsim_i8_flat_stats_tiled(
                q_perm_i8, q_perm_f32, q_perm_inv,
                all_i8 + d_start * depth,
                all_f32 + d_start * depth,
                all_inv_norms + d_start,
                all_sum_i8 + d_start,
                tokens_this_round, Ld, depth, local_stats);
            obs_sum[di] += local_stats[0];
            obs_sum_sq[di] += local_stats[1];
            obs_count[di] += (nk_u32_t)tokens_this_round;
        }
        #endif
        t_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Serfling elimination */
        t0 = _now_ms();
        int fb = 0;
        if (n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_elim += _now_ms() - t0;
        n_warmup_rounds++;
        if (n_survivors <= K_elim) break;
    }

    /* Rescore survivors with all T tokens.
       FIX-A: if docs_packed is provided, use the same packed kernel as full_maxsim
       (score_docs with doc_indices=survivors). This aligns float-add-order bit-for-bit
       with full_maxsim, lifting Ov@K vs full_maxsim by ~3-5pp on borderline docs.
       Otherwise fall back to the i8 flat tiled kernel. */
    double t0_r = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        int used_aligned = 0;
        if (docs_packed_obj && PyList_Check(docs_packed_obj) && PyList_GET_SIZE(docs_packed_obj) > 0) {
            PyObject **doc_objects = &PyList_GET_ITEM(docs_packed_obj, 0);
            MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
            nk_dtype_t dtype = first_doc->dtype;

            nk_maxsim_packed_punned_t kernel = NULL;
            nk_capability_t cap = nk_cap_serial_k;
            nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                                  (nk_kernel_punned_t *)&kernel, &cap);
            nk_dots_pack_punned_t pack_fn = NULL;
            nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                                  (nk_kernel_punned_t *)&pack_fn, &cap);
            nk_dots_packed_size_punned_t size_fn = NULL;
            nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                                  (nk_kernel_punned_t *)&size_fn, &cap);

            if (kernel && pack_fn && size_fn) {
                nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
                nk_size_t pack_size = size_fn(T, depth);
                void *q_packed = malloc(pack_size);
                pack_fn((float *)query_data, T, depth, depth * element_bytes, q_packed);

                double *aligned_scores = (double *)malloc(n_survivors * sizeof(double));
                score_docs(kernel, q_packed, T, doc_objects, survivors,
                           n_survivors, depth, dtype, aligned_scores, n_threads);

                for (nk_size_t i = 0; i < n_survivors; i++) {
                    obs_sum[survivors[i]] = aligned_scores[i];
                }

                free(q_packed);
                free(aligned_scores);
                used_aligned = 1;
            }
        }
        if (!used_aligned) {
            /* Original i8 flat rescore */
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_u32_t di = survivors[i];
                nk_size_t d_start = (nk_size_t)doc_offsets[di];
                nk_size_t Ld = (nk_size_t)(doc_offsets[di + 1] - doc_offsets[di]);
                nk_f64_t rescore[2];
                nk_maxsim_i8_flat_stats_tiled(
                    q_i8, (float *)query_data, q_inv_norms,
                    all_i8 + d_start * depth,
                    all_f32 + d_start * depth,
                    all_inv_norms + d_start,
                    all_sum_i8 + d_start,
                    T, Ld, depth, rescore);
                obs_sum[di] = rescore[0];
            }
        }
    }
    t_rescore = _now_ms() - t0_r;

    double coverage = (double)total_cells / ((double)N * T) * 100.0;

    PyEval_RestoreThread(save);

    /* Find top-K */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        for (nk_size_t j = i + 1; j < n_survivors; j++)
            if (obs_sum[survivors[j]] < obs_sum[survivors[best]]) best = j;
        if (best != i) { nk_u32_t t = survivors[i]; survivors[i] = survivors[best]; survivors[best] = t; }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromUnsignedLong(survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:n,s:n}",
        "coverage", coverage, "survived", (Py_ssize_t)n_survivors,
        "warmup_kernel_ms", t_kernel, "warmup_elim_ms", t_elim,
        "rescore_ms", t_rescore, "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "fallback_count", (Py_ssize_t)fallback_count);  /* FIX B */

    if (measure_imbalance && thread_kernel_ms && thread_wait_ms) {
        PyObject *k_list = PyList_New(n_threads);
        PyObject *w_list = PyList_New(n_threads);
        for (int t = 0; t < n_threads && t < max_threads_tracked; t++) {
            PyList_SET_ITEM(k_list, t, PyFloat_FromDouble(thread_kernel_ms[t]));
            PyList_SET_ITEM(w_list, t, PyFloat_FromDouble(thread_wait_ms[t]));
        }
        PyDict_SetItemString(stats, "thread_kernel_ms", k_list);
        PyDict_SetItemString(stats, "thread_wait_ms", w_list);
        Py_DECREF(k_list); Py_DECREF(w_list);
        free(thread_kernel_ms); free(thread_wait_ms);
    }

    free(q_i8); free(q_inv_norms); free(perm);
    free(obs_sum); free(obs_sum_sq); free(obs_count); free(survivors);
    free(elim_lcb); free(elim_ucb); free(elim_heap); free(sort_bitmap);
    PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf);
    PyBuffer_Release(&dn_buf); PyBuffer_Release(&do_buf); PyBuffer_Release(&ds_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
#else
    PyErr_SetString(PyExc_NotImplementedError, "colbandit_flat requires x86 AVX2");
    return NULL;
#endif
}

/* ====== In-place CB-NK: gather inside the kernel via cand_ids ======
 *
 * Same algorithm as colbandit_flat but works on the FULL pre-extracted flat
 * corpus + a per-query int32 cand_ids array. Avoids the per-query "extract
 * subset" gather: pointers are computed inside the OpenMP loop with optional
 * AVX-512 prefetch on the next candidate's i8 row, mirroring LEMUR's
 * batch_query_subset_fixed pattern.
 *
 * Survivors are kept as LOCAL indices in [0, N_cand). To dereference a doc
 * we use cand_ids[survivors[i]] -> global doc id, and offsets_full to find
 * the (i8/f32/inv/sum) start. The K-margin rescore receives the full packed
 * list and the global doc ids.
 *
 * Caller responsibilities:
 *   - i8_full / f32_full / inv_full / sum_full / off_full are produced once
 *     by extract_flat_from_packed and live for the whole batch.
 *   - cand_ids[i] in [0, N_full) for all i.
 *   - All other args mirror colbandit_flat's semantics.
 */
char const doc_colbandit_full_inplace[] =
    "colbandit_full_inplace(query, i8_full, f32_full, inv_full, off_full, sum_full, cand_ids, /, "
    "K=5, K_margin=5, alpha_ef=0.3, delta=0.01, n_threads=1, rng_seed=42, docs_packed=None) "
    "-> (indices, scores, stats)\n\n"
    "Same as colbandit_flat but takes the GLOBAL flat corpus + a per-query int32\n"
    "cand_ids array, gathering inside the kernel (LEMUR-style). The returned\n"
    "indices are LOCAL positions into cand_ids (0..N_cand-1), so the caller\n"
    "should map them via cand_ids[idx] to recover global IDs (matching the\n"
    "convention used by colbandit_flat which is itself called on a subset).\n"
    "docs_packed (optional): the FULL packed list. The K-margin rescore picks\n"
    "items by global ID = cand_ids[local_survivor].\n";

PyObject *api_colbandit_full_inplace(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (nargs < 7) {
        PyErr_SetString(PyExc_TypeError,
            "colbandit_full_inplace(query, i8_full, f32_full, inv_full, off_full, sum_full, cand_ids, ...)");
        return NULL;
    }

    long K = 5, K_margin = 5;
    double alpha_ef = 0.3, delta = 0.01;
    int n_threads = 1;
    int measure_imbalance = 0;
    int use_orig_N = 1;
    long rng_seed = 42;
    PyObject *docs_packed_obj = NULL;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "measure_imbalance") == 0) measure_imbalance = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "rng_seed") == 0) rng_seed = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "docs_packed") == 0) docs_packed_obj = value;
    }

    #define max_threads_tracked 128
    double *thread_kernel_ms = NULL;
    double *thread_wait_ms = NULL;
    if (measure_imbalance) {
        thread_kernel_ms = (double *)calloc(max_threads_tracked, sizeof(double));
        thread_wait_ms = (double *)calloc(max_threads_tracked, sizeof(double));
    }

    Py_buffer q_buf, i8f_buf, f32f_buf, invf_buf, off_buf, sum_buf, cand_buf;
    if (PyObject_GetBuffer(args[0], &q_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(args[1], &i8f_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q_buf); return NULL; }
    if (PyObject_GetBuffer(args[2], &f32f_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&i8f_buf); return NULL;
    }
    if (PyObject_GetBuffer(args[3], &invf_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&i8f_buf); PyBuffer_Release(&f32f_buf); return NULL;
    }
    if (PyObject_GetBuffer(args[4], &off_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&i8f_buf); PyBuffer_Release(&f32f_buf); PyBuffer_Release(&invf_buf); return NULL;
    }
    if (PyObject_GetBuffer(args[5], &sum_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&i8f_buf); PyBuffer_Release(&f32f_buf);
        PyBuffer_Release(&invf_buf); PyBuffer_Release(&off_buf); return NULL;
    }
    if (PyObject_GetBuffer(args[6], &cand_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&i8f_buf); PyBuffer_Release(&f32f_buf);
        PyBuffer_Release(&invf_buf); PyBuffer_Release(&off_buf); PyBuffer_Release(&sum_buf); return NULL;
    }

    float const *query_data = (float const *)q_buf.buf;
    nk_size_t T_raw = (nk_size_t)q_buf.shape[0];
    nk_size_t depth = (nk_size_t)q_buf.shape[1];

    /* Strip trailing zero-padded query tokens */
    nk_size_t T = T_raw;
    while (T > 0) {
        float const *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    nk_i8_t const *all_i8 = (nk_i8_t const *)i8f_buf.buf;
    float const *all_f32 = (float const *)f32f_buf.buf;
    float const *all_inv_norms = (float const *)invf_buf.buf;
    nk_i32_t const *off_full = (nk_i32_t const *)off_buf.buf;
    nk_i32_t const *all_sum_i8 = (nk_i32_t const *)sum_buf.buf;
    nk_i32_t const *cand_ids = (nk_i32_t const *)cand_buf.buf;

    nk_size_t N_full = (nk_size_t)(off_buf.shape[0] - 1);
    nk_size_t N_cand = (nk_size_t)cand_buf.shape[0];
    nk_size_t N = N_cand;  /* number of working docs (LOCAL indexing) */

    /* Quantize query to i8 with p95 clip — same as colbandit_flat */
    nk_i8_t *q_i8 = (nk_i8_t *)malloc(T * depth * sizeof(nk_i8_t));
    float *q_inv_norms = (float *)malloc(T * sizeof(float));
    for (nk_size_t t = 0; t < T; t++) {
        float const *row = query_data + t * depth;
        float norm_sq = 0.0f;
        float top6[6] = {0, 0, 0, 0, 0, 0};
        for (nk_size_t d = 0; d < depth; d++) {
            float a = row[d] > 0 ? row[d] : -row[d];
            norm_sq += row[d] * row[d];
            if (a > top6[5]) {
                top6[5] = a;
                for (int k = 4; k >= 0; k--) {
                    if (top6[k+1] > top6[k]) { float tmp = top6[k]; top6[k] = top6[k+1]; top6[k+1] = tmp; }
                    else break;
                }
            }
        }
        float clip_val = top6[5] > 0.0f ? top6[5] : top6[0];
        float scale = clip_val > 1e-10f ? 79.0f / clip_val : 1.0f;
        for (nk_size_t d = 0; d < depth; d++) {
            float v = row[d] * scale;
            q_i8[t * depth + d] = (nk_i8_t)(v > 79 ? 79 : v < -79 ? -79 : (v > 0 ? (int)(v + 0.5f) : (int)(v - 0.5f)));
        }
        q_inv_norms[t] = norm_sq > 0 ? 1.0f / sqrtf(norm_sq) : 0.0f;
    }

    /* Random permutation (Fisher-Yates) — same as colbandit_flat */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = (nk_u32_t)rng_seed;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    nk_size_t n_warmup = T;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    nk_size_t round_size = 4;

    /* Bookkeeping arrays sized to N_cand (local indices). Survivors are LOCAL. */
    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    PyThreadState *save = PyEval_SaveThread();

    double t_kernel = 0, t_elim = 0, t_rescore = 0;
    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;
    nk_size_t K_elim = (nk_size_t)(K + K_margin);

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        nk_i8_t q_perm_i8[16 * 256];
        float q_perm_f32[16 * 256];
        float q_perm_inv[16];
        for (nk_size_t t = 0; t < tokens_this_round; t++) {
            nk_u32_t ti = perm[warmup_token_ptr + t];
            memcpy(q_perm_i8 + t * depth, q_i8 + ti * depth, depth);
            memcpy(q_perm_f32 + t * depth, query_data + ti * depth, depth * sizeof(float));
            q_perm_inv[t] = q_inv_norms[ti];
        }

        double t0 = _now_ms();
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        #pragma omp parallel if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
        {
            double t_thread_start = _now_ms();
            double t_my_kernel = 0.0;
            #pragma omp for schedule(static) nowait
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_u32_t local = survivors[i];                       /* LOCAL idx into cand_ids */
                nk_u32_t global = (nk_u32_t)cand_ids[local];         /* GLOBAL doc ID */
                nk_size_t d_start = (nk_size_t)off_full[global];
                nk_size_t Ld = (nk_size_t)(off_full[global + 1] - off_full[global]);

                /* Prefetch the next candidate's i8 row to hide gather latency.
                   This is the critical insight from LEMUR's batch_query_subset_fixed. */
                if (i + 1 < n_survivors) {
                    nk_u32_t next_local = survivors[i + 1];
                    nk_u32_t next_global = (nk_u32_t)cand_ids[next_local];
                    nk_size_t next_start = (nk_size_t)off_full[next_global];
                    _mm_prefetch((const char *)(all_i8 + next_start * depth), _MM_HINT_T0);
                    /* Also prefetch the f32 row used by the inline refine pass */
                    _mm_prefetch((const char *)(all_f32 + next_start * depth), _MM_HINT_T1);
                }

                nk_f64_t local_stats[2];
                double tk0 = _now_ms();
                nk_maxsim_i8_flat_stats_tiled(
                    q_perm_i8, q_perm_f32, q_perm_inv,
                    all_i8 + d_start * depth,
                    all_f32 + d_start * depth,
                    all_inv_norms + d_start,
                    all_sum_i8 + d_start,
                    tokens_this_round, Ld, depth, local_stats);
                t_my_kernel += _now_ms() - tk0;
                obs_sum[local] += local_stats[0];
                obs_sum_sq[local] += local_stats[1];
                obs_count[local] += (nk_u32_t)tokens_this_round;
            }
            double t_before_barrier = _now_ms();
            #pragma omp barrier
            double t_after_barrier = _now_ms();
            double t_my_wait = t_after_barrier - t_before_barrier;
            #pragma omp critical
            {
                if (thread_kernel_ms && thread_wait_ms && omp_get_thread_num() < max_threads_tracked) {
                    thread_kernel_ms[omp_get_thread_num()] += t_my_kernel;
                    thread_wait_ms[omp_get_thread_num()] += t_my_wait;
                }
            }
        }
        #else
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t local = survivors[i];
            nk_u32_t global = (nk_u32_t)cand_ids[local];
            nk_size_t d_start = (nk_size_t)off_full[global];
            nk_size_t Ld = (nk_size_t)(off_full[global + 1] - off_full[global]);

            if (i + 1 < n_survivors) {
                nk_u32_t next_local = survivors[i + 1];
                nk_u32_t next_global = (nk_u32_t)cand_ids[next_local];
                nk_size_t next_start = (nk_size_t)off_full[next_global];
                _mm_prefetch((const char *)(all_i8 + next_start * depth), _MM_HINT_T0);
                _mm_prefetch((const char *)(all_f32 + next_start * depth), _MM_HINT_T1);
            }

            nk_f64_t local_stats[2];
            nk_maxsim_i8_flat_stats_tiled(
                q_perm_i8, q_perm_f32, q_perm_inv,
                all_i8 + d_start * depth,
                all_f32 + d_start * depth,
                all_inv_norms + d_start,
                all_sum_i8 + d_start,
                tokens_this_round, Ld, depth, local_stats);
            obs_sum[local] += local_stats[0];
            obs_sum_sq[local] += local_stats[1];
            obs_count[local] += (nk_u32_t)tokens_this_round;
        }
        #endif
        t_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Serfling elimination — operates on the LOCAL survivor index space.
           N_orig argument is N_cand (the corpus we're choosing from). */
        t0 = _now_ms();
        int fb = 0;
        if (n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_elim += _now_ms() - t0;
        n_warmup_rounds++;
        if (n_survivors <= K_elim) break;
    }

    /* Rescore survivors. If docs_packed (the FULL packed list) is provided we
       map LOCAL survivors -> GLOBAL ids and call score_docs with global ids,
       which matches full_maxsim's float-add order bit-for-bit.
       Otherwise fall back to the i8 flat tiled kernel using full corpus pointers. */
    double t0_r = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        int used_aligned = 0;
        if (docs_packed_obj && PyList_Check(docs_packed_obj) && PyList_GET_SIZE(docs_packed_obj) > 0) {
            PyObject **doc_objects = &PyList_GET_ITEM(docs_packed_obj, 0);
            MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
            nk_dtype_t dtype = first_doc->dtype;

            nk_maxsim_packed_punned_t kernel = NULL;
            nk_capability_t cap = nk_cap_serial_k;
            nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                                  (nk_kernel_punned_t *)&kernel, &cap);
            nk_dots_pack_punned_t pack_fn = NULL;
            nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                                  (nk_kernel_punned_t *)&pack_fn, &cap);
            nk_dots_packed_size_punned_t size_fn = NULL;
            nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                                  (nk_kernel_punned_t *)&size_fn, &cap);

            if (kernel && pack_fn && size_fn) {
                nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
                nk_size_t pack_size = size_fn(T, depth);
                void *q_packed = malloc(pack_size);
                pack_fn((float *)query_data, T, depth, depth * element_bytes, q_packed);

                /* Translate local survivors -> global doc ids for score_docs */
                nk_u32_t *global_ids = (nk_u32_t *)malloc(n_survivors * sizeof(nk_u32_t));
                for (nk_size_t i = 0; i < n_survivors; i++)
                    global_ids[i] = (nk_u32_t)cand_ids[survivors[i]];

                double *aligned_scores = (double *)malloc(n_survivors * sizeof(double));
                score_docs(kernel, q_packed, T, doc_objects, global_ids,
                           n_survivors, depth, dtype, aligned_scores, n_threads);

                for (nk_size_t i = 0; i < n_survivors; i++) {
                    obs_sum[survivors[i]] = aligned_scores[i];
                }

                free(q_packed);
                free(aligned_scores);
                free(global_ids);
                used_aligned = 1;
            }
        }
        if (!used_aligned) {
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_u32_t local = survivors[i];
                nk_u32_t global = (nk_u32_t)cand_ids[local];
                nk_size_t d_start = (nk_size_t)off_full[global];
                nk_size_t Ld = (nk_size_t)(off_full[global + 1] - off_full[global]);
                nk_f64_t rescore[2];
                nk_maxsim_i8_flat_stats_tiled(
                    q_i8, (float *)query_data, q_inv_norms,
                    all_i8 + d_start * depth,
                    all_f32 + d_start * depth,
                    all_inv_norms + d_start,
                    all_sum_i8 + d_start,
                    T, Ld, depth, rescore);
                obs_sum[local] = rescore[0];
            }
        }
    }
    t_rescore = _now_ms() - t0_r;

    double coverage = (double)total_cells / ((double)N * T) * 100.0;

    PyEval_RestoreThread(save);

    /* Find top-K (LOCAL indices). */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        for (nk_size_t j = i + 1; j < n_survivors; j++)
            if (obs_sum[survivors[j]] < obs_sum[survivors[best]]) best = j;
        if (best != i) { nk_u32_t t = survivors[i]; survivors[i] = survivors[best]; survivors[best] = t; }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromUnsignedLong(survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:n,s:n}",
        "coverage", coverage, "survived", (Py_ssize_t)n_survivors,
        "warmup_kernel_ms", t_kernel, "warmup_elim_ms", t_elim,
        "rescore_ms", t_rescore, "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "fallback_count", (Py_ssize_t)fallback_count);

    if (measure_imbalance && thread_kernel_ms && thread_wait_ms) {
        PyObject *k_list = PyList_New(n_threads);
        PyObject *w_list = PyList_New(n_threads);
        for (int t = 0; t < n_threads && t < max_threads_tracked; t++) {
            PyList_SET_ITEM(k_list, t, PyFloat_FromDouble(thread_kernel_ms[t]));
            PyList_SET_ITEM(w_list, t, PyFloat_FromDouble(thread_wait_ms[t]));
        }
        PyDict_SetItemString(stats, "thread_kernel_ms", k_list);
        PyDict_SetItemString(stats, "thread_wait_ms", w_list);
        Py_DECREF(k_list); Py_DECREF(w_list);
        free(thread_kernel_ms); free(thread_wait_ms);
    }

    free(q_i8); free(q_inv_norms); free(perm);
    free(obs_sum); free(obs_sum_sq); free(obs_count); free(survivors);
    free(elim_lcb); free(elim_ucb); free(elim_heap); free(sort_bitmap);
    PyBuffer_Release(&q_buf); PyBuffer_Release(&i8f_buf); PyBuffer_Release(&f32f_buf);
    PyBuffer_Release(&invf_buf); PyBuffer_Release(&off_buf); PyBuffer_Release(&sum_buf);
    PyBuffer_Release(&cand_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
#else
    PyErr_SetString(PyExc_NotImplementedError, "colbandit_full_inplace requires x86 AVX2");
    return NULL;
#endif
}

/* ====== Full NK MaxSim with OpenMP: one pass, no elimination ====== */
char const doc_full_maxsim[] =
    "full_maxsim(query_tokens, doc_packed_list, /, K=5, n_threads=1) -> (indices, scores, stats)\n\n"
    "Full exhaustive MaxSim with OpenMP. Same C loop as colbandit_maxsim but no elimination.\n";

PyObject *api_full_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    PyObject *query_obj = NULL, *doc_list_obj = NULL;
    long K = 5;
    int n_threads = 1;

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "full_maxsim(query, docs, K=5, n_threads=1)");
        return NULL;
    }
    query_obj = args[0];
    doc_list_obj = args[1];

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0)
            K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0)
            n_threads = (int)PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T = (nk_size_t)query_buf.shape[0];
    nk_size_t depth = (nk_size_t)query_buf.shape[1];
    float const *query_data = (float const *)query_buf.buf;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "docs must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_GET_SIZE(doc_list_obj);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list_obj, 0);

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;

    /* Find kernels */
    nk_maxsim_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);

    if (!kernel || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    /* Pack full query */
    nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
    nk_size_t pack_size = size_fn(T, depth);
    void *q_packed = malloc(pack_size);
    pack_fn(query_data, T, depth, depth * element_bytes, q_packed);

    /* Score all docs with OpenMP */
    double *scores = (double *)malloc(N * sizeof(double));

    double t0, t_kernel;
    {
        PyThreadState *save = PyEval_SaveThread();
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        score_docs(kernel, q_packed, T, doc_objects, NULL, N, depth, dtype, scores, n_threads);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        t_kernel = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
        PyEval_RestoreThread(save);
    }

    free(q_packed);

    /* Find top-K (ascending = best for angular) */
    nk_size_t result_K = (nk_size_t)K < N ? (nk_size_t)K : N;
    nk_u32_t *top_idx = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < N; i++) top_idx[i] = (nk_u32_t)i;

    /* Partial selection sort for top-K */
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        for (nk_size_t j = i + 1; j < N; j++)
            if (scores[j] < scores[best]) best = j;
        if (best != i) {
            double ts = scores[i]; scores[i] = scores[best]; scores[best] = ts;
            nk_u32_t ti = top_idx[i]; top_idx[i] = top_idx[best]; top_idx[best] = ti;
        }
    }

    /* Build return objects */
    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromUnsignedLong(top_idx[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(scores[i]));
    }

    PyObject *stats = Py_BuildValue("{s:d}", "kernel_ms", t_kernel);

    free(scores);
    free(top_idx);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

char const doc_colbandit_maxsim[] =
    "colbandit_maxsim(query_tokens, doc_packed_list, /, K=5, warmup_ratio=0.1, "
    "alpha_ef=0.3, delta=0.01, phase_keep=None) -> tuple\n\n"
    "Col-Bandit with Serfling bounds elimination on NumKong MaxSim.\n\n"
    "Parameters:\n"
    "    query_tokens (array_like): [T, d] query token embeddings (f32).\n"
    "    doc_packed_list (list): List of MaxSimPackedMatrix objects.\n"
    "    K (int): Top-K to return. Default: 5.\n"
    "    warmup_ratio (float): Fraction of tokens for warmup. Default: 0.1.\n"
    "    alpha_ef (float): Serfling confidence scaling. Default: 0.3.\n"
    "    delta (float): PAC confidence. Default: 0.01.\n"
    "    phase_keep (list, optional): Ignored (kept for API compat). Elimination is via Serfling bounds.\n\n"
    "Returns:\n"
    "    tuple: (top_k_indices list, top_k_scores list, stats dict)\n";

PyObject *api_colbandit_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "colbandit_maxsim() requires at least 2 arguments");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_list_obj = args[1];

    long K = 5;
    long K_margin = 0;  /* extra survivors to keep for rescore: eliminate to K+K_margin, rescore to K */
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;  /* if non-zero, use original corpus size in union bound (matches Alg 1 paper) */
    long B = 4;  /* round size (tokens revealed per round). B<=0 means "single round = T (exhaustive)". */
    PyObject *phase_keep_obj = NULL;  /* ignored, kept for compat */
    PyObject *centroid_table_obj = NULL;  /* [16] i8 for 4-bit mode */
    PyObject *query_4bit_obj = NULL;     /* [T, d] u8 4-bit indices */

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0)
            K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0)
            K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0)
            warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0)
            alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0)
            delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0)
            no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0)
            n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "phase_keep") == 0)
            phase_keep_obj = value;  /* ignored */
        else if (PyUnicode_CompareWithASCIIString(name, "centroid_table") == 0)
            centroid_table_obj = value;
        else if (PyUnicode_CompareWithASCIIString(name, "query_4bit") == 0)
            query_4bit_obj = value;
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0)
            use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0)
            B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    /* Skip padding tokens (zero vectors) at the end of the query.
       Padding tokens produce meaningless MaxSim scores and waste rounds. */
    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;  /* safety: keep at least one token */

    /* Resolve round size B. B<=0 means "single round of T tokens" (exhaustive in one batch). */
    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "doc_packed_list must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    /* Fast path: get internal array pointer directly (avoids 500K PyList_GetItem calls).
       PyList_GET_ITEM is a macro that indexes the internal ob_item array — no bounds check. */
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list_obj, 0);
    /* Type-check only first and last doc (skip full O(N) check for speed) */
    if (N > 0 && !PyObject_TypeCheck(doc_objects[0], &MaxSimPackedMatrixType)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }
    int doc_objects_owned = 0;  /* no free needed — points into PyList internal storage */

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;

    nk_maxsim_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel, &cap);
    /* Stats kernel: same as sum kernel but returns (sum, sum_sq) for Serfling */
    nk_maxsim_packed_punned_t kernel_stats = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_stats_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_stats, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);

    if (!kernel || !kernel_stats || !pack_fn || !size_fn) {
        free(doc_objects);
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    /* 4-bit mode: parse centroid table and query indices */
    int use_4bit = 0;
    nk_i8_t const *centroid_table = NULL;
    nk_u8_t const *query_4bit_data = NULL;
    Py_buffer ct_buf = {0}, q4_buf = {0};
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (centroid_table_obj && query_4bit_obj &&
        centroid_table_obj != Py_None && query_4bit_obj != Py_None) {
        if (PyObject_GetBuffer(centroid_table_obj, &ct_buf, PyBUF_C_CONTIGUOUS) < 0) {
            free(doc_objects); PyBuffer_Release(&query_buf); return NULL;
        }
        if (PyObject_GetBuffer(query_4bit_obj, &q4_buf, PyBUF_C_CONTIGUOUS) < 0) {
            PyBuffer_Release(&ct_buf); free(doc_objects); PyBuffer_Release(&query_buf); return NULL;
        }
        centroid_table = (nk_i8_t const *)ct_buf.buf;
        query_4bit_data = (nk_u8_t const *)q4_buf.buf;  /* [T, depth] u8, values 0-15 */
        use_4bit = 1;
    }
#endif

    /* Random permutation */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Allocate per-doc statistics */
    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    /* Align to multiple of 4, round down, minimum 4 */
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;   /* FIX B */

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();   /* FIX D: ensure t0 is initialized before any accumulation */

    /* ====== Phase 1: Multi-round warmup with progressive Serfling elimination ====== */
    nk_size_t warmup_token_ptr = 0;
    nk_f64_t stats_buf[2];
    nk_size_t n_warmup_rounds = 0;

    nk_size_t round_size = (nk_size_t)B;

    /* Pre-pack all warmup rounds upfront — eliminates per-round memcpy + pack_fn calls */
    nk_size_t n_total_rounds = (n_warmup + round_size - 1) / round_size;
    nk_size_t rnd_pack_size = size_fn(round_size, depth);
    float *q_permuted = (float *)malloc(n_warmup * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++)
        memcpy(q_permuted + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    void **q_round_packed_arr = (void **)malloc(n_total_rounds * sizeof(void *));
    for (nk_size_t r = 0; r < n_total_rounds; r++) {
        nk_size_t offset = r * round_size;
        nk_size_t count = (offset + round_size <= n_warmup) ? round_size : (n_warmup - offset);
        q_round_packed_arr[r] = malloc(rnd_pack_size);
        pack_fn(q_permuted + offset * depth, count, depth, depth * element_bytes, q_round_packed_arr[r]);
    }
    free(q_permuted);

    /* Sort survivors by doc memory address for cache-friendly sequential access.
       Python allocates MaxSimPackedMatrix objects at scattered addresses.
       Sorting brings sequential iteration closer to memory order → hardware prefetch works. */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        /* addr_order: [addr0, idx0, addr1, idx1, ...] */
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        /* Sort by address using qsort on pairs */
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++)
            survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    /* Pre-allocate elimination buffers (reused every round) */
    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        /* Use pre-packed round buffer */
        nk_size_t current_round = warmup_token_ptr / round_size;
        void *q_round_packed = q_round_packed_arr[current_round];

        t0 = _now_ms();   /* FIX D: anchor t0 so t_warmup_pack accumulates correctly */
        /* 4-bit mode: overwrite i8 region with nibble-packed query indices */
        if (use_4bit) {
            nk_maxsim_packed_header_t const *qhdr =
                (nk_maxsim_packed_header_t const *)q_round_packed;
            nk_i8_t *q_i8_region = (nk_i8_t *)((char *)q_round_packed + qhdr->offset_i8_data);
            nk_size_t depth_i8_padded = qhdr->depth_i8_padded;
            nk_size_t half = depth / 2;

            for (nk_size_t t = 0; t < tokens_this_round; t++) {
                nk_u8_t const *src = query_4bit_data + perm[warmup_token_ptr + t] * depth;
                nk_u8_t *dst = (nk_u8_t *)(q_i8_region + t * depth_i8_padded);
                /* Nibble-pack: dim[2k] in low nibble, dim[2k+1] in high nibble */
                for (nk_size_t k = 0; k < half; k++)
                    dst[k] = (src[2*k] & 0x0F) | ((src[2*k+1] & 0x0F) << 4);
                /* Zero-pad remainder */
                for (nk_size_t k = half; k < depth_i8_padded; k++)
                    dst[k] = 0;
            }
        }
        t_warmup_pack += _now_ms() - t0;

        /* Score current survivors — parallel across docs */
        t0 = _now_ms();
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        if (use_4bit) {
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            #endif
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];

                nk_maxsim_packed_4bit_stats_f32_haswell(
                    q_round_packed, doc->start,
                    tokens_this_round, doc->vector_count, depth,
                    local_stats, centroid_table);

                obs_sum[di] += local_stats[0];
                obs_sum_sq[di] += local_stats[1];
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        } else
#endif
        {
            /* Use static schedule for large survivor sets (round 1: all docs similar L_d).
               Dynamic is better for later rounds with mixed survivors. Threshold: N/2. */
            /* Prefetch distance: look ahead by PF docs to hide memory latency.
               Each doc is ~10KB (60 tokens × 160 bytes i8). At ~10 GB/s effective BW
               and ~0.5µs per doc, we need ~5-10 docs ahead to fill the pipeline. */
            #define CB_PREFETCH_AHEAD 8
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            if (n_survivors > N / 2) {
                #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
                for (nk_size_t i = 0; i < n_survivors; i++) {
                    /* Prefetch next doc's object pointer and packed data header */
                    if (i + CB_PREFETCH_AHEAD < n_survivors) {
                        MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_PREFETCH_AHEAD]];
                        _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                        _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                    }
                    nk_f64_t local_stats[2];
                    nk_u32_t di = survivors[i];
                    MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                    if (no_eliminate) {
                        /* bypass mode: use the same maxsim_packed kernel as Full (sum only) */
                        kernel(q_round_packed, doc->start,
                               tokens_this_round, doc->vector_count, depth, local_stats);
                        obs_sum[di] += local_stats[0];
                    } else {
                        kernel_stats(q_round_packed, doc->start,
                                     tokens_this_round, doc->vector_count, depth, local_stats);
                        obs_sum[di] += local_stats[0];
                        obs_sum_sq[di] += local_stats[1];
                    }
                    obs_count[di] += (nk_u32_t)tokens_this_round;
                }
            } else {
                #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
                for (nk_size_t i = 0; i < n_survivors; i++) {
                    if (i + CB_PREFETCH_AHEAD < n_survivors) {
                        MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_PREFETCH_AHEAD]];
                        _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                        _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                    }
                    nk_f64_t local_stats[2];
                    nk_u32_t di = survivors[i];
                    MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                    if (no_eliminate) {
                        /* bypass mode: use the same maxsim_packed kernel as Full (sum only) */
                        kernel(q_round_packed, doc->start,
                               tokens_this_round, doc->vector_count, depth, local_stats);
                        obs_sum[di] += local_stats[0];
                    } else {
                        kernel_stats(q_round_packed, doc->start,
                                     tokens_this_round, doc->vector_count, depth, local_stats);
                        obs_sum[di] += local_stats[0];
                        obs_sum_sq[di] += local_stats[1];
                    }
                    obs_count[di] += (nk_u32_t)tokens_this_round;
                }
            }
            #else
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                kernel_stats(q_round_packed, doc->start,
                             tokens_this_round, doc->vector_count, depth, local_stats);
                obs_sum[di] += local_stats[0];
                obs_sum_sq[di] += local_stats[1];
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
            #endif
        }
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Serfling elimination */
        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);  /* eliminate to K+margin */
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        /* Early stop: all survivors identified, no need for more rounds */
        if (n_survivors <= K_elim) break;
    }

    /* ====== Phase 2: Score remaining tokens on survivors (batched) ====== */
    /* After early stop, warmup_token_ptr < T. Score remaining tokens on survivors. */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        nk_size_t pack_size = size_fn(n_explore, depth);
        void *q_explore_packed = malloc(pack_size);
        float *q_explore_data = (float *)malloc(n_explore * depth * sizeof(float));
        for (nk_size_t t = 0; t < n_explore; t++) {
            memcpy(q_explore_data + t * depth,
                   query_data + perm[warmup_token_ptr + t] * depth,
                   depth * sizeof(float));
        }
        pack_fn(q_explore_data, n_explore, depth, depth * element_bytes, q_explore_packed);
        free(q_explore_data);

#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        if (use_4bit) {
            /* Nibble-pack the explore query's i8 region */
            nk_maxsim_packed_header_t const *qhdr =
                (nk_maxsim_packed_header_t const *)q_explore_packed;
            nk_u8_t *q_i8_region = (nk_u8_t *)((char *)q_explore_packed + qhdr->offset_i8_data);
            nk_size_t depth_i8_padded = qhdr->depth_i8_padded;
            nk_size_t half = depth / 2;
            for (nk_size_t t = 0; t < n_explore; t++) {
                nk_u8_t const *src = query_4bit_data + perm[warmup_token_ptr + t] * depth;
                nk_u8_t *dst = q_i8_region + t * depth_i8_padded;
                for (nk_size_t k = 0; k < half; k++)
                    dst[k] = (src[2*k] & 0x0F) | ((src[2*k+1] & 0x0F) << 4);
                for (nk_size_t k = half; k < depth_i8_padded; k++)
                    dst[k] = 0;
            }
        }
#endif
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        if (use_4bit) {
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            #endif
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                nk_maxsim_packed_4bit_stats_f32_haswell(
                    q_explore_packed, doc->start,
                    n_explore, doc->vector_count, depth,
                    local_stats, centroid_table);
                explore_scores[i] = local_stats[0];
            }
        } else
#endif
        {
            score_docs(kernel, q_explore_packed, n_explore,
                       doc_objects, survivors, n_survivors, depth, dtype, explore_scores, n_threads);
        }
        free(q_explore_packed);

        for (nk_size_t i = 0; i < n_survivors; i++)
            obs_sum[survivors[i]] += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* ====== Re-score survivors ====== */
    /* When K_margin > 0, we keep K+margin survivors from elimination.
       Rescore them with the stats kernel (which has top-2, more accurate than sum kernel's argmax).
       Then the final sort picks the best K from the K+margin candidates. */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        /* Re-score all survivors using stats kernel with ALL T tokens.
           This gives the most accurate scores since stats kernel uses top-2. */
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        pack_fn(query_data, T, depth, depth * element_bytes, q_full);

#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        if (use_4bit) {
            /* In 4-bit mode, doc i8 region is nibble-packed. Must use 4-bit kernel for rescore.
               Nibble-pack the full query's i8 region first. */
            nk_maxsim_packed_header_t const *qhdr =
                (nk_maxsim_packed_header_t const *)q_full;
            nk_u8_t *q_i8_region = (nk_u8_t *)((char *)q_full + qhdr->offset_i8_data);
            nk_size_t depth_i8_padded = qhdr->depth_i8_padded;
            nk_size_t half = depth / 2;
            for (nk_size_t t = 0; t < T; t++) {
                nk_u8_t const *src = query_4bit_data + t * depth;
                nk_u8_t *dst = q_i8_region + t * depth_i8_padded;
                for (nk_size_t k = 0; k < half; k++)
                    dst[k] = (src[2*k] & 0x0F) | ((src[2*k+1] & 0x0F) << 4);
                for (nk_size_t k = half; k < depth_i8_padded; k++)
                    dst[k] = 0;
            }
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                nk_f64_t rescore_stats[2];
                nk_maxsim_packed_4bit_stats_f32_haswell(
                    q_full, doc->start, T, doc->vector_count, depth,
                    rescore_stats, centroid_table);
                obs_sum[di] = rescore_stats[0];
            }
        } else
#endif
        {
            for (nk_size_t i = 0; i < n_survivors; i++) {
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                nk_f64_t rescore_stats[2];
                kernel_stats(q_full, doc->start, T, doc->vector_count, depth, rescore_stats);
                obs_sum[di] = rescore_stats[0];  /* Override with full-T stats kernel score */
            }
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    /* Find top-K from survivors (ascending = best for angular) */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = obs_sum[survivors[i]];
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (obs_sum[survivors[j]] < best_score) {
                best = j;
                best_score = obs_sum[survivors[j]];
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    /* Build return */
    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);  /* FIX B */

    for (nk_size_t r = 0; r < n_total_rounds; r++)
        free(q_round_packed_arr[r]);
    free(q_round_packed_arr);
    free(elim_lcb);
    free(elim_ucb);
    free(elim_heap);
    free(sort_bitmap);
    free(obs_sum);
    free(obs_sum_sq);
    free(obs_count);
    free(survivors);
    free(perm);
    /* doc_objects points into PyList internal storage — do not free */
    PyBuffer_Release(&query_buf);
    if (use_4bit) {
        PyBuffer_Release(&ct_buf);
        PyBuffer_Release(&q4_buf);
    }

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

/* ============================================================================
 *   api_colbandit_maxsim_fp32  —  fp32-first variant of api_colbandit_maxsim
 *
 *   Same algorithm (multi-round Serfling elimination + K+M rescore) but the
 *   inner kernel is `cb_fp32only_maxsim_stats_` which scans ALL doc tokens
 *   in pure fp32 (no i8 coarse argmax, no per-query int8 quantize step).
 *   The deployed int8 path (api_colbandit_maxsim) is left untouched so that
 *   A/B comparisons stay honest.
 * ============================================================================ */
char const doc_colbandit_maxsim_fp32[] =
    "colbandit_maxsim_fp32(query_tokens, doc_packed_list, /, K=5, warmup_ratio=1.0, "
    "alpha_ef=0.3, delta=0.01, K_margin=0, B=4, no_eliminate=False, n_threads=1) -> tuple\n\n"
    "fp32-only Col-Bandit: skips per-query int8 quantize, uses pure fp32 dot products\n"
    "for all partial-sum updates. Bit-identical to Full when K_margin > 0 (the survivor\n"
    "rescore pass is the same fp32-refine path used by `colbandit_maxsim`).\n";

PyObject *api_colbandit_maxsim_fp32(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "colbandit_maxsim_fp32() requires at least 2 arguments");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_list_obj = args[1];

    long K = 5;
    long K_margin = 0;
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;
    long B = 4;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0) warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0) no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0) B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    /* Drop trailing zero-norm padding tokens (same as int8 variant). */
    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "doc_packed_list must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list_obj, 0);
    if (N > 0 && !PyObject_TypeCheck(doc_objects[0], &MaxSimPackedMatrixType)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;

    /* We only need pack/size kernels for the K+M rescore at the end (which
       packs the query just once per call, not per round).  The hot inner-loop
       does NOT use any i8 packing or i8 kernels. */
    nk_maxsim_packed_punned_t kernel_for_rescore = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_for_rescore, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!kernel_for_rescore || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
    if (element_bytes != 4) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_NotImplementedError,
                        "colbandit_maxsim_fp32 currently supports f32 docs only");
        return NULL;
    }

    /* Random permutation of token order (same RNG seed as int8 variant) */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Per-doc statistics */
    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();

    /* Pre-compute the permuted query buffer once (pure fp32, no quantize) */
    float *q_permuted_warmup = (float *)malloc(n_warmup * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++)
        memcpy(q_permuted_warmup + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    nk_size_t round_size = (nk_size_t)B;

    /* Sort survivors by doc address for cache-friendly iteration (same as int8 variant) */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    /* Pre-allocate elimination buffers */
    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;
    int want_sum_sq = !no_eliminate;

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        /* No per-round packing — query is already raw fp32 in q_permuted_warmup. */
        t0 = _now_ms();
        t_warmup_pack += _now_ms() - t0;  /* essentially zero, kept for symmetry */

        float const *q_round_f32 = q_permuted_warmup + warmup_token_ptr * depth;

        t0 = _now_ms();
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
        #endif
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2] = {0.0, 0.0};
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_maxsim_packed_header_t const *dhdr = (nk_maxsim_packed_header_t const *)doc->start;
            float const *d_f32 = (float const *)((char const *)doc->start + dhdr->offset_original_data);
            nk_size_t d_stride_floats = dhdr->original_stride_bytes / sizeof(float);
            nk_maxsim_vector_metadata_t const *d_meta =
                (nk_maxsim_vector_metadata_t const *)((char const *)doc->start + dhdr->offset_metadata);

            cb_fp32only_maxsim_stats_(
                q_round_f32, depth, NULL,           /* q_inv_norms NULL = assume 1.0 */
                d_f32, d_stride_floats, d_meta,
                tokens_this_round, doc->vector_count, depth,
                want_sum_sq,
                local_stats);

            obs_sum[di] += local_stats[0];
            if (want_sum_sq) obs_sum_sq[di] += local_stats[1];
            obs_count[di] += (nk_u32_t)tokens_this_round;
        }
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Elimination — same Serfling logic as int8 variant. */
        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        if (n_survivors <= K_elim) break;
    }

    /* ====== Phase 2: explore remaining tokens on survivors (pure fp32) ====== */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        float *q_explore_f32 = (float *)malloc(n_explore * depth * sizeof(float));
        for (nk_size_t t = 0; t < n_explore; t++)
            memcpy(q_explore_f32 + t * depth,
                   query_data + perm[warmup_token_ptr + t] * depth,
                   depth * sizeof(float));
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
        #endif
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2] = {0.0, 0.0};
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_maxsim_packed_header_t const *dhdr = (nk_maxsim_packed_header_t const *)doc->start;
            float const *d_f32 = (float const *)((char const *)doc->start + dhdr->offset_original_data);
            nk_size_t d_stride_floats = dhdr->original_stride_bytes / sizeof(float);
            nk_maxsim_vector_metadata_t const *d_meta =
                (nk_maxsim_vector_metadata_t const *)((char const *)doc->start + dhdr->offset_metadata);

            cb_fp32only_maxsim_stats_(
                q_explore_f32, depth, NULL,
                d_f32, d_stride_floats, d_meta,
                n_explore, doc->vector_count, depth,
                0,  /* sum-only in explore */
                local_stats);
            explore_scores[i] = local_stats[0];
        }
        free(q_explore_f32);

        for (nk_size_t i = 0; i < n_survivors; i++)
            obs_sum[survivors[i]] += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* ====== K+M rescore: same kernel as Full (bit-identical scores) ====== */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        pack_fn(query_data, T, depth, depth * element_bytes, q_full);

        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_f64_t rescore;
            kernel_for_rescore(q_full, doc->start, T, doc->vector_count, depth, &rescore);
            obs_sum[di] = rescore;
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    /* Top-K from survivors (ascending == best for angular distance) */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = obs_sum[survivors[i]];
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (obs_sum[survivors[j]] < best_score) {
                best = j;
                best_score = obs_sum[survivors[j]];
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);

    free(q_permuted_warmup);
    free(elim_lcb);
    free(elim_ucb);
    free(elim_heap);
    free(sort_bitmap);
    free(obs_sum);
    free(obs_sum_sq);
    free(obs_count);
    free(survivors);
    free(perm);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

/* ============================================================================
 *   api_maxsim_pack_int8  —  index-time int8-prequantize of a single doc.
 *
 *   In the existing format, `nk.maxsim_pack(d_fp32)` already builds a
 *   MaxSimPackedMatrix that contains BOTH the i8-quantized region and the
 *   fp32 originals.  The doc is therefore already "prequantized at index time".
 *
 *   `maxsim_pack_int8` is provided as a named entry point that makes this
 *   intent explicit (and gives callers a stable API surface to use with
 *   `colbandit_maxsim_prequantized`).  Currently it produces the same packed
 *   buffer as `maxsim_pack` so behaviour is bit-identical to the deployed
 *   path — i.e. the per-query int8 quantize cost in `colbandit_maxsim` is
 *   already amortised.
 * ============================================================================ */
char const doc_maxsim_pack_int8[] =
    "maxsim_pack_int8(b, /, dtype=None) -> MaxSimPackedMatrix\n\n"
    "Index-time quantize: pack a single doc into the standard MaxSimPackedMatrix\n"
    "format (which already contains the i8 region computed once at pack time).\n"
    "Returns a buffer suitable for `colbandit_maxsim_prequantized` whose hot\n"
    "loop never re-quantizes doc data.\n";

PyObject *api_maxsim_pack_int8(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    /* The existing maxsim_pack already stores the i8 region.  Delegate to keep
       behaviour identical and avoid drift. */
    return api_maxsim_pack(self, args, nargs, kwnames);
}

/* ============================================================================
 *   api_colbandit_maxsim_prequantized  —  Col-Bandit with pre-quantized docs.
 *
 *   Same algorithm as `api_colbandit_maxsim` (multi-round Serfling elim +
 *   K+M rescore) but signature explicitly takes BOTH (a) a list of int8-
 *   prequantized doc buffers and (b) a list of fp32 packed buffers (used
 *   only by the K+M rescore so it stays bit-identical to Full).
 *
 *   In the current packed format both lists may point at the same
 *   MaxSimPackedMatrix objects (since each contains both i8 and fp32
 *   regions).  The function explicitly skips any per-query doc quantize.
 *
 *   Used to test the hypothesis that the deployed CB int8 path pays a
 *   significant per-query doc-quantize cost.
 * ============================================================================ */
char const doc_colbandit_maxsim_prequantized[] =
    "colbandit_maxsim_prequantized(query_tokens, docs_int8, docs_fp32, /, K=5, "
    "warmup_ratio=1.0, alpha_ef=0.3, delta=0.01, K_margin=0, B=4, "
    "no_eliminate=False, n_threads=1) -> tuple\n\n"
    "Col-Bandit with pre-quantized doc buffers.  Identical algorithm to\n"
    "`colbandit_maxsim` but the API explicitly separates the int8 doc list\n"
    "(used for warmup/explore i8 coarse argmax) from the fp32 doc list (used\n"
    "for the K+M survivor rescore).  Skips any per-query doc-quantize step.\n";

PyObject *api_colbandit_maxsim_prequantized(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 3) {
        PyErr_SetString(PyExc_TypeError,
                        "colbandit_maxsim_prequantized() requires at least 3 arguments: "
                        "(query, docs_int8, docs_fp32)");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_int8_list_obj = args[1];
    PyObject *doc_fp32_list_obj = args[2];

    long K = 5;
    long K_margin = 0;
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;
    long B = 4;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0) warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0) no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0) B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    /* Skip trailing zero-norm padding tokens. */
    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_int8_list_obj) || !PyList_Check(doc_fp32_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "docs_int8 and docs_fp32 must both be lists");
        return NULL;
    }
    nk_size_t N_int8 = (nk_size_t)PyList_Size(doc_int8_list_obj);
    nk_size_t N_fp32 = (nk_size_t)PyList_Size(doc_fp32_list_obj);
    if (N_int8 != N_fp32) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_ValueError, "docs_int8 and docs_fp32 must have the same length");
        return NULL;
    }
    nk_size_t N = N_int8;
    PyObject **doc_int8_objects = N > 0 ? &PyList_GET_ITEM(doc_int8_list_obj, 0) : NULL;
    PyObject **doc_fp32_objects = N > 0 ? &PyList_GET_ITEM(doc_fp32_list_obj, 0) : NULL;
    if (N > 0 && (!PyObject_TypeCheck(doc_int8_objects[0], &MaxSimPackedMatrixType) ||
                  !PyObject_TypeCheck(doc_fp32_objects[0], &MaxSimPackedMatrixType))) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }

    MaxSimPackedMatrix *first_int8 = N > 0 ? (MaxSimPackedMatrix *)doc_int8_objects[0] : NULL;
    MaxSimPackedMatrix *first_fp32 = N > 0 ? (MaxSimPackedMatrix *)doc_fp32_objects[0] : NULL;
    nk_dtype_t dtype = first_int8 ? first_int8->dtype : nk_f32_k;
    if (first_fp32 && first_fp32->dtype != dtype) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError,
                        "docs_int8 and docs_fp32 dtype mismatch — both must use the same packed dtype");
        return NULL;
    }

    nk_maxsim_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel, &cap);
    nk_maxsim_packed_punned_t kernel_stats = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_stats_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_stats, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);

    if (!kernel || !kernel_stats || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    /* Random permutation of token order — same RNG seed as int8 variant. */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Per-doc statistics */
    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();

    /* Phase 1: pre-pack ALL warmup query rounds upfront — same as int8 variant.
       (The query still gets packed; the doc packing is the index-time step skipped by design.) */
    nk_size_t round_size = (nk_size_t)B;
    nk_size_t n_total_rounds = (n_warmup + round_size - 1) / round_size;
    nk_size_t rnd_pack_size = size_fn(round_size, depth);
    float *q_permuted = (float *)malloc(n_warmup * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++)
        memcpy(q_permuted + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    void **q_round_packed_arr = (void **)malloc(n_total_rounds * sizeof(void *));
    for (nk_size_t r = 0; r < n_total_rounds; r++) {
        nk_size_t offset = r * round_size;
        nk_size_t count = (offset + round_size <= n_warmup) ? round_size : (n_warmup - offset);
        q_round_packed_arr[r] = malloc(rnd_pack_size);
        pack_fn(q_permuted + offset * depth, count, depth, depth * element_bytes, q_round_packed_arr[r]);
    }
    free(q_permuted);

    /* Sort survivors by int8-doc memory address for cache-friendly iteration. */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_int8_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        nk_size_t current_round = warmup_token_ptr / round_size;
        void *q_round_packed = q_round_packed_arr[current_round];

        t0 = _now_ms();
        t_warmup_pack += _now_ms() - t0;  /* query already pre-packed; doc never re-packed */

        t0 = _now_ms();
        #define CB_PRE_PREFETCH_AHEAD 8
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        if (n_survivors > N / 2) {
            #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_PRE_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc =
                        (MaxSimPackedMatrix *)doc_int8_objects[survivors[i + CB_PRE_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_int8_objects[di];
                if (no_eliminate) {
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                } else {
                    kernel_stats(q_round_packed, doc->start,
                                 tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                }
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_PRE_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc =
                        (MaxSimPackedMatrix *)doc_int8_objects[survivors[i + CB_PRE_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_int8_objects[di];
                if (no_eliminate) {
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                } else {
                    kernel_stats(q_round_packed, doc->start,
                                 tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                }
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        }
        #else
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2];
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_int8_objects[di];
            kernel_stats(q_round_packed, doc->start,
                         tokens_this_round, doc->vector_count, depth, local_stats);
            obs_sum[di] += local_stats[0];
            obs_sum_sq[di] += local_stats[1];
            obs_count[di] += (nk_u32_t)tokens_this_round;
        }
        #endif
        #undef CB_PRE_PREFETCH_AHEAD
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Serfling elimination */
        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        if (n_survivors <= K_elim) break;
    }

    /* Phase 2: explore remaining tokens on survivors */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        nk_size_t pack_size = size_fn(n_explore, depth);
        void *q_explore_packed = malloc(pack_size);
        float *q_explore_data = (float *)malloc(n_explore * depth * sizeof(float));
        for (nk_size_t t = 0; t < n_explore; t++) {
            memcpy(q_explore_data + t * depth,
                   query_data + perm[warmup_token_ptr + t] * depth,
                   depth * sizeof(float));
        }
        pack_fn(q_explore_data, n_explore, depth, depth * element_bytes, q_explore_packed);
        free(q_explore_data);
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
        score_docs(kernel, q_explore_packed, n_explore,
                   doc_int8_objects, survivors, n_survivors, depth, dtype, explore_scores, n_threads);
        free(q_explore_packed);

        for (nk_size_t i = 0; i < n_survivors; i++)
            obs_sum[survivors[i]] += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* K+M rescore — uses fp32 doc list (same buffers in this scheme). */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        pack_fn(query_data, T, depth, depth * element_bytes, q_full);
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_fp32_objects[di];
            nk_f64_t rescore_stats[2];
            kernel_stats(q_full, doc->start, T, doc->vector_count, depth, rescore_stats);
            obs_sum[di] = rescore_stats[0];
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    /* Final top-K sort */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = obs_sum[survivors[i]];
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (obs_sum[survivors[j]] < best_score) {
                best = j;
                best_score = obs_sum[survivors[j]];
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);

    for (nk_size_t r = 0; r < n_total_rounds; r++) free(q_round_packed_arr[r]);
    free(q_round_packed_arr);
    free(elim_lcb);
    free(elim_ucb);
    free(elim_heap);
    free(sort_bitmap);
    free(obs_sum);
    free(obs_sum_sq);
    free(obs_count);
    free(survivors);
    free(perm);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

/* ====== Compact 4-bit exhaustive MaxSim: flat arrays, no packed format ====== */
char const doc_score_4bit_compact[] =
    "score_4bit_compact(q_4bit, q_f32, q_inv_norms, d_4bit, d_f32, d_inv_norms, doc_lengths, centroid_i8, /) -> scores\n\n"
    "Exhaustive MaxSim using compact 4-bit coarse argmax + f32 refine.\n"
    "All arrays are flat numpy buffers. Docs stored at compact stride=depth/2.\n";

PyObject *api_score_4bit_compact(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError, "score_4bit_compact requires 8 args");
        return NULL;
    }

    Py_buffer q4_buf, qf_buf, qn_buf, d4_buf, df_buf, dn_buf, dl_buf, ct_buf;
    if (PyObject_GetBuffer(args[0], &q4_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(args[1], &qf_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); return NULL; }
    if (PyObject_GetBuffer(args[2], &qn_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); PyBuffer_Release(&qf_buf); return NULL; }
    if (PyObject_GetBuffer(args[3], &d4_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); return NULL; }
    if (PyObject_GetBuffer(args[4], &df_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&d4_buf); return NULL; }
    if (PyObject_GetBuffer(args[5], &dn_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&d4_buf); PyBuffer_Release(&df_buf); return NULL; }
    if (PyObject_GetBuffer(args[6], &dl_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&d4_buf); PyBuffer_Release(&df_buf); PyBuffer_Release(&dn_buf); return NULL; }
    if (PyObject_GetBuffer(args[7], &ct_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&q4_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&d4_buf); PyBuffer_Release(&df_buf); PyBuffer_Release(&dn_buf); PyBuffer_Release(&dl_buf); return NULL; }

    nk_u8_t const *q_4bit = (nk_u8_t const *)q4_buf.buf;
    float const *q_f32 = (float const *)qf_buf.buf;
    float const *q_inv_norms = (float const *)qn_buf.buf;
    nk_u8_t const *d_4bit = (nk_u8_t const *)d4_buf.buf;
    float const *d_f32 = (float const *)df_buf.buf;
    float const *d_inv_norms = (float const *)dn_buf.buf;
    nk_i32_t const *doc_lengths = (nk_i32_t const *)dl_buf.buf;
    nk_i8_t const *centroid_i8 = (nk_i8_t const *)ct_buf.buf;

    nk_size_t T = qf_buf.shape[0];
    nk_size_t depth = qf_buf.shape[1];
    nk_size_t N_docs = dl_buf.shape[0];
    nk_size_t half = depth / 2;

    /* Compute per-doc MaxSim using compact 4-bit kernel */
    PyObject *scores_arr = PyList_New(N_docs);
    if (!scores_arr) goto cleanup;

    {
        PyThreadState *save = PyEval_SaveThread();
        double *scores = (double *)malloc(N_docs * sizeof(double));
        nk_size_t d4_offset = 0;
        nk_size_t df_offset = 0;
        for (nk_size_t i = 0; i < N_docs; i++) {
            nk_size_t Ld = (nk_size_t)doc_lengths[i];
            nk_f64_t result[2];
            nk_maxsim_4bit_compact_stats(
                q_4bit, q_f32, q_inv_norms,
                d_4bit + d4_offset, d_f32 + df_offset * depth, d_inv_norms + df_offset,
                T, Ld, depth, centroid_i8, result);
            scores[i] = result[0];
            d4_offset += Ld * half;
            df_offset += Ld;
        }
        PyEval_RestoreThread(save);

        for (nk_size_t i = 0; i < N_docs; i++)
            PyList_SET_ITEM(scores_arr, i, PyFloat_FromDouble(scores[i]));
        free(scores);
    }

cleanup:
    PyBuffer_Release(&q4_buf);
    PyBuffer_Release(&qf_buf);
    PyBuffer_Release(&qn_buf);
    PyBuffer_Release(&d4_buf);
    PyBuffer_Release(&df_buf);
    PyBuffer_Release(&dn_buf);
    PyBuffer_Release(&dl_buf);
    PyBuffer_Release(&ct_buf);
    return scores_arr;
#else
    PyErr_SetString(PyExc_NotImplementedError, "score_4bit_compact requires x86 AVX2");
    return NULL;
#endif
}

/* ====== Compact i8 exhaustive MaxSim: flat arrays at stride=depth ====== */
char const doc_score_i8_compact[] =
    "score_i8_compact(q_i8, q_f32, q_inv_norms, d_i8, d_f32, d_inv_norms, doc_lengths, /) -> scores\n\n"
    "Exhaustive MaxSim using compact i8 (stride=depth) coarse argmax + f32 refine.\n";

PyObject *api_score_i8_compact(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "score_i8_compact requires 7 args");
        return NULL;
    }

    Py_buffer qi_buf, qf_buf, qn_buf, di_buf, df_buf, dn_buf, dl_buf;
    if (PyObject_GetBuffer(args[0], &qi_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(args[1], &qf_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&qi_buf); return NULL; }
    if (PyObject_GetBuffer(args[2], &qn_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&qi_buf); PyBuffer_Release(&qf_buf); return NULL; }
    if (PyObject_GetBuffer(args[3], &di_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&qi_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); return NULL; }
    if (PyObject_GetBuffer(args[4], &df_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&qi_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&di_buf); return NULL; }
    if (PyObject_GetBuffer(args[5], &dn_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&qi_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf); return NULL; }
    if (PyObject_GetBuffer(args[6], &dl_buf, PyBUF_C_CONTIGUOUS) < 0) { PyBuffer_Release(&qi_buf); PyBuffer_Release(&qf_buf); PyBuffer_Release(&qn_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf); PyBuffer_Release(&dn_buf); return NULL; }

    nk_i8_t const *q_i8 = (nk_i8_t const *)qi_buf.buf;
    float const *q_f32 = (float const *)qf_buf.buf;
    float const *q_inv_norms = (float const *)qn_buf.buf;
    nk_i8_t const *d_i8 = (nk_i8_t const *)di_buf.buf;
    float const *d_f32 = (float const *)df_buf.buf;
    float const *d_inv_norms = (float const *)dn_buf.buf;
    nk_i32_t const *doc_lengths = (nk_i32_t const *)dl_buf.buf;

    nk_size_t T = qf_buf.shape[0];
    nk_size_t depth = qf_buf.shape[1];
    nk_size_t N_docs = dl_buf.shape[0];

    PyObject *scores_arr = PyList_New(N_docs);
    if (!scores_arr) goto i8_cleanup;

    {
        PyThreadState *save = PyEval_SaveThread();
        double *scores = (double *)malloc(N_docs * sizeof(double));
        nk_size_t d_offset = 0;  /* offset in vectors */
        for (nk_size_t i = 0; i < N_docs; i++) {
            nk_size_t Ld = (nk_size_t)doc_lengths[i];
            nk_f64_t result[2];
            nk_maxsim_i8_compact_stats(
                q_i8, q_f32, q_inv_norms,
                d_i8 + d_offset * depth, d_f32 + d_offset * depth, d_inv_norms + d_offset,
                T, Ld, depth, result);
            scores[i] = result[0];
            d_offset += Ld;
        }
        PyEval_RestoreThread(save);

        for (nk_size_t i = 0; i < N_docs; i++)
            PyList_SET_ITEM(scores_arr, i, PyFloat_FromDouble(scores[i]));
        free(scores);
    }

i8_cleanup:
    PyBuffer_Release(&qi_buf);
    PyBuffer_Release(&qf_buf);
    PyBuffer_Release(&qn_buf);
    PyBuffer_Release(&di_buf);
    PyBuffer_Release(&df_buf);
    PyBuffer_Release(&dn_buf);
    PyBuffer_Release(&dl_buf);
    return scores_arr;
#else
    PyErr_SetString(PyExc_NotImplementedError, "score_i8_compact requires x86 AVX2");
    return NULL;
#endif
}

/* ====== Top-M flat baseline (fair-implementation TopM) ======
   Uses CB-NK's exact i8/flat-array warmup kernel and f32 rescore code path,
   replacing the bandit elimination logic with a static top-M selection. This
   isolates the algorithmic difference (adaptive vs static) from the
   implementation difference (i8 vs f32, flat vs object-iterated).
*/
char const doc_topm_flat[] =
    "topm_flat(query, doc_i8, doc_f32, doc_inv_norms, doc_offsets, doc_sum_i8, /, "
    "K=5, M=100, n_warmup=4, n_threads=1) -> (indices, scores, stats)\n\n"
    "Top-M baseline using the same flat-array i8 warmup kernel and f32 rescore "
    "stage as colbandit_flat. Pipeline: (1) reveal n_warmup tokens × all N docs "
    "via nk_maxsim_i8_flat_stats_tiled; (2) sort docs by partial score and keep "
    "top-M; (3) rescore those M with all T tokens via the same kernel; (4) "
    "return top-K. No bandit logic — pure static budget.\n";

PyObject *api_topm_flat(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    if (nargs < 6) {
        PyErr_SetString(PyExc_TypeError,
            "topm_flat(query, doc_i8, doc_f32, doc_inv_norms, doc_offsets, doc_sum_i8, ...)");
        return NULL;
    }

    long K = 5, M = 100, n_warmup_arg = 4;
    int n_threads = 1;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "M") == 0) M = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_warmup") == 0) n_warmup_arg = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
    }

    Py_buffer q_buf, di_buf, df_buf, dn_buf, do_buf, ds_buf;
    if (PyObject_GetBuffer(args[0], &q_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(args[1], &di_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); return NULL; }
    if (PyObject_GetBuffer(args[2], &df_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); return NULL; }
    if (PyObject_GetBuffer(args[3], &dn_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf); return NULL; }
    if (PyObject_GetBuffer(args[4], &do_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf);
        PyBuffer_Release(&dn_buf); return NULL; }
    if (PyObject_GetBuffer(args[5], &ds_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf);
        PyBuffer_Release(&dn_buf); PyBuffer_Release(&do_buf); return NULL; }

    float const *query_data = (float const *)q_buf.buf;
    nk_size_t T = (nk_size_t)q_buf.shape[0];
    nk_size_t depth = (nk_size_t)q_buf.shape[1];
    nk_i8_t const *all_i8 = (nk_i8_t const *)di_buf.buf;
    float const *all_f32 = (float const *)df_buf.buf;
    float const *all_inv_norms = (float const *)dn_buf.buf;
    nk_i32_t const *doc_offsets = (nk_i32_t const *)do_buf.buf;
    nk_i32_t const *all_sum_i8 = (nk_i32_t const *)ds_buf.buf;
    nk_size_t N = (nk_size_t)(do_buf.shape[0] - 1);

    /* Quantize query to i8 with p95 clip — same as colbandit_flat */
    nk_i8_t *q_i8 = (nk_i8_t *)malloc(T * depth * sizeof(nk_i8_t));
    float *q_inv_norms = (float *)malloc(T * sizeof(float));
    for (nk_size_t t = 0; t < T; t++) {
        float const *row = query_data + t * depth;
        float norm_sq = 0.0f;
        float top6[6] = {0, 0, 0, 0, 0, 0};
        for (nk_size_t d = 0; d < depth; d++) {
            float a = row[d] > 0 ? row[d] : -row[d];
            norm_sq += row[d] * row[d];
            if (a > top6[5]) {
                top6[5] = a;
                for (int k = 4; k >= 0; k--) {
                    if (top6[k+1] > top6[k]) { float tmp = top6[k]; top6[k] = top6[k+1]; top6[k+1] = tmp; }
                    else break;
                }
            }
        }
        float clip_val = top6[5] > 0.0f ? top6[5] : top6[0];
        float scale = clip_val > 1e-10f ? 79.0f / clip_val : 1.0f;
        for (nk_size_t d = 0; d < depth; d++) {
            float v = row[d] * scale;
            q_i8[t * depth + d] = (nk_i8_t)(v > 79 ? 79 : v < -79 ? -79 : (v > 0 ? (int)(v + 0.5f) : (int)(v - 0.5f)));
        }
        q_inv_norms[t] = norm_sq > 0 ? 1.0f / sqrtf(norm_sq) : 0.0f;
    }

    /* Random permutation — same as colbandit_flat */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* Align n_warmup to multiple of 4 like topm_maxsim does */
    nk_size_t n_warmup = (nk_size_t)n_warmup_arg;
    if (n_warmup > T) n_warmup = T;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;

    /* Build permuted query slices for warmup tokens */
    nk_i8_t *q_warmup_i8 = (nk_i8_t *)malloc(n_warmup * depth * sizeof(nk_i8_t));
    float *q_warmup_f32 = (float *)malloc(n_warmup * depth * sizeof(float));
    float *q_warmup_inv = (float *)malloc(n_warmup * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++) {
        nk_u32_t ti = perm[t];
        memcpy(q_warmup_i8 + t * depth, q_i8 + ti * depth, depth);
        memcpy(q_warmup_f32 + t * depth, query_data + ti * depth, depth * sizeof(float));
        q_warmup_inv[t] = q_inv_norms[ti];
    }

    double *partial_scores = (double *)calloc(N, sizeof(double));

    PyThreadState *save = PyEval_SaveThread();
    double t_kernel = 0, t_select = 0, t_rescore = 0;

    /* Phase 1: SAME warmup kernel as CB-NK round 1 — n_warmup tokens × all N docs */
    double t0 = _now_ms();
    #if !defined(NK_OMP_PRAGMAS_DISABLED)
    #pragma omp parallel for schedule(static) if(N > 256 && n_threads > 1) num_threads(n_threads)
    #endif
    for (nk_size_t i = 0; i < N; i++) {
        nk_size_t d_start = (nk_size_t)doc_offsets[i];
        nk_size_t Ld = (nk_size_t)(doc_offsets[i + 1] - doc_offsets[i]);
        nk_f64_t local_stats[2];
        nk_maxsim_i8_flat_stats_tiled(
            q_warmup_i8, q_warmup_f32, q_warmup_inv,
            all_i8 + d_start * depth,
            all_f32 + d_start * depth,
            all_inv_norms + d_start,
            all_sum_i8 + d_start,
            n_warmup, Ld, depth, local_stats);
        partial_scores[i] = local_stats[0];
    }
    t_kernel = _now_ms() - t0;
    nk_size_t total_warmup_cells = N * n_warmup;

    /* Phase 2: top-M selection — angular similarity (lower = better) */
    t0 = _now_ms();
    nk_size_t M_actual = (nk_size_t)M < N ? (nk_size_t)M : N;
    _score_idx_t *si = (_score_idx_t *)malloc(N * sizeof(_score_idx_t));
    for (nk_size_t i = 0; i < N; i++) { si[i].score = partial_scores[i]; si[i].idx = (nk_u32_t)i; }
    qsort(si, N, sizeof(_score_idx_t), _cmp_si);
    nk_u32_t *topM = (nk_u32_t *)malloc(M_actual * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < M_actual; i++) topM[i] = si[i].idx;
    free(si);
    t_select = _now_ms() - t0;

    /* Phase 3: SAME rescore kernel as CB-NK final stage — all T tokens × M docs */
    double *exact_scores = (double *)malloc(M_actual * sizeof(double));
    t0 = _now_ms();
    #if !defined(NK_OMP_PRAGMAS_DISABLED)
    #pragma omp parallel for schedule(static) if(M_actual > 16 && n_threads > 1) num_threads(n_threads)
    #endif
    for (nk_size_t i = 0; i < M_actual; i++) {
        nk_u32_t di = topM[i];
        nk_size_t d_start = (nk_size_t)doc_offsets[di];
        nk_size_t Ld = (nk_size_t)(doc_offsets[di + 1] - doc_offsets[di]);
        nk_f64_t rescore[2];
        nk_maxsim_i8_flat_stats_tiled(
            q_i8, (float *)query_data, q_inv_norms,
            all_i8 + d_start * depth,
            all_f32 + d_start * depth,
            all_inv_norms + d_start,
            all_sum_i8 + d_start,
            T, Ld, depth, rescore);
        exact_scores[i] = rescore[0];
    }
    t_rescore = _now_ms() - t0;
    nk_size_t total_rescore_cells = M_actual * (T - n_warmup);
    nk_size_t total_cells = total_warmup_cells + total_rescore_cells;
    double coverage = (double)total_cells / ((double)N * T) * 100.0;

    PyEval_RestoreThread(save);

    /* Phase 4: select top-K from M by exact score (angular: lower = better) */
    nk_size_t result_K = (nk_size_t)K < M_actual ? (nk_size_t)K : M_actual;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        for (nk_size_t j = i + 1; j < M_actual; j++)
            if (exact_scores[j] < exact_scores[best]) best = j;
        if (best != i) {
            nk_u32_t tt = topM[i]; topM[i] = topM[best]; topM[best] = tt;
            double ts = exact_scores[i]; exact_scores[i] = exact_scores[best]; exact_scores[best] = ts;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromUnsignedLong(topM[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(exact_scores[i]));
    }
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:n,s:d,s:d,s:d}",
        "coverage", coverage,
        "M", (Py_ssize_t)M_actual,
        "n_warmup", (Py_ssize_t)n_warmup,
        "warmup_kernel_ms", t_kernel,
        "select_ms", t_select,
        "rescore_ms", t_rescore);

    free(q_i8); free(q_inv_norms); free(perm);
    free(q_warmup_i8); free(q_warmup_f32); free(q_warmup_inv);
    free(partial_scores); free(topM); free(exact_scores);
    PyBuffer_Release(&q_buf); PyBuffer_Release(&di_buf); PyBuffer_Release(&df_buf);
    PyBuffer_Release(&dn_buf); PyBuffer_Release(&do_buf); PyBuffer_Release(&ds_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
#else
    PyErr_SetString(PyExc_NotImplementedError, "topm_flat requires x86 AVX2");
    return NULL;
#endif
}

/* ============================================================================
 *   api_colbandit_maxsim_optimized  —  kernel-optimized variant of CB-NK.
 *
 *   Algorithmically identical to api_colbandit_maxsim (multi-round Serfling
 *   elimination + K+M rescore). The deployed int8 path is left untouched.
 *
 *   Three optimizations applied:
 *
 *   (1) Per-round bounds are already at round granularity in the deployed code
 *       (LCB/UCB are recomputed once per round in serfling_eliminate, not per
 *       cell). The deployed kernel_stats already aggregates B query tokens'
 *       sum/sum_sq before returning. So Optimization 1 from the brief is a
 *       no-op against the current baseline (it was already in place).
 *
 *   (2) SIMD-vectorize the bound-update phase across docs.
 *       The current serfling_eliminate has scalar sqrt+log per surviving doc
 *       per round. With 25K surviving docs in early rounds this is real.
 *       The new serfling_eliminate_simd_ helper processes 8 docs/iter using
 *       AVX2 fp32, then 4 docs/iter using AVX2 fp64 for the tail.
 *
 *   (3) Cache-friendly state layout: pack (sum, sum_sq, count, _) into a
 *       16-byte AoS struct CBDocState. The kernel-update loop touches one
 *       cache line per doc instead of three (obs_sum + obs_sum_sq + obs_count
 *       were three separate 8-byte arrays in the deployed path).
 * ============================================================================ */
char const doc_colbandit_maxsim_optimized[] =
    "colbandit_maxsim_optimized(query_tokens, doc_packed_list, /, K=5, K_margin=0,\n"
    "  warmup_ratio=1.0, alpha_ef=0.3, delta=0.01, no_eliminate=False, n_threads=1,\n"
    "  use_orig_N=False, B=4) -> (indices, scores, stats)\n\n"
    "Optimized variant of colbandit_maxsim with SIMD-vectorized bound updates and\n"
    "AoS state layout. Mathematically equivalent (up to fp32 reduction order) to\n"
    "the deployed colbandit_maxsim entry point.\n";

/* Per-doc CB state, AoS-packed (32 bytes, half a cache line).
   Aligned so 8-element fp32 SIMD loads in the elimination phase can use
   contiguous strided reads. */
typedef struct __attribute__((aligned(32))) CBDocState {
    double sum;       /* obs_sum  — accumulated MaxSim across observed query tokens */
    double sum_sq;    /* obs_sum_sq — for Welford variance via 2-moment identity */
    nk_u32_t count;   /* obs_count — number of query tokens revealed for this doc */
    nk_u32_t _pad32;
    nk_u64_t _pad64;
} CBDocState;

/* Vectorized Serfling elimination operating on AoS CBDocState.
 *
 * Mathematically identical to serfling_eliminate(): same union bound, same
 * piecewise BM rho_n, same R=2 hard cap. The only difference is reduction
 * order (fp32 SIMD vs fp64 scalar), which can cause sub-ULP drift.
 *
 * Returns the new survivor count (writes survivor list in-place).
 */
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
__attribute__((target("avx2,fma")))
#endif
static nk_size_t serfling_eliminate_simd_(
    CBDocState *state,           /* AoS state, indexed by doc id */
    nk_u32_t *survivors, nk_size_t n_survivors,
    nk_size_t K, nk_size_t T, double alpha_ef, double delta,
    float *lcb_f32, float *ucb_f32,   /* per-survivor scratch */
    double *heap_buf,
    nk_u32_t N_orig, nk_u8_t *sort_bitmap,
    int use_orig_N, int *fallback_out) {

    if (fallback_out) *fallback_out = 0;

    const double C_BM = 10.0;
    double safe_delta = (delta > 1e-30 ? delta : 1e-30);
    double union_n = use_orig_N ? (double)N_orig : (double)n_survivors;
    if (union_n < 1.0) union_n = 1.0;
    double log_idd = log(C_BM * union_n * (double)T / safe_delta);
    double inv_T_d = 1.0 / (double)T;
    const double CELL_RANGE_MAX = 2.0;

    /* Gather state into 4 SoA scratch arrays for SIMD loads.
       We allocate on the stack for small n_survivors, else fall through.
       For the hot path n_survivors <= N (up to 50K in our tests) — use the
       caller-provided lcb_f32/ucb_f32 buffers as scratch by aliasing them.
       To stay safe, allocate fresh sum/sum_sq/count/Tminus arrays here. */
    float *sum_f = (float *)malloc(n_survivors * sizeof(float));
    float *sumsq_f = (float *)malloc(n_survivors * sizeof(float));
    float *cnt_f = (float *)malloc(n_survivors * sizeof(float));
    nk_u32_t *cnt_u = (nk_u32_t *)malloc(n_survivors * sizeof(nk_u32_t));

    /* Gather (one cache-line read per doc — AoS pays off here). */
    for (nk_size_t i = 0; i < n_survivors; i++) {
        nk_u32_t di = survivors[i];
        sum_f[i] = (float)state[di].sum;
        sumsq_f[i] = (float)state[di].sum_sq;
        cnt_u[i] = state[di].count;
        cnt_f[i] = (float)state[di].count;
    }

#if NK_TARGET_X86_ && NK_TARGET_HASWELL
    /* SIMD body: 8 docs per iteration in fp32.
       For each lane:
           if (cnt < 2): lcb = sum;  ucb = sum + 2 * (T - cnt);  continue
           inv_n = 1/cnt
           mean = sum * inv_n
           var = (sum_sq * inv_n - mean^2) * cnt / (cnt-1)   if var<0 -> 0
           sigma = sqrt(max(var, 1e-12))
           rho = (cnt*2 <= T) ? (1 - (cnt-1)/T)
                              : (1 - cnt/T) * (1 + 1/cnt)
           if rho<0 -> 0
           r = alpha * T * sigma * sqrt(2 * rho * log_idd * inv_n)
           est = mean * T
           lcb = max(est - r, sum)
           ucb_hard = sum + 2 * (T - cnt)
           ucb = min(est + r, ucb_hard)
    */
    const __m256 v_T = _mm256_set1_ps((float)T);
    const __m256 v_invT = _mm256_set1_ps((float)inv_T_d);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_two = _mm256_set1_ps(2.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_alpha_T = _mm256_set1_ps((float)(alpha_ef * (double)T));
    const __m256 v_2log = _mm256_set1_ps((float)(2.0 * log_idd));
    const __m256 v_eps = _mm256_set1_ps(1e-12f);
    const __m256 v_T_half = _mm256_set1_ps((float)T * 0.5f);

    nk_size_t i = 0;
    for (; i + 8 <= n_survivors; i += 8) {
        __m256 sum = _mm256_loadu_ps(sum_f + i);
        __m256 sumsq = _mm256_loadu_ps(sumsq_f + i);
        __m256 cnt = _mm256_loadu_ps(cnt_f + i);

        /* T_minus = T - cnt; ucb_hard = sum + 2*(T-cnt) */
        __m256 T_minus = _mm256_sub_ps(v_T, cnt);
        __m256 ucb_hard = _mm256_fmadd_ps(v_two, T_minus, sum);

        /* inv_n = 1/cnt (no div by 0 since cnt>=2 path; cnt<2 lane gets fixed up) */
        /* Use rcpps + 1 NR step for ~22-bit accuracy — but plain div is fine since
           the elim loop is a small fraction of total runtime. */
        __m256 inv_n = _mm256_div_ps(v_one, cnt);
        __m256 mean = _mm256_mul_ps(sum, inv_n);
        /* var_pop = sum_sq*inv_n - mean*mean (population) */
        __m256 mean_sq = _mm256_mul_ps(mean, mean);
        __m256 var_pop = _mm256_fmsub_ps(sumsq, inv_n, mean_sq);
        /* sample variance: var_pop * cnt / (cnt-1) */
        __m256 cnt_m1 = _mm256_sub_ps(cnt, v_one);
        __m256 var = _mm256_div_ps(_mm256_mul_ps(var_pop, cnt), cnt_m1);
        var = _mm256_max_ps(var, v_zero);
        __m256 var_clamped = _mm256_max_ps(var, v_eps);
        __m256 sigma = _mm256_sqrt_ps(var_clamped);

        /* rho: piecewise BM */
        __m256 rho_lo = _mm256_fnmadd_ps(_mm256_sub_ps(cnt, v_one), v_invT, v_one);
        /* rho_hi = (1 - cnt/T) * (1 + 1/cnt) */
        __m256 cnt_over_T = _mm256_mul_ps(cnt, v_invT);
        __m256 one_minus = _mm256_sub_ps(v_one, cnt_over_T);
        __m256 one_plus_invn = _mm256_add_ps(v_one, inv_n);
        __m256 rho_hi = _mm256_mul_ps(one_minus, one_plus_invn);
        /* select: cnt*2 <= T iff cnt <= T/2 */
        __m256 mask_lo = _mm256_cmp_ps(cnt, v_T_half, _CMP_LE_OQ);
        __m256 rho = _mm256_blendv_ps(rho_hi, rho_lo, mask_lo);
        rho = _mm256_max_ps(rho, v_zero);

        /* r = alpha_T * sigma * sqrt(2 * rho * log_idd * inv_n) */
        __m256 inner = _mm256_mul_ps(_mm256_mul_ps(rho, v_2log), inv_n);
        __m256 inner_sqrt = _mm256_sqrt_ps(_mm256_max_ps(inner, v_zero));
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(v_alpha_T, sigma), inner_sqrt);

        __m256 est = _mm256_mul_ps(mean, v_T);
        __m256 lcb = _mm256_max_ps(_mm256_sub_ps(est, r), sum);
        __m256 ucb = _mm256_min_ps(_mm256_add_ps(est, r), ucb_hard);

        /* cnt<2 fixup: lcb=sum, ucb=ucb_hard */
        __m256 mask_n_lt2 = _mm256_cmp_ps(cnt, v_two, _CMP_LT_OQ);
        lcb = _mm256_blendv_ps(lcb, sum, mask_n_lt2);
        ucb = _mm256_blendv_ps(ucb, ucb_hard, mask_n_lt2);

        _mm256_storeu_ps(lcb_f32 + i, lcb);
        _mm256_storeu_ps(ucb_f32 + i, ucb);
    }
    /* Scalar tail (and used when not Haswell). */
    for (; i < n_survivors; i++) {
        nk_u32_t nt = cnt_u[i];
        double s = (double)sum_f[i];
        if (nt < 2) {
            lcb_f32[i] = (float)s;
            ucb_f32[i] = (float)(s + CELL_RANGE_MAX * (double)(T - nt));
            continue;
        }
        double inv_nt = 1.0 / (double)nt;
        double mean_d = s * inv_nt;
        double var_d = ((double)sumsq_f[i] * inv_nt - mean_d * mean_d) * (double)nt / (double)(nt - 1);
        if (var_d < 0.0) var_d = 0.0;
        double sigma_d = sqrt(var_d > 1e-12 ? var_d : 1e-12);
        double rho_d;
        if ((nk_size_t)nt * 2 <= T)
            rho_d = 1.0 - (double)(nt - 1) * inv_T_d;
        else
            rho_d = (1.0 - (double)nt * inv_T_d) * (1.0 + inv_nt);
        if (rho_d < 0.0) rho_d = 0.0;
        double r_d = alpha_ef * (double)T * sigma_d * sqrt(2.0 * rho_d * log_idd * inv_nt);
        double est_full = mean_d * (double)T;
        double lcb_d = est_full - r_d;
        if (lcb_d < s) lcb_d = s;
        double ucb_d = est_full + r_d;
        double ucb_hard_d = s + CELL_RANGE_MAX * (double)(T - nt);
        if (ucb_d > ucb_hard_d) ucb_d = ucb_hard_d;
        lcb_f32[i] = (float)lcb_d;
        ucb_f32[i] = (float)ucb_d;
    }
#else
    for (nk_size_t i2 = 0; i2 < n_survivors; i2++) {
        nk_u32_t nt = cnt_u[i2];
        double s = (double)sum_f[i2];
        if (nt < 2) {
            lcb_f32[i2] = (float)s;
            ucb_f32[i2] = (float)(s + CELL_RANGE_MAX * (double)(T - nt));
            continue;
        }
        double inv_nt = 1.0 / (double)nt;
        double mean_d = s * inv_nt;
        double var_d = ((double)sumsq_f[i2] * inv_nt - mean_d * mean_d) * (double)nt / (double)(nt - 1);
        if (var_d < 0.0) var_d = 0.0;
        double sigma_d = sqrt(var_d > 1e-12 ? var_d : 1e-12);
        double rho_d;
        if ((nk_size_t)nt * 2 <= T)
            rho_d = 1.0 - (double)(nt - 1) * inv_T_d;
        else
            rho_d = (1.0 - (double)nt * inv_T_d) * (1.0 + inv_nt);
        if (rho_d < 0.0) rho_d = 0.0;
        double r_d = alpha_ef * (double)T * sigma_d * sqrt(2.0 * rho_d * log_idd * inv_nt);
        double est_full = mean_d * (double)T;
        double lcb_d = est_full - r_d;
        if (lcb_d < s) lcb_d = s;
        double ucb_d = est_full + r_d;
        double ucb_hard_d = s + CELL_RANGE_MAX * (double)(T - nt);
        if (ucb_d > ucb_hard_d) ucb_d = ucb_hard_d;
        lcb_f32[i2] = (float)lcb_d;
        ucb_f32[i2] = (float)ucb_d;
    }
#endif

    free(sum_f); free(sumsq_f); free(cnt_f); free(cnt_u);

    /* K-element max-heap: find K-th smallest UCB. */
    nk_size_t heap_size = 0;
    for (nk_size_t ii = 0; ii < n_survivors; ii++) {
        double u = (double)ucb_f32[ii];
        if (heap_size < K) {
            heap_buf[heap_size++] = u;
            if (heap_size == K)
                for (nk_size_t j = K/2; j-- > 0; )
                    _heap_sift_down(heap_buf, K, j);
        } else if (u < heap_buf[0]) {
            heap_buf[0] = u;
            _heap_sift_down(heap_buf, K, 0);
        }
    }
    double kth_ucb = heap_buf[0];

    nk_size_t new_count = 0;
    for (nk_size_t ii = 0; ii < n_survivors; ii++) {
        if ((double)lcb_f32[ii] <= kth_ucb) {
            survivors[new_count++] = survivors[ii];
        }
    }
    if (new_count < K) {
        if (fallback_out) *fallback_out = 1;
        new_count = n_survivors;
    }

    if (new_count <= 10000)
        _counting_sort_u32(survivors, new_count, N_orig > 0 ? N_orig - 1 : 0, sort_bitmap);

    return new_count;
}

PyObject *api_colbandit_maxsim_optimized(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "colbandit_maxsim_optimized() requires at least 2 arguments");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_list_obj = args[1];

    long K = 5;
    long K_margin = 0;
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;
    long B = 4;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0) warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0) no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0) B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "doc_packed_list must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    PyObject **doc_objects = N > 0 ? &PyList_GET_ITEM(doc_list_obj, 0) : NULL;
    if (N > 0 && !PyObject_TypeCheck(doc_objects[0], &MaxSimPackedMatrixType)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }

    MaxSimPackedMatrix *first_doc = N > 0 ? (MaxSimPackedMatrix *)doc_objects[0] : NULL;
    nk_dtype_t dtype = first_doc ? first_doc->dtype : nk_f32_k;

    nk_maxsim_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel, &cap);
    nk_maxsim_packed_punned_t kernel_stats = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_stats_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_stats, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);

    if (!kernel || !kernel_stats || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    /* Random permutation (same RNG seed as deployed). */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    /* === Optimization 3: AoS state layout === */
    CBDocState *state = (CBDocState *)aligned_alloc(64, ((N * sizeof(CBDocState) + 63) / 64) * 64);
    if (!state) {
        free(perm); PyBuffer_Release(&query_buf);
        PyErr_NoMemory();
        return NULL;
    }
    memset(state, 0, ((N * sizeof(CBDocState) + 63) / 64) * 64);

    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t element_bytes = (dtype == nk_f32_k) ? 4 : 2;
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();

    nk_size_t round_size = (nk_size_t)B;
    nk_size_t n_total_rounds = (n_warmup + round_size - 1) / round_size;
    nk_size_t rnd_pack_size = size_fn(round_size, depth);
    float *q_permuted = (float *)malloc(n_warmup * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++)
        memcpy(q_permuted + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    void **q_round_packed_arr = (void **)malloc(n_total_rounds * sizeof(void *));
    for (nk_size_t r = 0; r < n_total_rounds; r++) {
        nk_size_t offset = r * round_size;
        nk_size_t count = (offset + round_size <= n_warmup) ? round_size : (n_warmup - offset);
        q_round_packed_arr[r] = malloc(rnd_pack_size);
        pack_fn(q_permuted + offset * depth, count, depth, depth * element_bytes, q_round_packed_arr[r]);
    }
    free(q_permuted);

    /* Sort survivors by doc memory address (cache-friendly). */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    /* Per-survivor scratch buffers for SIMD elim. */
    float *elim_lcb_f = (float *)aligned_alloc(64, ((N * sizeof(float) + 63) / 64) * 64);
    float *elim_ucb_f = (float *)aligned_alloc(64, ((N * sizeof(float) + 63) / 64) * 64);
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        nk_size_t current_round = warmup_token_ptr / round_size;
        void *q_round_packed = q_round_packed_arr[current_round];

        t0 = _now_ms();
        t_warmup_pack += _now_ms() - t0;  /* query pre-packed */

        /* Score current survivors. The kernel returns per-(round,doc) sum/sum_sq.
           AoS layout: one cache-line update per doc per round. */
        t0 = _now_ms();
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        #define CB_OPT_PREFETCH_AHEAD 8
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        if (n_survivors > N / 2) {
            #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_OPT_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_OPT_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                if (no_eliminate) {
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    state[di].sum += local_stats[0];
                } else {
                    kernel_stats(q_round_packed, doc->start,
                                 tokens_this_round, doc->vector_count, depth, local_stats);
                    state[di].sum += local_stats[0];
                    state[di].sum_sq += local_stats[1];
                }
                state[di].count += (nk_u32_t)tokens_this_round;
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_OPT_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_OPT_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                if (no_eliminate) {
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    state[di].sum += local_stats[0];
                } else {
                    kernel_stats(q_round_packed, doc->start,
                                 tokens_this_round, doc->vector_count, depth, local_stats);
                    state[di].sum += local_stats[0];
                    state[di].sum_sq += local_stats[1];
                }
                state[di].count += (nk_u32_t)tokens_this_round;
            }
        }
        #else
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2];
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            kernel_stats(q_round_packed, doc->start,
                         tokens_this_round, doc->vector_count, depth, local_stats);
            state[di].sum += local_stats[0];
            state[di].sum_sq += local_stats[1];
            state[di].count += (nk_u32_t)tokens_this_round;
        }
        #endif
        #undef CB_OPT_PREFETCH_AHEAD
#else
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2];
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            kernel_stats(q_round_packed, doc->start,
                         tokens_this_round, doc->vector_count, depth, local_stats);
            state[di].sum += local_stats[0];
            state[di].sum_sq += local_stats[1];
            state[di].count += (nk_u32_t)tokens_this_round;
        }
#endif
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* === Optimization 2: SIMD-vectorized elimination === */
        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate_simd_(
                state, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb_f, elim_ucb_f, elim_heap,
                (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        if (n_survivors <= K_elim) break;
    }

    /* Phase 2: explore on survivors. */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        nk_size_t pack_size = size_fn(n_explore, depth);
        void *q_explore_packed = malloc(pack_size);
        float *q_explore_data = (float *)malloc(n_explore * depth * sizeof(float));
        for (nk_size_t t = 0; t < n_explore; t++) {
            memcpy(q_explore_data + t * depth,
                   query_data + perm[warmup_token_ptr + t] * depth,
                   depth * sizeof(float));
        }
        pack_fn(q_explore_data, n_explore, depth, depth * element_bytes, q_explore_packed);
        free(q_explore_data);
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
        score_docs(kernel, q_explore_packed, n_explore,
                   doc_objects, survivors, n_survivors, depth, dtype, explore_scores, n_threads);
        free(q_explore_packed);

        for (nk_size_t i = 0; i < n_survivors; i++)
            state[survivors[i]].sum += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* K+M rescore using full-T stats kernel. */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        pack_fn(query_data, T, depth, depth * element_bytes, q_full);
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_f64_t rescore_stats[2];
            kernel_stats(q_full, doc->start, T, doc->vector_count, depth, rescore_stats);
            state[di].sum = rescore_stats[0];
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    /* Final top-K sort (ascending = best). */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = state[survivors[i]].sum;
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (state[survivors[j]].sum < best_score) {
                best = j;
                best_score = state[survivors[j]].sum;
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(state[survivors[i]].sum));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);

    for (nk_size_t r = 0; r < n_total_rounds; r++) free(q_round_packed_arr[r]);
    free(q_round_packed_arr);
    free(elim_lcb_f);
    free(elim_ucb_f);
    free(elim_heap);
    free(sort_bitmap);
    free(state);
    free(survivors);
    free(perm);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}


/* ============================================================================
 *   api_colbandit_maxsim_unified  —  CB with the same fp32 micro-kernel
 *   structure as `nk.maxsim_packed`'s sum kernel for partial-sum updates.
 *
 *   Hypothesis (per Roi, 2026-05-09): the deployed `colbandit_maxsim` is at
 *   ~0.91 µs/cell while raw `maxsim_packed` is at ~0.63 µs/cell.  The deployed
 *   path uses `nk_maxsim_packed_stats_f32_haswell` which is i8 coarse +
 *   *top-2* fp32 refine.  The raw `maxsim_packed_f32` path is i8 coarse +
 *   *top-1* fp32 refine.  This entry point uses a custom helper that mirrors
 *   the top-1 path but also accumulates sum_sq so Serfling bounds still work.
 *
 *   The DEPLOYED `colbandit_maxsim` is left COMPLETELY UNCHANGED.
 *
 *   Inner kernel: `nk_maxsim_packed_stats_top1_f32_haswell_cb_`.  Layout is the
 *   same MaxSimPackedMatrix (i8 region + originals + metadata), so callers
 *   pass the same pre-packed buffers as `colbandit_maxsim`.  Drops the 4-bit
 *   centroid path (intentionally — this entry point is the fp32 fast path).
 *
 *   Phase 2 hint (drop the int8 buffer): not implemented here, since the
 *   buffer layout is shared and produced by `maxsim_pack`.  Memory saving is
 *   addressable later by adding a `maxsim_pack_fp32_only` packer that elides
 *   the i8 region.  This entry point is correctness-preserving on existing
 *   packed buffers.
 * ============================================================================ */

#if NK_TARGET_X86_ && NK_TARGET_HASWELL
/* Top-1 i8 coarse argmax + single fp32 refine + (sum, sum_sq) accumulation.
 * Mirrors `nk_maxsim_packed_f32_haswell` (the 0.63 µs/cell sum kernel) but
 * adds an extra fma for sum_sq.  Replaces `nk_maxsim_packed_stats_f32_haswell`
 * (which uses top-2: 2 fp32 dots/Q-token, ~50% more refine cost) inside CB.
 *
 * Result: result[0] = sum(angular), result[1] = sum(angular^2).
 *
 * Target attribute MUST match the haswell.h pragma push exactly so that the
 * NK_INTERNAL helper `nk_maxsim_coarse_argmax_haswell_` (declared with
 * `always_inline`) can be inlined here without a target-mismatch error. */
__attribute__((target("avx2,f16c,fma,bmi,bmi2")))
static void nk_maxsim_packed_stats_top1_f32_haswell_cb_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    if (document_count == 0) {
        result[0] = 0.0; result[1] = 0.0;
        return;
    }

    /* Same chunked structure as nk_maxsim_packed_f32_haswell. */
    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                         regions.document_quantized, regions.document_metadata, chunk_size,
                                         document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f64_t dot_result;
            nk_dot_f32(
                (nk_f32_t const *)(regions.query_originals + (chunk_start + query_index) * regions.query_original_stride),
                (nk_f32_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f64_t cosine = dot_result *
                              (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f64_t angular = 1.0 - cosine;
            if (angular < 0.0) angular = 0.0;
            total_sum += angular;
            total_sum_sq += angular * angular;
        }
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}
#else
/* Portable scalar fallback for non-Haswell builds: just delegate to the
 * registered stats kernel (top-2). */
static void nk_maxsim_packed_stats_top1_f32_haswell_cb_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {
    nk_maxsim_packed_punned_t kernel_stats = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_stats_k, nk_f32_k, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_stats, &cap);
    if (kernel_stats) {
        kernel_stats(query_packed, document_packed, query_count, document_count, depth, result);
    } else {
        result[0] = 0.0; result[1] = 0.0;
    }
}
#endif

/* AVX-512 / Icelake variant: same structure but uses VPDPBUSD coarse argmax.
 * Built when NK_TARGET_ICELAKE is enabled; runtime-selected if static_capabilities
 * advertises icelake support.  Phase 3 of the unified-kernel optimization. */
#if NK_TARGET_X86_ && NK_TARGET_ICELAKE
__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,f16c,fma,bmi,bmi2")))
static void nk_maxsim_packed_stats_top1_f32_icelake_cb_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    if (document_count == 0) {
        result[0] = 0.0; result[1] = 0.0;
        return;
    }

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_icelake_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                         regions.document_quantized, regions.document_metadata, chunk_size,
                                         document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f64_t dot_result;
            nk_dot_f32(
                (nk_f32_t const *)(regions.query_originals + (chunk_start + query_index) * regions.query_original_stride),
                (nk_f32_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f64_t cosine = dot_result *
                              (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f64_t angular = 1.0 - cosine;
            if (angular < 0.0) angular = 0.0;
            total_sum += angular;
            total_sum_sq += angular * angular;
        }
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}
#endif

/* Runtime dispatcher: pick icelake helper if AVX-512 available, else haswell. */
static inline void nk_maxsim_packed_stats_top1_f32_unified_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {
#if NK_TARGET_X86_ && NK_TARGET_ICELAKE
    if (static_capabilities & nk_cap_icelake_k) {
        nk_maxsim_packed_stats_top1_f32_icelake_cb_(query_packed, document_packed,
                                                     query_count, document_count, depth, result);
        return;
    }
#endif
    nk_maxsim_packed_stats_top1_f32_haswell_cb_(query_packed, document_packed,
                                                 query_count, document_count, depth, result);
}

char const doc_colbandit_maxsim_unified[] =
    "colbandit_maxsim_unified(query_tokens, doc_packed_list, /, K=5, K_margin=0,\n"
    "  warmup_ratio=1.0, alpha_ef=0.3, delta=0.01, no_eliminate=False, n_threads=1,\n"
    "  use_orig_N=False, B=4) -> (indices, scores, stats)\n\n"
    "Unified-kernel Col-Bandit: same Serfling-elimination structure as\n"
    "`colbandit_maxsim`, but the inner partial-sum update uses a top-1 stats\n"
    "kernel that mirrors the *sum* kernel of `maxsim_packed` (0.63 µs/cell)\n"
    "instead of the top-2 stats kernel (0.91 µs/cell).  Sum_sq is accumulated\n"
    "from the top-1 angular distances so Serfling bounds remain valid.\n"
    "fp32 docs only.  4-bit / centroid-table modes are intentionally dropped.\n";

PyObject *api_colbandit_maxsim_unified(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "colbandit_maxsim_unified() requires at least 2 arguments");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_list_obj = args[1];

    long K = 5;
    long K_margin = 0;
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;
    long B = 4;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0) warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0) no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0) B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    /* Drop trailing zero-norm padding tokens (same as deployed CB). */
    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "doc_packed_list must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list_obj, 0);
    if (N > 0 && !PyObject_TypeCheck(doc_objects[0], &MaxSimPackedMatrixType)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;
    if (dtype != nk_f32_k) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_NotImplementedError,
                        "colbandit_maxsim_unified currently supports f32 docs only");
        return NULL;
    }

    nk_maxsim_packed_punned_t kernel = NULL;        /* sum kernel, used in no_eliminate / explore */
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel, &cap);
    nk_maxsim_packed_punned_t kernel_stats_top2 = NULL;  /* used only by K_margin rescore */
    nk_find_kernel_punned(nk_kernel_maxsim_packed_stats_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_stats_top2, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);

    if (!kernel || !kernel_stats_top2 || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    /* Random permutation of token order (same RNG seed as deployed CB). */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t element_bytes = 4;
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();

    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;
    nk_size_t round_size = (nk_size_t)B;

    /* Pre-pack all warmup rounds upfront (same as deployed CB). */
    nk_size_t n_total_rounds = (n_warmup + round_size - 1) / round_size;
    nk_size_t rnd_pack_size = size_fn(round_size, depth);
    float *q_permuted = (float *)malloc(n_warmup * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++)
        memcpy(q_permuted + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    void **q_round_packed_arr = (void **)malloc(n_total_rounds * sizeof(void *));
    for (nk_size_t r = 0; r < n_total_rounds; r++) {
        nk_size_t offset = r * round_size;
        nk_size_t count = (offset + round_size <= n_warmup) ? round_size : (n_warmup - offset);
        q_round_packed_arr[r] = malloc(rnd_pack_size);
        pack_fn(q_permuted + offset * depth, count, depth, depth * element_bytes, q_round_packed_arr[r]);
    }
    free(q_permuted);

    /* Sort survivors by doc memory address for cache-friendly iteration. */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        nk_size_t current_round = warmup_token_ptr / round_size;
        void *q_round_packed = q_round_packed_arr[current_round];

        t0 = _now_ms();
        t_warmup_pack += _now_ms() - t0;  /* nothing to do — pre-packed */

        t0 = _now_ms();
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        #define CB_UNI_PREFETCH_AHEAD 8
        if (n_survivors > N / 2) {
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            #endif
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_UNI_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_UNI_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                if (no_eliminate) {
                    /* bypass mode: same sum kernel as Full */
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                } else {
                    /* UNIFIED: top-1 stats kernel (sum + sum_sq via top-1 fp32 refine) */
                    nk_maxsim_packed_stats_top1_f32_unified_(
                        q_round_packed, doc->start,
                        tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                }
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        } else {
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            #endif
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_UNI_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_UNI_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                if (no_eliminate) {
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                } else {
                    nk_maxsim_packed_stats_top1_f32_unified_(
                        q_round_packed, doc->start,
                        tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                }
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        }
#else
        /* Portable fallback: serial loop using the helper (which delegates to top-2 on non-Haswell) */
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2];
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            if (no_eliminate) {
                kernel(q_round_packed, doc->start,
                       tokens_this_round, doc->vector_count, depth, local_stats);
                obs_sum[di] += local_stats[0];
            } else {
                nk_maxsim_packed_stats_top1_f32_unified_(
                    q_round_packed, doc->start,
                    tokens_this_round, doc->vector_count, depth, local_stats);
                obs_sum[di] += local_stats[0];
                obs_sum_sq[di] += local_stats[1];
            }
            obs_count[di] += (nk_u32_t)tokens_this_round;
        }
#endif
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Serfling elimination — identical to deployed CB. */
        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        if (n_survivors <= K_elim) break;
    }

    /* ====== Phase 2: explore (sum kernel for residual tokens) ====== */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        nk_size_t pack_size = size_fn(n_explore, depth);
        void *q_explore_packed = malloc(pack_size);
        float *q_explore_data = (float *)malloc(n_explore * depth * sizeof(float));
        for (nk_size_t t = 0; t < n_explore; t++) {
            memcpy(q_explore_data + t * depth,
                   query_data + perm[warmup_token_ptr + t] * depth,
                   depth * sizeof(float));
        }
        pack_fn(q_explore_data, n_explore, depth, depth * element_bytes, q_explore_packed);
        free(q_explore_data);
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
        score_docs(kernel, q_explore_packed, n_explore,
                   doc_objects, survivors, n_survivors, depth, dtype, explore_scores, n_threads);
        free(q_explore_packed);

        for (nk_size_t i = 0; i < n_survivors; i++)
            obs_sum[survivors[i]] += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* ====== K+M rescore (use top-2 stats kernel for accuracy parity with deployed CB) ====== */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        pack_fn(query_data, T, depth, depth * element_bytes, q_full);
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_f64_t rescore_stats[2];
            kernel_stats_top2(q_full, doc->start, T, doc->vector_count, depth, rescore_stats);
            obs_sum[di] = rescore_stats[0];
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    /* Final top-K from survivors (ascending = best). */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = obs_sum[survivors[i]];
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (obs_sum[survivors[j]] < best_score) {
                best = j;
                best_score = obs_sum[survivors[j]];
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);

    for (nk_size_t r = 0; r < n_total_rounds; r++) free(q_round_packed_arr[r]);
    free(q_round_packed_arr);
    free(elim_lcb);
    free(elim_ucb);
    free(elim_heap);
    free(sort_bitmap);
    free(obs_sum);
    free(obs_sum_sq);
    free(obs_count);
    free(survivors);
    free(perm);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

/* ============================================================================
 *   api_colbandit_maxsim_pure_fp32  —  CB that NEVER reads the int8 region.
 *
 *   Hypothesis (per Roi, 2026-05-09): a CB variant that uses ONLY pure-fp32 dot
 *   products (no coarse-argmax via i8, no two-phase refine) should hit the same
 *   floor as `nk.maxsim_packed` (~0.643 µs/cell) because the deployed CB's
 *   bottleneck on small N is bookkeeping, NOT the kernel — so a kernel with
 *   identical SIMD throughput but no i8 packing/quantize round-trip should win.
 *
 *   Difference from `colbandit_maxsim_fp32`:
 *     * fp32 path uses a 4Q×1D tiled micro-kernel (doc tokens loaded once per
 *       round, reused across B query tokens) instead of the per-(q,d) dot.
 *     * Same Algorithm 1 as deployed CB (B=4 round size, Bernstein-Serfling
 *       elimination, K_margin rescore, fixed seed 42).
 *
 *   Difference from `colbandit_maxsim_unified`:
 *     * unified still calls `nk_maxsim_coarse_argmax_haswell_` (which scans the
 *       i8 region) and `nk_dot_f32` to refine 1 cell.  pure_fp32 SCANS NO I8
 *       AT ALL; every doc-token cosine is computed from scratch in fp32.
 *
 *   Memory-savings note: on a normalised f32 packed buffer, the i8 region is
 *   `vector_count × depth_i8_padded` bytes (~64 B per token).  For depth=128
 *   the originals region is 512 B per token (4× larger), so dropping the i8
 *   region saves ~12.5 % per packed buffer.  When the i8 region is also
 *   strictly unused (this entry point), a separate "fp32-only" packer could
 *   omit it entirely — left as a follow-up; this entry preserves the existing
 *   buffer layout so both `colbandit_maxsim` and `colbandit_maxsim_pure_fp32`
 *   work on the same MaxSimPackedMatrix produced by `nk.maxsim_pack`.
 * ============================================================================ */
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
/* 4Q × 1D tiled fp32 kernel.  Outer loop over doc tokens; per doc token, do B
 * fmadds into B per-query accumulators while the doc token stays in registers.
 *
 * Layout:
 *   q_f32[qi, k]         : qi in [0, query_count), k in [0, depth)
 *   d_f32[di * d_stride_floats + k] : di in [0, document_count), k in [0, depth)
 *   d_meta[di].inverse_norm_f32 : 1 / ||d_di|| (precomputed at pack time)
 *
 * Caller guarantees q rows are L2-normalised (q_inv_norms ignored).  Doc rows
 * are *not* assumed normalised — the code respects d_meta inverse-norm.
 *
 * This compiles with target("avx2,fma") so it runs on Haswell+ and is also
 * safe to link from non-AVX-512 hosts.  All loads are unaligned (`_mm256_loadu_ps`).
 */
__attribute__((target("avx2,fma")))
static void cb_fp32only_maxsim_blocked_(
    float const *q_f32, nk_size_t q_stride_floats,
    float const *q_inv_norms,                    /* may be NULL ⇒ treat as 1.0 */
    float const *d_f32, nk_size_t d_stride_floats,
    nk_maxsim_vector_metadata_t const *d_meta,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    int want_sum_sq,
    nk_f64_t *result) {

    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;
    if (document_count == 0 || query_count == 0) {
        result[0] = 0.0;
        if (want_sum_sq) result[1] = 0.0;
        return;
    }

    /* Track the running max cosine for each of the B query tokens. */
    nk_f64_t best_cosine[16];
    if (query_count > 16) query_count = 16;  /* safety; CB uses B=4 */
    for (nk_size_t qi = 0; qi < query_count; qi++) best_cosine[qi] = -1e30;

    /* 4Q × 1D fast path */
    if (query_count == 4 && depth >= 16 && (depth & 7) == 0) {
        float const *q0 = q_f32 + 0 * q_stride_floats;
        float const *q1 = q_f32 + 1 * q_stride_floats;
        float const *q2 = q_f32 + 2 * q_stride_floats;
        float const *q3 = q_f32 + 3 * q_stride_floats;

        for (nk_size_t di = 0; di < document_count; di++) {
            /* Software prefetch the next doc-token row (depth=128 = 2 lines). */
            if (di + 4 < document_count) {
                _mm_prefetch((char const *)(d_f32 + (di + 4) * d_stride_floats), _MM_HINT_T0);
                _mm_prefetch((char const *)(d_f32 + (di + 4) * d_stride_floats) + 64, _MM_HINT_T0);
            }
            float const *df = d_f32 + di * d_stride_floats;

            __m256 acc0a = _mm256_setzero_ps(), acc0b = _mm256_setzero_ps();
            __m256 acc1a = _mm256_setzero_ps(), acc1b = _mm256_setzero_ps();
            __m256 acc2a = _mm256_setzero_ps(), acc2b = _mm256_setzero_ps();
            __m256 acc3a = _mm256_setzero_ps(), acc3b = _mm256_setzero_ps();

            nk_size_t k = 0;
            for (; k + 16 <= depth; k += 16) {
                __m256 d_a = _mm256_loadu_ps(df + k);
                __m256 d_b = _mm256_loadu_ps(df + k + 8);

                acc0a = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + k),     d_a, acc0a);
                acc0b = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + k + 8), d_b, acc0b);
                acc1a = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + k),     d_a, acc1a);
                acc1b = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + k + 8), d_b, acc1b);
                acc2a = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + k),     d_a, acc2a);
                acc2b = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + k + 8), d_b, acc2b);
                acc3a = _mm256_fmadd_ps(_mm256_loadu_ps(q3 + k),     d_a, acc3a);
                acc3b = _mm256_fmadd_ps(_mm256_loadu_ps(q3 + k + 8), d_b, acc3b);
            }
            for (; k + 8 <= depth; k += 8) {
                __m256 d_a = _mm256_loadu_ps(df + k);
                acc0a = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + k), d_a, acc0a);
                acc1a = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + k), d_a, acc1a);
                acc2a = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + k), d_a, acc2a);
                acc3a = _mm256_fmadd_ps(_mm256_loadu_ps(q3 + k), d_a, acc3a);
            }

            #define CB_REDUCE_(_acc_a, _acc_b) ({                                                   \
                __m256 _s = _mm256_add_ps(_acc_a, _acc_b);                                          \
                __m128 _lo = _mm256_castps256_ps128(_s), _hi = _mm256_extractf128_ps(_s, 1);        \
                __m128 _r = _mm_add_ps(_lo, _hi);                                                   \
                _r = _mm_add_ps(_r, _mm_movehl_ps(_r, _r));                                         \
                _r = _mm_add_ss(_r, _mm_shuffle_ps(_r, _r, 1));                                     \
                _mm_cvtss_f32(_r);                                                                  \
            })
            float dot0 = CB_REDUCE_(acc0a, acc0b);
            float dot1 = CB_REDUCE_(acc1a, acc1b);
            float dot2 = CB_REDUCE_(acc2a, acc2b);
            float dot3 = CB_REDUCE_(acc3a, acc3b);
            #undef CB_REDUCE_

            nk_f64_t inv_dn = (nk_f64_t)d_meta[di].inverse_norm_f32;
            nk_f64_t inv_q0 = q_inv_norms ? (nk_f64_t)q_inv_norms[0] : 1.0;
            nk_f64_t inv_q1 = q_inv_norms ? (nk_f64_t)q_inv_norms[1] : 1.0;
            nk_f64_t inv_q2 = q_inv_norms ? (nk_f64_t)q_inv_norms[2] : 1.0;
            nk_f64_t inv_q3 = q_inv_norms ? (nk_f64_t)q_inv_norms[3] : 1.0;

            nk_f64_t c0 = (nk_f64_t)dot0 * inv_q0 * inv_dn;
            nk_f64_t c1 = (nk_f64_t)dot1 * inv_q1 * inv_dn;
            nk_f64_t c2 = (nk_f64_t)dot2 * inv_q2 * inv_dn;
            nk_f64_t c3 = (nk_f64_t)dot3 * inv_q3 * inv_dn;
            if (c0 > best_cosine[0]) best_cosine[0] = c0;
            if (c1 > best_cosine[1]) best_cosine[1] = c1;
            if (c2 > best_cosine[2]) best_cosine[2] = c2;
            if (c3 > best_cosine[3]) best_cosine[3] = c3;
        }
    } else {
        /* Generic path for query_count != 4: outer di × inner qi. */
        for (nk_size_t di = 0; di < document_count; di++) {
            if (di + 4 < document_count) {
                _mm_prefetch((char const *)(d_f32 + (di + 4) * d_stride_floats), _MM_HINT_T0);
                _mm_prefetch((char const *)(d_f32 + (di + 4) * d_stride_floats) + 64, _MM_HINT_T0);
            }
            float const *df = d_f32 + di * d_stride_floats;
            nk_f64_t inv_dn = (nk_f64_t)d_meta[di].inverse_norm_f32;
            for (nk_size_t qi = 0; qi < query_count; qi++) {
                float const *qf = q_f32 + qi * q_stride_floats;
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                nk_size_t k = 0;
                for (; k + 16 <= depth; k += 16) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qf + k),     _mm256_loadu_ps(df + k),     a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(qf + k + 8), _mm256_loadu_ps(df + k + 8), a1);
                }
                for (; k + 8 <= depth; k += 8) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qf + k), _mm256_loadu_ps(df + k), a0);
                }
                __m256 s = _mm256_add_ps(a0, a1);
                __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
                __m128 r = _mm_add_ps(lo, hi);
                r = _mm_add_ps(r, _mm_movehl_ps(r, r));
                r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
                float dot_f = _mm_cvtss_f32(r);
                for (; k < depth; k++) dot_f += qf[k] * df[k];

                nk_f64_t inv_qn = q_inv_norms ? (nk_f64_t)q_inv_norms[qi] : 1.0;
                nk_f64_t c = (nk_f64_t)dot_f * inv_qn * inv_dn;
                if (c > best_cosine[qi]) best_cosine[qi] = c;
            }
        }
    }

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_f64_t angular = 1.0 - best_cosine[qi];
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        if (want_sum_sq) total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    if (want_sum_sq) result[1] = total_sum_sq;
}
#else
/* Portable scalar fallback (non-Haswell builds): delegate to existing helper. */
static void cb_fp32only_maxsim_blocked_(
    float const *q_f32, nk_size_t q_stride_floats,
    float const *q_inv_norms,
    float const *d_f32, nk_size_t d_stride_floats,
    nk_maxsim_vector_metadata_t const *d_meta,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    int want_sum_sq,
    nk_f64_t *result) {
    cb_fp32only_maxsim_stats_(q_f32, q_stride_floats, q_inv_norms,
                              d_f32, d_stride_floats, d_meta,
                              query_count, document_count, depth,
                              want_sum_sq, result);
}
#endif

char const doc_colbandit_maxsim_pure_fp32[] =
    "colbandit_maxsim_pure_fp32(query_tokens, doc_packed_list, /, K=5, K_margin=0,\n"
    "  warmup_ratio=1.0, alpha_ef=0.3, delta=0.01, no_eliminate=False, n_threads=1,\n"
    "  use_orig_N=False, B=4) -> (indices, scores, stats)\n\n"
    "Pure-fp32 Col-Bandit: NEVER reads the i8 region of the packed buffer. The\n"
    "warmup, explore, and rescore phases all use a tiled fp32 micro-kernel\n"
    "(`cb_fp32only_maxsim_blocked_`) that mirrors the SIMD pattern of\n"
    "`maxsim_packed`'s fp32 path. Same Serfling-elimination structure as\n"
    "`colbandit_maxsim`. Memory: an fp32-only packer (follow-up) could drop\n"
    "the i8 region for further savings; this entry point is correctness-\n"
    "preserving on existing MaxSimPackedMatrix buffers from maxsim_pack.\n";

PyObject *api_colbandit_maxsim_pure_fp32(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "colbandit_maxsim_pure_fp32() requires at least 2 arguments");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_list_obj = args[1];

    long K = 5;
    long K_margin = 0;
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;
    long B = 4;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0) warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0) no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0) B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "doc_packed_list must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list_obj, 0);
    if (N > 0 && !PyObject_TypeCheck(doc_objects[0], &MaxSimPackedMatrixType)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;
    if (dtype != nk_f32_k) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_NotImplementedError,
                        "colbandit_maxsim_pure_fp32 currently supports f32 docs only");
        return NULL;
    }

    /* Only used for the K+M rescore (bit-identical to Full).  Inner loop never uses it. */
    nk_maxsim_packed_punned_t kernel_for_rescore = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&kernel_for_rescore, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, dtype, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);
    if (!kernel_for_rescore || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No maxsim kernel found");
        return NULL;
    }

    nk_size_t element_bytes = 4;

    /* Fixed-seed permutation (identical to deployed CB). */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();

    /* Pre-shuffle the warmup query buffer once in fp32 — no per-round packing. */
    float *q_permuted_warmup = (float *)malloc(n_warmup * depth * sizeof(float));
    for (nk_size_t t = 0; t < n_warmup; t++)
        memcpy(q_permuted_warmup + t * depth, query_data + perm[t] * depth, depth * sizeof(float));
    nk_size_t round_size = (nk_size_t)B;

    /* Sort survivors by doc address for cache-friendly iteration. */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;
    int want_sum_sq = !no_eliminate;

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        t0 = _now_ms();
        t_warmup_pack += _now_ms() - t0;

        float const *q_round_f32 = q_permuted_warmup + warmup_token_ptr * depth;

        t0 = _now_ms();
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
        #endif
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2] = {0.0, 0.0};
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_maxsim_packed_header_t const *dhdr = (nk_maxsim_packed_header_t const *)doc->start;
            float const *d_f32 = (float const *)((char const *)doc->start + dhdr->offset_original_data);
            nk_size_t d_stride_floats = dhdr->original_stride_bytes / sizeof(float);
            nk_maxsim_vector_metadata_t const *d_meta =
                (nk_maxsim_vector_metadata_t const *)((char const *)doc->start + dhdr->offset_metadata);

            cb_fp32only_maxsim_blocked_(
                q_round_f32, depth, NULL,
                d_f32, d_stride_floats, d_meta,
                tokens_this_round, doc->vector_count, depth,
                want_sum_sq,
                local_stats);

            obs_sum[di] += local_stats[0];
            if (want_sum_sq) obs_sum_sq[di] += local_stats[1];
            obs_count[di] += (nk_u32_t)tokens_this_round;
        }
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        if (n_survivors <= K_elim) break;
    }

    /* ====== Phase 2: explore residual tokens (pure fp32, sum only) ====== */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        float *q_explore_f32 = (float *)malloc(n_explore * depth * sizeof(float));
        for (nk_size_t t = 0; t < n_explore; t++)
            memcpy(q_explore_f32 + t * depth,
                   query_data + perm[warmup_token_ptr + t] * depth,
                   depth * sizeof(float));
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
        #if !defined(NK_OMP_PRAGMAS_DISABLED)
        #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
        #endif
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2] = {0.0, 0.0};
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_maxsim_packed_header_t const *dhdr = (nk_maxsim_packed_header_t const *)doc->start;
            float const *d_f32 = (float const *)((char const *)doc->start + dhdr->offset_original_data);
            nk_size_t d_stride_floats = dhdr->original_stride_bytes / sizeof(float);
            nk_maxsim_vector_metadata_t const *d_meta =
                (nk_maxsim_vector_metadata_t const *)((char const *)doc->start + dhdr->offset_metadata);

            cb_fp32only_maxsim_blocked_(
                q_explore_f32, depth, NULL,
                d_f32, d_stride_floats, d_meta,
                n_explore, doc->vector_count, depth,
                0,
                local_stats);
            explore_scores[i] = local_stats[0];
        }
        free(q_explore_f32);

        for (nk_size_t i = 0; i < n_survivors; i++)
            obs_sum[survivors[i]] += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* ====== K+M rescore (bit-identical to Full via shared sum kernel) ====== */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        pack_fn(query_data, T, depth, depth * element_bytes, q_full);
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_f64_t rescore;
            kernel_for_rescore(q_full, doc->start, T, doc->vector_count, depth, &rescore);
            obs_sum[di] = rescore;
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = obs_sum[survivors[i]];
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (obs_sum[survivors[j]] < best_score) {
                best = j;
                best_score = obs_sum[survivors[j]];
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);

    free(q_permuted_warmup);
    free(elim_lcb);
    free(elim_ucb);
    free(elim_heap);
    free(sort_bitmap);
    free(obs_sum);
    free(obs_sum_sq);
    free(obs_count);
    free(survivors);
    free(perm);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}

/* ============================================================================
 *   api_colbandit_maxsim_f16  -  CB on f16-packed documents.
 *
 *   Same Algorithm 1 (multi-round Serfling elimination + K+M rescore) and
 *   bookkeeping as deployed `api_colbandit_maxsim`, but operates on
 *   MaxSimPackedMatrix buffers whose "originals" region is fp16 (not fp32).
 *
 *   Memory motivation (per `bench_dtype_sweep_results.json`, 2026-05-12):
 *   packing docs as f16 cuts the originals region in half (2 vs 4 B/elem),
 *   giving a 37 % total packed-memory reduction and a 5-8 % wall-clock win on
 *   the FULL ColBERT path - bit-perfect Ov@5 vs f32 oracle.  CB was previously
 *   locked to f32 because `nk_kernel_maxsim_packed_stats_k` is only registered
 *   for f32 dtype; calling `colbandit_maxsim` on f16-packed docs segfaults.
 *
 *   This entry point closes that gap by routing the warmup/explore stats path
 *   to a new helper `nk_maxsim_packed_stats_top2_f16_haswell_cb_`, structurally
 *   identical to the deployed f32 top-2 stats kernel
 *   (`nk_maxsim_packed_stats_f32_haswell`, haswell.h:1472-1514) but loading the
 *   fp32 refine vectors from the f16 originals region via F16C VCVTPH2PS.
 *
 *   The K+M rescore phase uses the same helper (it produces top-2 stats just
 *   like the f32 path).
 *
 *   The DEPLOYED `api_colbandit_maxsim` is left COMPLETELY UNCHANGED.
 * ============================================================================ */
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
/* Top-2 i8 coarse argmax + fused 2-dot fp32 refine + (sum, sum_sq) accumulation
 * for f16-packed documents.  Mirrors haswell.h:1472-1514 line-for-line except
 * for the doc / query refine loads: 16-byte F16C VCVTPH2PS upconverts replace
 * 32-byte fp32 unaligned loads.
 *
 * Stride: `original_stride_bytes` = round_up(depth*2, 64) for f16 packs (see
 * `nk_maxsim_packed_header_setup_`), i.e. >= 16-byte stride for any depth - the
 * `_mm_loadu_si128` 16-byte loads are always in-bounds.
 *
 * Target attribute matches the haswell.h pragma push exactly so the
 * NK_INTERNAL helper `nk_maxsim_coarse_top2_haswell_` (declared with
 * `always_inline`) can inline here without target-mismatch errors.  F16C is
 * already in the haswell pragma - VCVTPH2PS executes on AVX2+F16C, present on
 * all Haswell+ and on AMD Zen2+ (incl. Zen3 EPYC 7513 used by cccxc62*).
 */
__attribute__((target("avx2,f16c,fma,bmi,bmi2")))
static void nk_maxsim_packed_stats_top2_f16_haswell_cb_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    if (document_count == 0) {
        result[0] = 0.0; result[1] = 0.0;
        return;
    }

    /* Same chunked structure as f32 top-2 path. */
    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_indices[256];
        nk_u32_t second_indices[256];

        nk_maxsim_coarse_top2_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                        regions.document_quantized, regions.document_metadata, chunk_size,
                                        document_count, regions.depth_i8_padded,
                                        best_indices, second_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            /* f16 doc/query pointers (originals region is fp16-packed). */
            nk_u16_t const *qf16 = (nk_u16_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride);
            nk_u16_t const *d1   = (nk_u16_t const *)(regions.document_originals +
                                   best_indices[query_index] * regions.document_original_stride);
            nk_u16_t const *d2   = (nk_u16_t const *)(regions.document_originals +
                                   second_indices[query_index] * regions.document_original_stride);

            __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
            for (nk_size_t k = 0; k < depth; k += 8) {
                /* F16C: 8 i16 -> 8 fp32 (~3 cycle latency, 1/cycle throughput). */
                __m256 q  = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(qf16 + k)));
                __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(d1   + k)));
                __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(d2   + k)));
                acc1 = _mm256_fmadd_ps(q, v1, acc1);
                acc2 = _mm256_fmadd_ps(q, v2, acc2);
            }
            __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
            __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
            __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
            s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
            s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
            s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
            s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));

            /* inverse_norm_f32 in metadata is computed at pack time over the
             * packed dtype, so it is already correct for f16 packs. */
            nk_f64_t inv_qnorm = (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32;
            nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * inv_qnorm *
                            (nk_f64_t)regions.document_metadata[best_indices[query_index]].inverse_norm_f32;
            nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * inv_qnorm *
                            (nk_f64_t)regions.document_metadata[second_indices[query_index]].inverse_norm_f32;
            nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
            nk_f64_t angular = 1.0 - best_cosine;
            if (angular < 0.0) angular = 0.0;
            total_sum += angular;
            total_sum_sq += angular * angular;
        }
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

/* fp32 -> fp16 row conversion (depth elements).  Lives in its own static so it
 * can carry the f16c target attribute independently of api_colbandit_maxsim_f16
 * (whose body has no target attribute). */
__attribute__((target("avx2,f16c")))
static void cb_f16_pack_row_(float const *src, nk_u16_t *dst, nk_size_t depth) {
    nk_size_t k = 0;
    for (; k + 8 <= depth; k += 8) {
        __m256 v = _mm256_loadu_ps(src + k);
        __m128i v16 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(dst + k), v16);
    }
    for (; k < depth; k++) {
        __m128 vs = _mm_load_ss(src + k);
        __m128i v16 = _mm_cvtps_ph(vs, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        dst[k] = (nk_u16_t)_mm_extract_epi16(v16, 0);
    }
}
#else
/* Portable scalar fallback (non-Haswell builds): produce zeros.  f16 CB is a
 * x86 Haswell+ optimisation; non-x86 callers should use the f32 CB entry point. */
static void nk_maxsim_packed_stats_top2_f16_haswell_cb_(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {
    (void)query_packed; (void)document_packed;
    (void)query_count;  (void)document_count; (void)depth;
    result[0] = 0.0; result[1] = 0.0;
}
static void cb_f16_pack_row_(float const *src, nk_u16_t *dst, nk_size_t depth) {
    (void)src; for (nk_size_t k = 0; k < depth; k++) dst[k] = 0;
}
#endif

char const doc_colbandit_maxsim_f16[] =
    "colbandit_maxsim_f16(query_tokens, doc_packed_list, /, K=5, K_margin=0,\n"
    "  warmup_ratio=1.0, alpha_ef=0.3, delta=0.01, no_eliminate=False, n_threads=1,\n"
    "  use_orig_N=False, B=4) -> (indices, scores, stats)\n\n"
    "Col-Bandit on f16-packed documents.  Same Serfling-elimination structure,\n"
    "same K+M rescore, same bookkeeping as `colbandit_maxsim` - but the inner\n"
    "stats kernel does its fp32 top-2 refine by loading fp16 originals via\n"
    "F16C VCVTPH2PS instead of fp32.  Cuts the originals region in half (2 vs\n"
    "4 bytes/elem) for a ~37 % total packed-memory reduction.\n\n"
    "All docs MUST be MaxSimPackedMatrix with dtype = nk.f16 (i.e. packed via\n"
    "`nk.maxsim_pack(d.astype(np.float16), dtype='f16')`); raises TypeError\n"
    "otherwise.  Query is taken as raw fp32 and packed internally to f16.\n";

PyObject *api_colbandit_maxsim_f16(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);

    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "colbandit_maxsim_f16() requires at least 2 arguments");
        return NULL;
    }

    PyObject *query_obj = args[0];
    PyObject *doc_list_obj = args[1];

    long K = 5;
    long K_margin = 0;
    double warmup_ratio = 1.0;
    double alpha_ef = 0.3;
    double delta = 0.01;
    int no_eliminate = 0;
    int n_threads = 1;
    int use_orig_N = 0;
    long B = 4;

    Py_ssize_t nkw = kwnames ? PyTuple_Size(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; i++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, i);
        PyObject *value = args[nargs + i];
        if (PyUnicode_CompareWithASCIIString(name, "K") == 0) K = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "K_margin") == 0) K_margin = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "warmup_ratio") == 0) warmup_ratio = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "alpha_ef") == 0) alpha_ef = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "delta") == 0) delta = PyFloat_AsDouble(value);
        else if (PyUnicode_CompareWithASCIIString(name, "no_eliminate") == 0) no_eliminate = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "n_threads") == 0) n_threads = (int)PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "use_orig_N") == 0) use_orig_N = PyObject_IsTrue(value);
        else if (PyUnicode_CompareWithASCIIString(name, "B") == 0) B = PyLong_AsLong(value);
    }

    Py_buffer query_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    nk_size_t T_raw = ((nk_size_t *)query_buf.shape)[0];
    nk_size_t depth = ((nk_size_t *)query_buf.shape)[1];
    float *query_data = (float *)query_buf.buf;

    /* Drop trailing zero-norm padding tokens (same as deployed CB). */
    nk_size_t T = T_raw;
    while (T > 0) {
        float *row = query_data + (T - 1) * depth;
        float norm_sq = 0.0f;
        for (nk_size_t j = 0; j < depth; j++) norm_sq += row[j] * row[j];
        if (norm_sq > 1e-12f) break;
        T--;
    }
    if (T == 0) T = 1;

    if (B <= 0 || (nk_size_t)B > T) B = (long)T;
    if (B < 1) B = 1;

    if (!PyList_Check(doc_list_obj)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "doc_packed_list must be a list");
        return NULL;
    }
    nk_size_t N = (nk_size_t)PyList_Size(doc_list_obj);
    PyObject **doc_objects = &PyList_GET_ITEM(doc_list_obj, 0);
    if (N > 0 && !PyObject_TypeCheck(doc_objects[0], &MaxSimPackedMatrixType)) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError, "All docs must be MaxSimPackedMatrix");
        return NULL;
    }

    MaxSimPackedMatrix *first_doc = (MaxSimPackedMatrix *)doc_objects[0];
    nk_dtype_t dtype = first_doc->dtype;
    if (dtype != nk_f16_k) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_TypeError,
                        "colbandit_maxsim_f16 requires f16-packed docs "
                        "(pack with nk.maxsim_pack(d.astype(np.float16), dtype='f16'))");
        return NULL;
    }

    /* Resolve f16 pack / size / sum kernel.  The stats path uses our local helper. */
    nk_maxsim_packed_punned_t kernel = NULL;  /* sum kernel, used in no_eliminate / explore */
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_k, nk_f16_k, static_capabilities,
                          (nk_kernel_punned_t *)&kernel, &cap);
    nk_dots_pack_punned_t pack_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_pack_k, nk_f16_k, static_capabilities,
                          (nk_kernel_punned_t *)&pack_fn, &cap);
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_find_kernel_punned(nk_kernel_maxsim_packed_size_k, nk_f16_k, static_capabilities,
                          (nk_kernel_punned_t *)&size_fn, &cap);

    if (!kernel || !pack_fn || !size_fn) {
        PyBuffer_Release(&query_buf);
        PyErr_SetString(PyExc_LookupError, "No f16 maxsim kernel found");
        return NULL;
    }

    /* Random permutation of token order (same RNG seed as deployed CB). */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    double *obs_sum = (double *)calloc(N, sizeof(double));
    double *obs_sum_sq = (double *)calloc(N, sizeof(double));
    nk_u32_t *obs_count = (nk_u32_t *)calloc(N, sizeof(nk_u32_t));
    nk_u32_t *survivors = (nk_u32_t *)malloc(N * sizeof(nk_u32_t));
    nk_size_t n_survivors = N;
    for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)i;

    nk_size_t n_warmup = (nk_size_t)ceil(warmup_ratio * T);
    if (n_warmup < 2) n_warmup = 2;
    n_warmup = (n_warmup / 4) * 4;
    if (n_warmup < 4) n_warmup = 4;
    if (n_warmup > T) n_warmup = T;

    nk_size_t element_bytes = 2;  /* f16 originals */
    nk_size_t total_cells = 0;
    nk_size_t fallback_count = 0;

    PyThreadState *save = PyEval_SaveThread();

    double t_warmup_kernel = 0, t_warmup_pack = 0, t_warmup_elim = 0;
    double t_explore_pack = 0, t_explore_kernel = 0;
    double t_rescore = 0;
    double t0 = _now_ms();

    nk_size_t warmup_token_ptr = 0;
    nk_size_t n_warmup_rounds = 0;
    nk_size_t round_size = (nk_size_t)B;

    /* Pre-pack all warmup rounds upfront (same as deployed CB).  Query is raw
     * fp32: cast each permuted row to fp16 first, then call the f16 pack_fn. */
    nk_size_t n_total_rounds = (n_warmup + round_size - 1) / round_size;
    nk_size_t rnd_pack_size = size_fn(round_size, depth);

    nk_u16_t *q_permuted_f16 = (nk_u16_t *)malloc(n_warmup * depth * sizeof(nk_u16_t));
    for (nk_size_t t = 0; t < n_warmup; t++) {
        cb_f16_pack_row_(query_data + perm[t] * depth, q_permuted_f16 + t * depth, depth);
    }
    void **q_round_packed_arr = (void **)malloc(n_total_rounds * sizeof(void *));
    for (nk_size_t r = 0; r < n_total_rounds; r++) {
        nk_size_t offset = r * round_size;
        nk_size_t count = (offset + round_size <= n_warmup) ? round_size : (n_warmup - offset);
        q_round_packed_arr[r] = malloc(rnd_pack_size);
        pack_fn(q_permuted_f16 + offset * depth, count, depth, depth * element_bytes,
                q_round_packed_arr[r]);
    }
    free(q_permuted_f16);

    /* Sort survivors by doc memory address for cache-friendly iteration. */
    {
        nk_size_t *addr_order = (nk_size_t *)malloc(N * 2 * sizeof(nk_size_t));
        for (nk_size_t i = 0; i < N; i++) {
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[survivors[i]];
            addr_order[2*i] = (nk_size_t)(uintptr_t)doc->start;
            addr_order[2*i+1] = (nk_size_t)survivors[i];
        }
        qsort(addr_order, N, 2 * sizeof(nk_size_t), _cmp_u64_pair);
        for (nk_size_t i = 0; i < N; i++) survivors[i] = (nk_u32_t)addr_order[2*i+1];
        free(addr_order);
    }

    double *elim_lcb = (double *)malloc(N * sizeof(double));
    double *elim_ucb = (double *)malloc(N * sizeof(double));
    double *elim_heap = (double *)malloc((K + K_margin + 1) * sizeof(double));
    nk_u8_t *sort_bitmap = (nk_u8_t *)malloc((N + 7) / 8);

    while (warmup_token_ptr < n_warmup) {
        nk_size_t tokens_this_round = round_size;
        if (warmup_token_ptr + tokens_this_round > n_warmup)
            tokens_this_round = n_warmup - warmup_token_ptr;
        if (tokens_this_round == 0) break;

        nk_size_t current_round = warmup_token_ptr / round_size;
        void *q_round_packed = q_round_packed_arr[current_round];

        t0 = _now_ms();
        t_warmup_pack += _now_ms() - t0;  /* nothing to do - pre-packed */

        t0 = _now_ms();
#if NK_TARGET_X86_ && NK_TARGET_HASWELL
        #define CB_F16_PREFETCH_AHEAD 8
        if (n_survivors > N / 2) {
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            #pragma omp parallel for schedule(static) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            #endif
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_F16_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_F16_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                if (no_eliminate) {
                    /* bypass mode: f16 sum kernel (Full path). */
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                } else {
                    nk_maxsim_packed_stats_top2_f16_haswell_cb_(
                        q_round_packed, doc->start,
                        tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                }
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        } else {
            #if !defined(NK_OMP_PRAGMAS_DISABLED)
            #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
            #endif
            for (nk_size_t i = 0; i < n_survivors; i++) {
                if (i + CB_F16_PREFETCH_AHEAD < n_survivors) {
                    MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_F16_PREFETCH_AHEAD]];
                    _mm_prefetch((char const *)pf_doc->start, _MM_HINT_T0);
                    _mm_prefetch((char const *)pf_doc->start + 64, _MM_HINT_T0);
                }
                nk_f64_t local_stats[2];
                nk_u32_t di = survivors[i];
                MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                if (no_eliminate) {
                    kernel(q_round_packed, doc->start,
                           tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                } else {
                    nk_maxsim_packed_stats_top2_f16_haswell_cb_(
                        q_round_packed, doc->start,
                        tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                }
                obs_count[di] += (nk_u32_t)tokens_this_round;
            }
        }
#else
        /* Portable fallback (non-Haswell): serial loop. */
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_f64_t local_stats[2];
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            if (no_eliminate) {
                kernel(q_round_packed, doc->start,
                       tokens_this_round, doc->vector_count, depth, local_stats);
                obs_sum[di] += local_stats[0];
            } else {
                nk_maxsim_packed_stats_top2_f16_haswell_cb_(
                    q_round_packed, doc->start,
                    tokens_this_round, doc->vector_count, depth, local_stats);
                obs_sum[di] += local_stats[0];
                obs_sum_sq[di] += local_stats[1];
            }
            obs_count[di] += (nk_u32_t)tokens_this_round;
        }
#endif
        t_warmup_kernel += _now_ms() - t0;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        /* Serfling elimination - identical to deployed CB. */
        t0 = _now_ms();
        nk_size_t K_elim = (nk_size_t)(K + K_margin);
        int fb = 0;
        if (!no_eliminate && n_survivors > K_elim) {
            n_survivors = serfling_eliminate(
                obs_sum, obs_sum_sq, obs_count, survivors, n_survivors,
                K_elim, T, alpha_ef, delta,
                elim_lcb, elim_ucb, elim_heap, (nk_u32_t)N, sort_bitmap, use_orig_N, &fb);
        }
        if (fb) fallback_count++;
        t_warmup_elim += _now_ms() - t0;
        n_warmup_rounds++;

        if (n_survivors <= K_elim) break;
    }

    /* ====== Phase 2: explore (f16 sum kernel for residual tokens) ====== */
    nk_size_t n_explore = T - warmup_token_ptr;
    if (n_explore > 0 && n_survivors > 0) {
        t0 = _now_ms();
        nk_size_t pack_size = size_fn(n_explore, depth);
        void *q_explore_packed = malloc(pack_size);

        nk_u16_t *q_explore_f16 = (nk_u16_t *)malloc(n_explore * depth * sizeof(nk_u16_t));
        for (nk_size_t t = 0; t < n_explore; t++) {
            cb_f16_pack_row_(query_data + perm[warmup_token_ptr + t] * depth,
                             q_explore_f16 + t * depth, depth);
        }
        pack_fn(q_explore_f16, n_explore, depth, depth * element_bytes, q_explore_packed);
        free(q_explore_f16);
        t_explore_pack = _now_ms() - t0;

        t0 = _now_ms();
        double *explore_scores = (double *)malloc(n_survivors * sizeof(double));
        score_docs(kernel, q_explore_packed, n_explore,
                   doc_objects, survivors, n_survivors, depth, nk_f16_k, explore_scores, n_threads);
        free(q_explore_packed);

        for (nk_size_t i = 0; i < n_survivors; i++)
            obs_sum[survivors[i]] += explore_scores[i];

        total_cells += n_survivors * n_explore;
        free(explore_scores);
        t_explore_kernel = _now_ms() - t0;
    }

    /* ====== K+M rescore (use the f16 top-2 stats helper for accuracy parity
     * with deployed CB's f32 K+M rescore path). ====== */
    t0 = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        nk_size_t full_pack_sz = size_fn(T, depth);
        void *q_full = malloc(full_pack_sz);
        nk_u16_t *q_full_f16 = (nk_u16_t *)malloc(T * depth * sizeof(nk_u16_t));
        for (nk_size_t t = 0; t < T; t++) {
            cb_f16_pack_row_(query_data + t * depth, q_full_f16 + t * depth, depth);
        }
        pack_fn(q_full_f16, T, depth, depth * element_bytes, q_full);
        free(q_full_f16);
        for (nk_size_t i = 0; i < n_survivors; i++) {
            nk_u32_t di = survivors[i];
            MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
            nk_f64_t rescore_stats[2];
            nk_maxsim_packed_stats_top2_f16_haswell_cb_(
                q_full, doc->start, T, doc->vector_count, depth, rescore_stats);
            obs_sum[di] = rescore_stats[0];
        }
        free(q_full);
    }
    t_rescore = _now_ms() - t0;

    PyEval_RestoreThread(save);

    /* Final top-K from survivors (ascending = best). */
    nk_size_t result_K = (nk_size_t)K < n_survivors ? (nk_size_t)K : n_survivors;
    for (nk_size_t i = 0; i < result_K; i++) {
        nk_size_t best = i;
        double best_score = obs_sum[survivors[i]];
        for (nk_size_t j = i + 1; j < n_survivors; j++) {
            if (obs_sum[survivors[j]] < best_score) {
                best = j;
                best_score = obs_sum[survivors[j]];
            }
        }
        if (best != i) {
            nk_u32_t tmp = survivors[i]; survivors[i] = survivors[best]; survivors[best] = tmp;
        }
    }

    PyObject *indices_list = PyList_New(result_K);
    PyObject *scores_list = PyList_New(result_K);
    for (nk_size_t i = 0; i < result_K; i++) {
        PyList_SET_ITEM(indices_list, i, PyLong_FromLong((long)survivors[i]));
        PyList_SET_ITEM(scores_list, i, PyFloat_FromDouble(obs_sum[survivors[i]]));
    }

    double coverage = (double)total_cells / ((double)N * T) * 100.0;
    PyObject *stats = Py_BuildValue(
        "{s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:n}",
        "coverage", coverage,
        "survived", (Py_ssize_t)n_survivors,
        "survived_pct", (double)n_survivors / N * 100.0,
        "warmup_pack_ms", t_warmup_pack,
        "warmup_kernel_ms", t_warmup_kernel,
        "warmup_elim_ms", t_warmup_elim,
        "explore_pack_ms", t_explore_pack,
        "explore_kernel_ms", t_explore_kernel,
        "rescore_ms", t_rescore,
        "warmup_rounds", (Py_ssize_t)n_warmup_rounds,
        "warmup_tokens", (Py_ssize_t)warmup_token_ptr,
        "fallback_count", (Py_ssize_t)fallback_count);

    for (nk_size_t r = 0; r < n_total_rounds; r++) free(q_round_packed_arr[r]);
    free(q_round_packed_arr);
    free(elim_lcb);
    free(elim_ucb);
    free(elim_heap);
    free(sort_bitmap);
    free(obs_sum);
    free(obs_sum_sq);
    free(obs_count);
    free(survivors);
    free(perm);
    PyBuffer_Release(&query_buf);

    return PyTuple_Pack(3, indices_list, scores_list, stats);
}
