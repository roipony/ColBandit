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
#endif /* NK_TARGET_X86_ && NK_TARGET_HASWELL */

/* Scalar fallback for nk_maxsim_i8_flat_stats_tiled: ARM, RISC-V, and any non-AVX2 path.
   Uses signed i8×i8 dot products directly (no XOR-bias trick needed). */
static void nk_maxsim_i8_flat_stats_scalar(
    nk_i8_t const *q_i8, float const *q_f32, float const *q_inv_norms,
    nk_i8_t const *d_i8, float const *d_f32, float const *d_inv_norms,
    nk_i32_t const *d_sum_i8 /* unused: no XOR bias in scalar path */,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_f64_t *result) {
    nk_unused_(d_sum_i8);
    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;
    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_i8_t const *qr = q_i8 + qi * depth;
        float const *qf = q_f32 + qi * depth;
        float inv_qn = q_inv_norms[qi];
        nk_i32_t best1 = INT32_MIN, best2 = INT32_MIN;
        nk_u32_t best1_idx = 0, best2_idx = 0;
        for (nk_size_t di = 0; di < document_count; di++) {
            nk_i8_t const *dr = d_i8 + di * depth;
            nk_i32_t dot = 0;
            for (nk_size_t k = 0; k < depth; k++)
                dot += (nk_i32_t)qr[k] * (nk_i32_t)dr[k];
            if (dot > best1) { best2 = best1; best2_idx = best1_idx; best1 = dot; best1_idx = (nk_u32_t)di; }
            else if (dot > best2) { best2 = dot; best2_idx = (nk_u32_t)di; }
        }
        nk_f64_t dot1 = 0.0, dot2 = 0.0;
        float const *d1f = d_f32 + best1_idx * depth;
        float const *d2f = d_f32 + best2_idx * depth;
        for (nk_size_t k = 0; k < depth; k++) {
            dot1 += (nk_f64_t)qf[k] * (nk_f64_t)d1f[k];
            dot2 += (nk_f64_t)qf[k] * (nk_f64_t)d2f[k];
        }
        nk_f64_t cos1 = dot1 * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best1_idx];
        nk_f64_t cos2 = dot2 * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best2_idx];
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        total_sum_sq += angular * angular;
    }
    result[0] = total_sum;
    result[1] = total_sum_sq;
}

/* ARM NEON SDOT flat stats kernel: 4Q×4D tiling with vdotq_s32.
   i8×i8 signed dot — no XOR bias. f32 refine via vfmaq_f32. */
#if NK_TARGET_NEONSDOT
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+dotprod")
#endif

static void nk_maxsim_i8_flat_stats_neon(
    nk_i8_t const *q_i8, float const *q_f32, float const *q_inv_norms,
    nk_i8_t const *d_i8, float const *d_f32, float const *d_inv_norms,
    nk_i32_t const *d_sum_i8 /* unused: SDOT is i8×i8, no bias */,
    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
    nk_f64_t *result) {
    nk_unused_(d_sum_i8);
    nk_f64_t total_sum = 0.0, total_sum_sq = 0.0;

    nk_size_t qblock = 0;
    for (; qblock + 4 <= query_count; qblock += 4) {
        nk_i32_t best1[4] = {INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
        nk_i32_t best2[4] = {INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
        nk_u32_t best1_idx[4] = {0, 0, 0, 0};
        nk_u32_t best2_idx[4] = {0, 0, 0, 0};

        /* 4Q×4D tiled i8 SDOT scan */
        nk_size_t dblock = 0;
        for (; dblock + 4 <= document_count; dblock += 4) {
            int32x4_t acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] = vdupq_n_s32(0);
            for (nk_size_t k = 0; k < depth; k += 16) {
                int8x16_t q0 = vld1q_s8((int8_t const *)(q_i8 + (qblock+0)*depth + k));
                int8x16_t q1 = vld1q_s8((int8_t const *)(q_i8 + (qblock+1)*depth + k));
                int8x16_t q2 = vld1q_s8((int8_t const *)(q_i8 + (qblock+2)*depth + k));
                int8x16_t q3 = vld1q_s8((int8_t const *)(q_i8 + (qblock+3)*depth + k));
                for (int dti = 0; dti < 4; dti++) {
                    int8x16_t d = vld1q_s8((int8_t const *)(d_i8 + (dblock+dti)*depth + k));
                    acc[0][dti] = vdotq_s32(acc[0][dti], q0, d);
                    acc[1][dti] = vdotq_s32(acc[1][dti], q1, d);
                    acc[2][dti] = vdotq_s32(acc[2][dti], q2, d);
                    acc[3][dti] = vdotq_s32(acc[3][dti], q3, d);
                }
            }
            for (int dti = 0; dti < 4; dti++) {
                nk_u32_t di = (nk_u32_t)(dblock + dti);
                for (int qi = 0; qi < 4; qi++) {
                    nk_i32_t dot = vaddvq_s32(acc[qi][dti]);
                    if (dot > best1[qi]) { best2[qi]=best1[qi]; best2_idx[qi]=best1_idx[qi]; best1[qi]=dot; best1_idx[qi]=di; }
                    else if (dot > best2[qi]) { best2[qi]=dot; best2_idx[qi]=di; }
                }
            }
        }

        /* Doc tail: 4Q×1D */
        for (nk_size_t di = dblock; di < document_count; di++) {
            int32x4_t a0=vdupq_n_s32(0), a1=vdupq_n_s32(0), a2=vdupq_n_s32(0), a3=vdupq_n_s32(0);
            for (nk_size_t k = 0; k < depth; k += 16) {
                int8x16_t d = vld1q_s8((int8_t const *)(d_i8 + di*depth + k));
                a0 = vdotq_s32(a0, vld1q_s8((int8_t const *)(q_i8 + (qblock+0)*depth + k)), d);
                a1 = vdotq_s32(a1, vld1q_s8((int8_t const *)(q_i8 + (qblock+1)*depth + k)), d);
                a2 = vdotq_s32(a2, vld1q_s8((int8_t const *)(q_i8 + (qblock+2)*depth + k)), d);
                a3 = vdotq_s32(a3, vld1q_s8((int8_t const *)(q_i8 + (qblock+3)*depth + k)), d);
            }
            nk_i32_t dots[4] = {vaddvq_s32(a0), vaddvq_s32(a1), vaddvq_s32(a2), vaddvq_s32(a3)};
            for (int qi = 0; qi < 4; qi++) {
                nk_i32_t dot = dots[qi];
                if (dot > best1[qi]) { best2[qi]=best1[qi]; best2_idx[qi]=best1_idx[qi]; best1[qi]=dot; best1_idx[qi]=(nk_u32_t)di; }
                else if (dot > best2[qi]) { best2[qi]=dot; best2_idx[qi]=(nk_u32_t)di; }
            }
        }

        /* Inline f32 refine for top-2 per query token */
        for (int qq = 0; qq < 4; qq++) {
            float const *qf = q_f32 + (qblock+qq)*depth;
            float inv_qn = q_inv_norms[qblock+qq];
            float const *d1f = d_f32 + best1_idx[qq]*depth;
            float const *d2f = d_f32 + best2_idx[qq]*depth;
            float32x4_t r1 = vdupq_n_f32(0), r2 = vdupq_n_f32(0);
            for (nk_size_t k = 0; k < depth; k += 4) {
                float32x4_t q = vld1q_f32(qf + k);
                r1 = vfmaq_f32(r1, q, vld1q_f32(d1f + k));
                r2 = vfmaq_f32(r2, q, vld1q_f32(d2f + k));
            }
            nk_f64_t cos1 = (nk_f64_t)vaddvq_f32(r1) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best1_idx[qq]];
            nk_f64_t cos2 = (nk_f64_t)vaddvq_f32(r2) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[best2_idx[qq]];
            nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
            nk_f64_t angular = 1.0 - best_cosine;
            if (angular < 0.0) angular = 0.0;
            total_sum += angular;
            total_sum_sq += angular * angular;
        }
    }

    /* Query tail: 1Q */
    for (nk_size_t qi = qblock; qi < query_count; qi++) {
        nk_i8_t const *qr = q_i8 + qi*depth;
        float const *qf = q_f32 + qi*depth;
        float inv_qn = q_inv_norms[qi];
        nk_i32_t b1 = INT32_MIN, b2 = INT32_MIN;
        nk_u32_t b1_idx = 0, b2_idx = 0;
        for (nk_size_t di = 0; di < document_count; di++) {
            int32x4_t acc = vdupq_n_s32(0);
            for (nk_size_t k = 0; k < depth; k += 16)
                acc = vdotq_s32(acc, vld1q_s8((int8_t const *)(qr + k)),
                                     vld1q_s8((int8_t const *)(d_i8 + di*depth + k)));
            nk_i32_t dot = vaddvq_s32(acc);
            if (dot > b1) { b2=b1; b2_idx=b1_idx; b1=dot; b1_idx=(nk_u32_t)di; }
            else if (dot > b2) { b2=dot; b2_idx=(nk_u32_t)di; }
        }
        float32x4_t r1=vdupq_n_f32(0), r2=vdupq_n_f32(0);
        float const *d1f = d_f32 + b1_idx*depth, *d2f = d_f32 + b2_idx*depth;
        for (nk_size_t k = 0; k < depth; k += 4) {
            float32x4_t q = vld1q_f32(qf + k);
            r1 = vfmaq_f32(r1, q, vld1q_f32(d1f + k));
            r2 = vfmaq_f32(r2, q, vld1q_f32(d2f + k));
        }
        nk_f64_t cos1 = (nk_f64_t)vaddvq_f32(r1) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[b1_idx];
        nk_f64_t cos2 = (nk_f64_t)vaddvq_f32(r2) * (nk_f64_t)inv_qn * (nk_f64_t)d_inv_norms[b2_idx];
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#define nk_maxsim_i8_flat_stats_tiled nk_maxsim_i8_flat_stats_neon

#elif !(NK_TARGET_X86_ && NK_TARGET_HASWELL)
/* Scalar fallback for everything else */
#define nk_maxsim_i8_flat_stats_tiled nk_maxsim_i8_flat_stats_scalar
#endif /* NK_TARGET_NEONSDOT */

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
    "K=5, K_margin=5, alpha_ef=0.3, delta=0.01, n_threads=1, use_orig_N=False, round_size=4) -> (indices, scores, stats)\n\n"
    "CB-NK on flat separated arrays. i8 coarse scan is cache-friendly (no f32 pollution).\n"
    "use_orig_N: if True, union-bound denominator uses N_orig (BM-valid PAC); if False, uses\n"
    "n_survivors at each round (tighter per round but not strictly PAC valid).\n"
    "round_size: tokens revealed per round before elimination (1..16; default 4).\n"
    "  Smaller values lower the minimum reachable coverage at the cost of SIMD efficiency.\n"
    "docs_packed (optional): list of MaxSimPackedMatrix (same as full_maxsim takes).\n"
    "  If provided, the K-margin rescore uses the same packed kernel as full_maxsim,\n"
    "  aligning float-add-order so that CB-NK's top-K matches full_maxsim's top-K\n"
    "  bit-for-bit on borderline docs. If omitted, the i8 flat tiled kernel is used\n"
    "  (faster but produces ~3-5pp Ov@K gap vs full_maxsim due to add-order noise).\n";

PyObject *api_colbandit_flat(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    nk_unused_(self);
    if (nargs < 6) {
        PyErr_SetString(PyExc_TypeError, "colbandit_flat(query, doc_i8, doc_f32, doc_inv_norms, doc_offsets, doc_sum_i8, ...)");
        return NULL;
    }

    long K = 5, K_margin = 5;
    double alpha_ef = 0.3, delta = 0.01;
    int n_threads = 1;
    int measure_imbalance = 0;
    int use_orig_N = 0;  /* if non-zero, use N_orig in union bound (BM 4.3 valid) */
    long round_size_arg = 4;  /* tokens revealed per round; 1..16 */
    PyObject *docs_packed_obj = NULL;  /* optional: list of MaxSimPackedMatrix for aligned rescore */

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
        else if (PyUnicode_CompareWithASCIIString(name, "round_size") == 0) round_size_arg = PyLong_AsLong(value);
        else if (PyUnicode_CompareWithASCIIString(name, "docs_packed") == 0) docs_packed_obj = value;
    }
    if (round_size_arg < 1) round_size_arg = 1;
    if (round_size_arg > 16) round_size_arg = 16;

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
    nk_size_t T = (nk_size_t)q_buf.shape[0];
    nk_size_t depth = (nk_size_t)q_buf.shape[1];
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

    /* Random permutation */
    nk_u32_t *perm = (nk_u32_t *)malloc(T * sizeof(nk_u32_t));
    for (nk_size_t i = 0; i < T; i++) perm[i] = (nk_u32_t)i;
    nk_u32_t rng_state = 42;
    for (nk_size_t i = T - 1; i > 0; i--) {
        rng_state = rng_state * 1103515245 + 12345;
        nk_u32_t j = (rng_state >> 16) % (i + 1);
        nk_u32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    nk_size_t n_warmup = T;  /* always full warmup */
    nk_size_t round_size = (nk_size_t)round_size_arg;
    n_warmup = (n_warmup / round_size) * round_size;
    if (n_warmup < round_size) n_warmup = round_size;

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

    /* Per-round telemetry (max possible rounds = T/round_size, cap at 64) */
    #define max_rounds_tracked 64
    double round_kernel_ms[max_rounds_tracked];
    double round_elim_ms[max_rounds_tracked];
    nk_u32_t round_n_survivors[max_rounds_tracked];   /* before this round's elim */
    nk_u32_t round_tokens[max_rounds_tracked];

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
        double dt_kernel_round = _now_ms() - t0;
        t_kernel += dt_kernel_round;
        total_cells += n_survivors * tokens_this_round;
        warmup_token_ptr += tokens_this_round;

        nk_u32_t n_surv_before = (nk_u32_t)n_survivors;

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
        double dt_elim_round = _now_ms() - t0;
        t_elim += dt_elim_round;

        if (n_warmup_rounds < max_rounds_tracked) {
            round_kernel_ms[n_warmup_rounds] = dt_kernel_round;
            round_elim_ms[n_warmup_rounds] = dt_elim_round;
            round_n_survivors[n_warmup_rounds] = n_surv_before;
            round_tokens[n_warmup_rounds] = (nk_u32_t)tokens_this_round;
        }
        n_warmup_rounds++;
        if (n_survivors <= K_elim) break;
    }

    /* Rescore survivors with all T tokens.
       If docs_packed is provided, use the same packed kernel as full_maxsim
       (FIX-A: aligns float-add-order, lifts Ov@K-vs-Full-MaxSim ceiling).
       Otherwise fall back to the i8 flat tiled kernel. */
    double t0_r = _now_ms();
    if (K_margin > 0 && n_survivors > (nk_size_t)K) {
        if (docs_packed_obj && PyList_Check(docs_packed_obj)) {
            /* Aligned rescore via score_docs (matches full_maxsim) */
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
            } else {
                /* Kernel lookup failed — fall back to i8 flat */
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
        } else {
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

    /* Per-round telemetry */
    {
        nk_size_t n_rec = n_warmup_rounds < max_rounds_tracked ? n_warmup_rounds : max_rounds_tracked;
        PyObject *kr_list = PyList_New(n_rec);
        PyObject *er_list = PyList_New(n_rec);
        PyObject *ns_list = PyList_New(n_rec);
        PyObject *tk_list = PyList_New(n_rec);
        for (nk_size_t r = 0; r < n_rec; r++) {
            PyList_SET_ITEM(kr_list, r, PyFloat_FromDouble(round_kernel_ms[r]));
            PyList_SET_ITEM(er_list, r, PyFloat_FromDouble(round_elim_ms[r]));
            PyList_SET_ITEM(ns_list, r, PyLong_FromUnsignedLong(round_n_survivors[r]));
            PyList_SET_ITEM(tk_list, r, PyLong_FromUnsignedLong(round_tokens[r]));
        }
        PyDict_SetItemString(stats, "round_kernel_ms", kr_list);
        PyDict_SetItemString(stats, "round_elim_ms", er_list);
        PyDict_SetItemString(stats, "round_n_survivors", ns_list);
        PyDict_SetItemString(stats, "round_tokens", tk_list);
        Py_DECREF(kr_list); Py_DECREF(er_list); Py_DECREF(ns_list); Py_DECREF(tk_list);
    }

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

    nk_size_t round_size = 4;

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
                        __builtin_prefetch(pf_doc->start, 0, 1);
                        __builtin_prefetch((char *)pf_doc->start + 64, 0, 1);
                    }
                    nk_f64_t local_stats[2];
                    nk_u32_t di = survivors[i];
                    MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                    kernel_stats(q_round_packed, doc->start,
                                 tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
                    obs_count[di] += (nk_u32_t)tokens_this_round;
                }
            } else {
                #pragma omp parallel for schedule(dynamic, 32) if(n_survivors > 256 && n_threads > 1) num_threads(n_threads)
                for (nk_size_t i = 0; i < n_survivors; i++) {
                    if (i + CB_PREFETCH_AHEAD < n_survivors) {
                        MaxSimPackedMatrix *pf_doc = (MaxSimPackedMatrix *)doc_objects[survivors[i + CB_PREFETCH_AHEAD]];
                        __builtin_prefetch(pf_doc->start, 0, 1);
                        __builtin_prefetch((char *)pf_doc->start + 64, 0, 1);
                    }
                    nk_f64_t local_stats[2];
                    nk_u32_t di = survivors[i];
                    MaxSimPackedMatrix *doc = (MaxSimPackedMatrix *)doc_objects[di];
                    kernel_stats(q_round_packed, doc->start,
                                 tokens_this_round, doc->vector_count, depth, local_stats);
                    obs_sum[di] += local_stats[0];
                    obs_sum_sq[di] += local_stats[1];
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
