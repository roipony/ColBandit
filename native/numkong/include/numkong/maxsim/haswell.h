/**
 *  @brief SIMD-accelerated MaxSim (angular distance late-interaction) for Haswell (AVX2).
 *  @file include/numkong/maxsim/haswell.h
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX2 VPMADDUBSW (u8×i8→i16) + VPMADDWD (i16→i32) for coarse i8 screening.
 *  Quantization range [-79, 79] ensures no i16 saturation: worst pair sum = 2 × 207 × 79 = 32706 < 32767.
 *  Bias correction via XOR-0x80 converts signed queries to unsigned, then subtracts 128 × sum_quantized.
 *
 *  4x4 register tiling: 4 queries × 4 documents = 16 YMM accumulators per depth loop.
 *  Depth steps at 32 bytes (YMM width in bytes).
 */
#ifndef NK_MAXSIM_HASWELL_H
#define NK_MAXSIM_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/maxsim/serial.h"   // `nk_maxsim_packed_header_t`
#include "numkong/dot.h"             // `nk_dot_bf16`, `nk_dot_f32`, `nk_dot_f16`
#include "numkong/cast/haswell.h"    // `nk_f16_to_f32_haswell`
#include "numkong/spatial/haswell.h" // `nk_f32_sqrt_haswell`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_haswell(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_bf16_t), 32);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_haswell(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f32_t), 32);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_haswell(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f16_t), 32);
}

NK_PUBLIC void nk_maxsim_pack_bf16_haswell( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 79.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_serial,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f32_haswell( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 79.0f, nk_f32_to_f32_,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f16_haswell( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 79.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_haswell,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

/** @brief Reduces 4 YMM i32x8 accumulators to a single __m128i with 4 horizontal sums. */
NK_INTERNAL __m128i nk_maxsim_reduce_i32x8x4_haswell_(        //
    __m256i accumulator_a_i32x8, __m256i accumulator_b_i32x8, //
    __m256i accumulator_c_i32x8, __m256i accumulator_d_i32x8) {
    // Step 1: 8 -> 4 (extract high 128-bit half and add to low half)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_a_i32x8),
                                        _mm256_extracti128_si256(accumulator_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_b_i32x8),
                                        _mm256_extracti128_si256(accumulator_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_c_i32x8),
                                        _mm256_extracti128_si256(accumulator_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_d_i32x8),
                                        _mm256_extracti128_si256(accumulator_d_i32x8, 1));
    // Step 2: 4x4 transpose + reduce -> [sum_a, sum_b, sum_c, sum_d]
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane_0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane_1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane_2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane_3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    return _mm_add_epi32(_mm_add_epi32(sum_lane_0_i32x4, sum_lane_1_i32x4),
                         _mm_add_epi32(sum_lane_2_i32x4, sum_lane_3_i32x4));
}

/**
 *  @brief Factored coarse i8 argmax kernel for Haswell.
 *  Uses AVX2 VPMADDUBSW (u8×i8→i16) + VPMADDWD (i16×1→i32) with XOR-0x80 bias.
 *  4Q×4D register tiling with 16 YMM accumulators.
 */
/**
 * Inline top-2 argmax: identical 4Q×4D tiling as standard argmax, but tracks
 * rank-1 AND rank-2 per query lane using SIMD blends.
 * When a new doc beats rank-1, old rank-1 becomes rank-2 candidate.
 * Only ~4 extra SIMD ops per doc vs the standard argmax.
 */
NK_INTERNAL void nk_maxsim_coarse_top2_haswell_(
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,
    nk_maxsim_vector_metadata_t const *document_metadata,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth_i8_padded,
    nk_u32_t *best_document_indices,
    nk_u32_t *second_document_indices) {

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    /* Macro: gated top-2 update. First check if dot > rank2 (gate).
       If no lane passes the gate, skip all updates (well-predicted branch, ~90% skip).
       Otherwise do the full rank1/rank2 cascade. */
    #define TOP2_UPDATE(new_dots, didx_i32x4)                                                   \
    do {                                                                                         \
        __m128i _gate = _mm_cmpgt_epi32(new_dots, running_max2);                                \
        if (!_mm_testz_si128(_gate, _gate)) {                                                   \
            __m128i _beats1 = _mm_cmpgt_epi32(new_dots, running_max1);                          \
            __m128i _old_max1 = running_max1;                                                    \
            __m128i _old_arg1 = running_argmax1;                                                 \
            running_max1 = _mm_blendv_epi8(running_max1, new_dots, _beats1);                    \
            running_argmax1 = _mm_blendv_epi8(running_argmax1, didx_i32x4, _beats1);           \
            __m128i _r2_cand = _mm_blendv_epi8(new_dots, _old_max1, _beats1);                  \
            __m128i _r2_arg  = _mm_blendv_epi8(didx_i32x4, _old_arg1, _beats1);                \
            __m128i _beats2 = _mm_cmpgt_epi32(_r2_cand, running_max2);                          \
            running_max2 = _mm_blendv_epi8(running_max2, _r2_cand, _beats2);                   \
            running_argmax2 = _mm_blendv_epi8(running_argmax2, _r2_arg, _beats2);              \
        }                                                                                        \
    } while(0)

    /* Primary path: 4-query grouping */
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        __m128i running_max1 = _mm_set1_epi32(NK_I32_MIN);
        __m128i running_argmax1 = _mm_setzero_si128();
        __m128i running_max2 = _mm_set1_epi32(NK_I32_MIN);
        __m128i running_argmax2 = _mm_setzero_si128();

        /* 4Q×4D document blocking — depth loop identical to standard argmax */
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x8[query_tile_index][document_tile_index] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i query_biased_u8x32_0 = _mm256_xor_si256(
                    _mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)), xor_mask_u8x32);
                __m256i query_biased_u8x32_1 = _mm256_xor_si256(
                    _mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)), xor_mask_u8x32);
                __m256i query_biased_u8x32_2 = _mm256_xor_si256(
                    _mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)), xor_mask_u8x32);
                __m256i query_biased_u8x32_3 = _mm256_xor_si256(
                    _mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)), xor_mask_u8x32);
                __m256i document_i8x32, products_i16x16, products_i32x8;
                for (nk_size_t dti = 0; dti < 4; dti++) {
                    document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8 + (document_block_start_index + dti) * depth_i8_padded + depth_index));
                    products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_0, document_i8x32); products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16); accumulator_tiles_i32x8[0][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[0][dti], products_i32x8);
                    products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_1, document_i8x32); products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16); accumulator_tiles_i32x8[1][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[1][dti], products_i32x8);
                    products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_2, document_i8x32); products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16); accumulator_tiles_i32x8[2][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[2][dti], products_i32x8);
                    products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_3, document_i8x32); products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16); accumulator_tiles_i32x8[3][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[3][dti], products_i32x8);
                }
            }

            /* Reduce + bias correct + top-2 update for each of 4 docs */
            for (nk_size_t dti = 0; dti < 4; dti++) {
                __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(
                    accumulator_tiles_i32x8[0][dti], accumulator_tiles_i32x8[1][dti],
                    accumulator_tiles_i32x8[2][dti], accumulator_tiles_i32x8[3][dti]);
                nk_i32_t bias = 128 * document_metadata[document_block_start_index + dti].sum_i8_i32;
                __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
                __m128i didx = _mm_set1_epi32((int)(document_block_start_index + dti));
                TOP2_UPDATE(dots, didx);
            }
        }

        /* Document tail: 4Q×1D */
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));
                __m256i p, r;
                p = _mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+0)*depth_i8_padded+depth_index)), xor_mask_u8x32), d);
                r = _mm256_madd_epi16(p, ones_i16x16); acc0 = _mm256_add_epi32(acc0, r);
                p = _mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+1)*depth_i8_padded+depth_index)), xor_mask_u8x32), d);
                r = _mm256_madd_epi16(p, ones_i16x16); acc1 = _mm256_add_epi32(acc1, r);
                p = _mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+2)*depth_i8_padded+depth_index)), xor_mask_u8x32), d);
                r = _mm256_madd_epi16(p, ones_i16x16); acc2 = _mm256_add_epi32(acc2, r);
                p = _mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+3)*depth_i8_padded+depth_index)), xor_mask_u8x32), d);
                r = _mm256_madd_epi16(p, ones_i16x16); acc3 = _mm256_add_epi32(acc3, r);
            }
            __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(acc0, acc1, acc2, acc3);
            nk_i32_t bias = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
            __m128i didx = _mm_set1_epi32((int)document_index);
            TOP2_UPDATE(dots, didx);
        }

        best_document_indices[query_block_start_index + 0] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 0);
        best_document_indices[query_block_start_index + 1] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 1);
        best_document_indices[query_block_start_index + 2] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 2);
        best_document_indices[query_block_start_index + 3] = (nk_u32_t)_mm_extract_epi32(running_argmax1, 3);
        second_document_indices[query_block_start_index + 0] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 0);
        second_document_indices[query_block_start_index + 1] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 1);
        second_document_indices[query_block_start_index + 2] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 2);
        second_document_indices[query_block_start_index + 3] = (nk_u32_t)_mm_extract_epi32(running_argmax2, 3);
    }

    /* Query tail: 1Q×1D with scalar top-2 */
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        nk_i32_t max1 = NK_I32_MIN, max2 = NK_I32_MIN;
        nk_u32_t arg1 = 0, arg2 = 0;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            __m256i accumulator_i32x8 = _mm256_setzero_si256();
            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));
                __m256i q = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8_row + depth_index)), xor_mask_u8x32);
                __m256i p = _mm256_maddubs_epi16(q, d);
                accumulator_i32x8 = _mm256_add_epi32(accumulator_i32x8, _mm256_madd_epi16(p, ones_i16x16));
            }
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(accumulator_i32x8), _mm256_extracti128_si256(accumulator_i32x8, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot = _mm_extract_epi32(s, 0) - 128 * document_metadata[document_index].sum_i8_i32;

            if (dot > max1) { max2 = max1; arg2 = arg1; max1 = dot; arg1 = (nk_u32_t)document_index; }
            else if (dot > max2) { max2 = dot; arg2 = (nk_u32_t)document_index; }
        }
        best_document_indices[query_index] = arg1;
        second_document_indices[query_index] = arg2;
    }

    #undef TOP2_UPDATE
}

/**
 * Inline top-4 argmax: same 4Q×4D tiling, tracks 4 ranks per query lane.
 * Cascade: new beats rank-1 → rank-1→2→3→4.
 * topk_indices: [query_count][4], topk_indices[q*4+0] = rank-1, [q*4+3] = rank-4.
 */
NK_INTERNAL void nk_maxsim_coarse_top4_haswell_(
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,
    nk_maxsim_vector_metadata_t const *document_metadata,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth_i8_padded,
    nk_u32_t *topk_indices) {  /* [query_count * 4] */

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    /* Top-4 cascade update macro.
       When new > r1: old r1→r2 candidate, old r2→r3 candidate, old r3→r4 candidate.
       When new > r2 (not r1): old r2→r3 cand, old r3→r4 cand.
       When new > r3 (not r2): old r3→r4 cand.
       When new > r4 (not r3): direct update. */
    #define TOP4_UPDATE(new_dots, didx_i32x4)                                                   \
    do {                                                                                         \
        __m128i _gate4 = _mm_cmpgt_epi32(new_dots, rmax4);                                      \
        if (_mm_testz_si128(_gate4, _gate4)) break;                                              \
        /* Check rank-1 */                                                                       \
        __m128i _b1 = _mm_cmpgt_epi32(new_dots, rmax1);                                        \
        __m128i _om1 = rmax1, _oa1 = rarg1;                                                     \
        rmax1 = _mm_blendv_epi8(rmax1, new_dots, _b1);                                         \
        rarg1 = _mm_blendv_epi8(rarg1, didx_i32x4, _b1);                                      \
        /* r2 candidate: if beat r1 → old r1; else → new */                                     \
        __m128i _c2 = _mm_blendv_epi8(new_dots, _om1, _b1);                                    \
        __m128i _a2 = _mm_blendv_epi8(didx_i32x4, _oa1, _b1);                                 \
        /* Check rank-2 */                                                                       \
        __m128i _b2 = _mm_cmpgt_epi32(_c2, rmax2);                                             \
        __m128i _om2 = rmax2, _oa2 = rarg2;                                                     \
        rmax2 = _mm_blendv_epi8(rmax2, _c2, _b2);                                              \
        rarg2 = _mm_blendv_epi8(rarg2, _a2, _b2);                                              \
        /* r3 candidate: if beat r2 → old r2; else → c2 itself */                               \
        __m128i _c3 = _mm_blendv_epi8(_c2, _om2, _b2);                                         \
        __m128i _a3 = _mm_blendv_epi8(_a2, _oa2, _b2);                                         \
        /* Check rank-3 */                                                                       \
        __m128i _b3 = _mm_cmpgt_epi32(_c3, rmax3);                                             \
        __m128i _om3 = rmax3, _oa3 = rarg3;                                                     \
        rmax3 = _mm_blendv_epi8(rmax3, _c3, _b3);                                              \
        rarg3 = _mm_blendv_epi8(rarg3, _a3, _b3);                                              \
        /* r4 candidate */                                                                       \
        __m128i _c4 = _mm_blendv_epi8(_c3, _om3, _b3);                                         \
        __m128i _a4 = _mm_blendv_epi8(_a3, _oa3, _b3);                                         \
        __m128i _b4 = _mm_cmpgt_epi32(_c4, rmax4);                                             \
        rmax4 = _mm_blendv_epi8(rmax4, _c4, _b4);                                              \
        rarg4 = _mm_blendv_epi8(rarg4, _a4, _b4);                                              \
    } while(0)

    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        __m128i rmax1 = _mm_set1_epi32(NK_I32_MIN), rarg1 = _mm_setzero_si128();
        __m128i rmax2 = _mm_set1_epi32(NK_I32_MIN), rarg2 = _mm_setzero_si128();
        __m128i rmax3 = _mm_set1_epi32(NK_I32_MIN), rarg3 = _mm_setzero_si128();
        __m128i rmax4 = _mm_set1_epi32(NK_I32_MIN), rarg4 = _mm_setzero_si128();

        /* 4Q×4D — depth loop identical to standard argmax */
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t qti = 0; qti < 4; qti++)
                for (nk_size_t dti = 0; dti < 4; dti++)
                    accumulator_tiles_i32x8[qti][dti] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i q0 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+0)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q1 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+1)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q2 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+2)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q3 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+3)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i d, p, r;
                for (nk_size_t dti = 0; dti < 4; dti++) {
                    d = _mm256_loadu_si256((__m256i const *)(document_i8 + (document_block_start_index+dti)*depth_i8_padded+depth_index));
                    p = _mm256_maddubs_epi16(q0,d); r = _mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[0][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[0][dti],r);
                    p = _mm256_maddubs_epi16(q1,d); r = _mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[1][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[1][dti],r);
                    p = _mm256_maddubs_epi16(q2,d); r = _mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[2][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[2][dti],r);
                    p = _mm256_maddubs_epi16(q3,d); r = _mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[3][dti] = _mm256_add_epi32(accumulator_tiles_i32x8[3][dti],r);
                }
            }

            for (nk_size_t dti = 0; dti < 4; dti++) {
                __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(
                    accumulator_tiles_i32x8[0][dti], accumulator_tiles_i32x8[1][dti],
                    accumulator_tiles_i32x8[2][dti], accumulator_tiles_i32x8[3][dti]);
                nk_i32_t bias = 128 * document_metadata[document_block_start_index + dti].sum_i8_i32;
                __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
                __m128i didx = _mm_set1_epi32((int)(document_block_start_index + dti));
                TOP4_UPDATE(dots, didx);
            }
        }

        /* Document tail: 4Q×1D */
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
            __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
            for (nk_size_t di = 0; di < depth_i8_padded; di += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(document_i8_row + di));
                __m256i p, r;
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+0)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc0=_mm256_add_epi32(acc0,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+1)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc1=_mm256_add_epi32(acc1,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+2)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc2=_mm256_add_epi32(acc2,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+3)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc3=_mm256_add_epi32(acc3,r);
            }
            __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(acc0,acc1,acc2,acc3);
            nk_i32_t bias = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
            __m128i didx = _mm_set1_epi32((int)document_index);
            TOP4_UPDATE(dots, didx);
        }

        for (int r = 0; r < 4; r++) {
            __m128i rarg = (r==0) ? rarg1 : (r==1) ? rarg2 : (r==2) ? rarg3 : rarg4;
            topk_indices[(query_block_start_index+0)*4+r] = (nk_u32_t)_mm_extract_epi32(rarg, 0);
            topk_indices[(query_block_start_index+1)*4+r] = (nk_u32_t)_mm_extract_epi32(rarg, 1);
            topk_indices[(query_block_start_index+2)*4+r] = (nk_u32_t)_mm_extract_epi32(rarg, 2);
            topk_indices[(query_block_start_index+3)*4+r] = (nk_u32_t)_mm_extract_epi32(rarg, 3);
        }
    }

    /* Query tail: 1Q×1D scalar top-4 */
    for (nk_size_t qi = query_block_start_index; qi < query_count; qi++) {
        nk_i8_t const *q_row = query_i8 + qi * depth_i8_padded;
        nk_i32_t mx[4] = {NK_I32_MIN, NK_I32_MIN, NK_I32_MIN, NK_I32_MIN};
        nk_u32_t ax[4] = {0, 0, 0, 0};
        for (nk_size_t di = 0; di < document_count; di++) {
            nk_i8_t const *d_row = document_i8 + di * depth_i8_padded;
            __m256i acc = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth_i8_padded; k += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_row + k));
                __m256i q = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_row + k)), xor_mask_u8x32);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q, d), ones_i16x16));
            }
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot = _mm_extract_epi32(s, 0) - 128 * document_metadata[di].sum_i8_i32;
            /* Scalar cascade insert */
            if (dot > mx[0]) { mx[3]=mx[2]; ax[3]=ax[2]; mx[2]=mx[1]; ax[2]=ax[1]; mx[1]=mx[0]; ax[1]=ax[0]; mx[0]=dot; ax[0]=(nk_u32_t)di; }
            else if (dot > mx[1]) { mx[3]=mx[2]; ax[3]=ax[2]; mx[2]=mx[1]; ax[2]=ax[1]; mx[1]=dot; ax[1]=(nk_u32_t)di; }
            else if (dot > mx[2]) { mx[3]=mx[2]; ax[3]=ax[2]; mx[2]=dot; ax[2]=(nk_u32_t)di; }
            else if (dot > mx[3]) { mx[3]=dot; ax[3]=(nk_u32_t)di; }
        }
        for (int r = 0; r < 4; r++) topk_indices[qi*4+r] = ax[r];
    }

    #undef TOP4_UPDATE
}

/**
 * Inline top-6 argmax: 4Q×4D tiling, cascade through 6 ranks.
 * 6 ranks × 2 registers (max+arg) = 12 XMMs — fits within AVX2's 16 without spills.
 */
#define NK_TOP6_K 6

NK_INTERNAL void nk_maxsim_coarse_top6_haswell_(
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,
    nk_maxsim_vector_metadata_t const *document_metadata,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth_i8_padded,
    nk_u32_t *topk_indices) {  /* [query_count * 6] */

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    #define TOP6_UPDATE(new_dots, didx_i32x4) do {                               \
        __m128i _gate6 = _mm_cmpgt_epi32(new_dots, rmax[NK_TOP6_K-1]);         \
        if (_mm_testz_si128(_gate6, _gate6)) break;                              \
        __m128i _c = (new_dots), _ci = (didx_i32x4);                            \
        for (int _r = 0; _r < NK_TOP6_K; _r++) {                                \
            __m128i _b = _mm_cmpgt_epi32(_c, rmax[_r]);                          \
            __m128i _ts = rmax[_r], _ti = rarg[_r];                              \
            rmax[_r] = _mm_blendv_epi8(rmax[_r], _c, _b);                       \
            rarg[_r] = _mm_blendv_epi8(rarg[_r], _ci, _b);                      \
            _c = _mm_blendv_epi8(_c, _ts, _b);                                  \
            _ci = _mm_blendv_epi8(_ci, _ti, _b);                                \
        }                                                                         \
    } while(0)

    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        __m128i rmax[NK_TOP6_K], rarg[NK_TOP6_K];
        for (int r = 0; r < NK_TOP6_K; r++) {
            rmax[r] = _mm_set1_epi32(NK_I32_MIN);
            rarg[r] = _mm_setzero_si128();
        }

        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t qti = 0; qti < 4; qti++)
                for (nk_size_t dti = 0; dti < 4; dti++)
                    accumulator_tiles_i32x8[qti][dti] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i q0 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+0)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q1 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+1)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q2 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+2)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q3 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+3)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i d, p, r;
                for (nk_size_t dti = 0; dti < 4; dti++) {
                    d = _mm256_loadu_si256((__m256i const *)(document_i8 + (document_block_start_index+dti)*depth_i8_padded+depth_index));
                    p=_mm256_maddubs_epi16(q0,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[0][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[0][dti],r);
                    p=_mm256_maddubs_epi16(q1,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[1][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[1][dti],r);
                    p=_mm256_maddubs_epi16(q2,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[2][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[2][dti],r);
                    p=_mm256_maddubs_epi16(q3,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[3][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[3][dti],r);
                }
            }

            for (nk_size_t dti = 0; dti < 4; dti++) {
                __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(
                    accumulator_tiles_i32x8[0][dti], accumulator_tiles_i32x8[1][dti],
                    accumulator_tiles_i32x8[2][dti], accumulator_tiles_i32x8[3][dti]);
                nk_i32_t bias = 128 * document_metadata[document_block_start_index + dti].sum_i8_i32;
                __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
                __m128i didx = _mm_set1_epi32((int)(document_block_start_index + dti));
                TOP6_UPDATE(dots, didx);
            }
        }

        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *d_row = document_i8 + document_index * depth_i8_padded;
            __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
            __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
            for (nk_size_t di = 0; di < depth_i8_padded; di += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_row + di));
                __m256i p, r;
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+0)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc0=_mm256_add_epi32(acc0,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+1)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc1=_mm256_add_epi32(acc1,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+2)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc2=_mm256_add_epi32(acc2,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+3)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc3=_mm256_add_epi32(acc3,r);
            }
            __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(acc0,acc1,acc2,acc3);
            nk_i32_t bias = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
            __m128i didx = _mm_set1_epi32((int)document_index);
            TOP6_UPDATE(dots, didx);
        }

        for (int r = 0; r < NK_TOP6_K; r++) {
            topk_indices[(query_block_start_index+0)*NK_TOP6_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 0);
            topk_indices[(query_block_start_index+1)*NK_TOP6_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 1);
            topk_indices[(query_block_start_index+2)*NK_TOP6_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 2);
            topk_indices[(query_block_start_index+3)*NK_TOP6_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 3);
        }
    }

    /* Query tail */
    for (nk_size_t qi = query_block_start_index; qi < query_count; qi++) {
        nk_i8_t const *q_row = query_i8 + qi * depth_i8_padded;
        nk_i32_t mx[NK_TOP6_K]; nk_u32_t ax[NK_TOP6_K];
        for (int r = 0; r < NK_TOP6_K; r++) { mx[r] = NK_I32_MIN; ax[r] = 0; }
        for (nk_size_t di = 0; di < document_count; di++) {
            nk_i8_t const *d_row = document_i8 + di * depth_i8_padded;
            __m256i acc = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth_i8_padded; k += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_row + k));
                __m256i q = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_row + k)), xor_mask_u8x32);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q, d), ones_i16x16));
            }
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot = _mm_extract_epi32(s, 0) - 128 * document_metadata[di].sum_i8_i32;
            nk_i32_t c = dot; nk_u32_t ci = (nk_u32_t)di;
            for (int r = 0; r < NK_TOP6_K; r++) {
                if (c > mx[r]) { nk_i32_t ts=mx[r]; nk_u32_t ti=ax[r]; mx[r]=c; ax[r]=ci; c=ts; ci=ti; }
            }
        }
        for (int r = 0; r < NK_TOP6_K; r++) topk_indices[qi*NK_TOP6_K+r] = ax[r];
    }

    #undef TOP6_UPDATE
}

/**
 * Inline top-10 argmax: 4Q×4D tiling, cascade update through 10 ranks.
 * Outputs both indices AND i8 scores (for gap-gated f32 refine).
 * Ranks are maintained in sorted order (descending) per query lane.
 */
#define NK_TOP10_K 10

NK_INTERNAL void nk_maxsim_coarse_top10_haswell_(
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,
    nk_maxsim_vector_metadata_t const *document_metadata,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth_i8_padded,
    nk_u32_t *topk_indices,   /* [query_count * 10] */
    nk_i32_t *topk_scores) {  /* [query_count * 10] */

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    /* Cascade update: new value bubbles down through sorted ranks.
       Each iteration: compare + 4 blends. For ranks where new > current,
       swap them — displaced value continues to next rank. */
    #define TOP10_UPDATE(new_dots, didx_i32x4) do {                              \
        __m128i _c = (new_dots), _ci = (didx_i32x4);                            \
        for (int _r = 0; _r < NK_TOP10_K; _r++) {                               \
            __m128i _b = _mm_cmpgt_epi32(_c, rmax[_r]);                          \
            __m128i _ts = rmax[_r], _ti = rarg[_r];                              \
            rmax[_r] = _mm_blendv_epi8(rmax[_r], _c, _b);                       \
            rarg[_r] = _mm_blendv_epi8(rarg[_r], _ci, _b);                      \
            _c = _mm_blendv_epi8(_c, _ts, _b);                                  \
            _ci = _mm_blendv_epi8(_ci, _ti, _b);                                \
        }                                                                         \
    } while(0)

    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        __m128i rmax[NK_TOP10_K], rarg[NK_TOP10_K];
        for (int r = 0; r < NK_TOP10_K; r++) {
            rmax[r] = _mm_set1_epi32(NK_I32_MIN);
            rarg[r] = _mm_setzero_si128();
        }

        /* 4Q×4D — identical depth loop */
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t qti = 0; qti < 4; qti++)
                for (nk_size_t dti = 0; dti < 4; dti++)
                    accumulator_tiles_i32x8[qti][dti] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i q0 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+0)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q1 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+1)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q2 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+2)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i q3 = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(query_i8 + (query_block_start_index+3)*depth_i8_padded+depth_index)), xor_mask_u8x32);
                __m256i d, p, r;
                for (nk_size_t dti = 0; dti < 4; dti++) {
                    d = _mm256_loadu_si256((__m256i const *)(document_i8 + (document_block_start_index+dti)*depth_i8_padded+depth_index));
                    p=_mm256_maddubs_epi16(q0,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[0][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[0][dti],r);
                    p=_mm256_maddubs_epi16(q1,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[1][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[1][dti],r);
                    p=_mm256_maddubs_epi16(q2,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[2][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[2][dti],r);
                    p=_mm256_maddubs_epi16(q3,d); r=_mm256_madd_epi16(p,ones_i16x16); accumulator_tiles_i32x8[3][dti]=_mm256_add_epi32(accumulator_tiles_i32x8[3][dti],r);
                }
            }

            for (nk_size_t dti = 0; dti < 4; dti++) {
                __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(
                    accumulator_tiles_i32x8[0][dti], accumulator_tiles_i32x8[1][dti],
                    accumulator_tiles_i32x8[2][dti], accumulator_tiles_i32x8[3][dti]);
                nk_i32_t bias = 128 * document_metadata[document_block_start_index + dti].sum_i8_i32;
                __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
                __m128i didx = _mm_set1_epi32((int)(document_block_start_index + dti));
                TOP10_UPDATE(dots, didx);
            }
        }

        /* Document tail: 4Q×1D */
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *d_row = document_i8 + document_index * depth_i8_padded;
            __m256i acc0=_mm256_setzero_si256(), acc1=_mm256_setzero_si256();
            __m256i acc2=_mm256_setzero_si256(), acc3=_mm256_setzero_si256();
            for (nk_size_t di = 0; di < depth_i8_padded; di += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_row + di));
                __m256i p, r;
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+0)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc0=_mm256_add_epi32(acc0,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+1)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc1=_mm256_add_epi32(acc1,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+2)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc2=_mm256_add_epi32(acc2,r);
                p=_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_loadu_si256((__m256i const*)(query_i8+(query_block_start_index+3)*depth_i8_padded+di)),xor_mask_u8x32),d); r=_mm256_madd_epi16(p,ones_i16x16); acc3=_mm256_add_epi32(acc3,r);
            }
            __m128i reduced = nk_maxsim_reduce_i32x8x4_haswell_(acc0,acc1,acc2,acc3);
            nk_i32_t bias = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i dots = _mm_sub_epi32(reduced, _mm_set1_epi32(bias));
            __m128i didx = _mm_set1_epi32((int)document_index);
            TOP10_UPDATE(dots, didx);
        }

        /* Extract results: 10 ranks × 4 query lanes */
        for (int r = 0; r < NK_TOP10_K; r++) {
            topk_indices[(query_block_start_index+0)*NK_TOP10_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 0);
            topk_indices[(query_block_start_index+1)*NK_TOP10_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 1);
            topk_indices[(query_block_start_index+2)*NK_TOP10_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 2);
            topk_indices[(query_block_start_index+3)*NK_TOP10_K+r] = (nk_u32_t)_mm_extract_epi32(rarg[r], 3);
            topk_scores[(query_block_start_index+0)*NK_TOP10_K+r] = _mm_extract_epi32(rmax[r], 0);
            topk_scores[(query_block_start_index+1)*NK_TOP10_K+r] = _mm_extract_epi32(rmax[r], 1);
            topk_scores[(query_block_start_index+2)*NK_TOP10_K+r] = _mm_extract_epi32(rmax[r], 2);
            topk_scores[(query_block_start_index+3)*NK_TOP10_K+r] = _mm_extract_epi32(rmax[r], 3);
        }
    }

    /* Query tail: 1Q×1D scalar top-10 */
    for (nk_size_t qi = query_block_start_index; qi < query_count; qi++) {
        nk_i8_t const *q_row = query_i8 + qi * depth_i8_padded;
        nk_i32_t mx[NK_TOP10_K]; nk_u32_t ax[NK_TOP10_K];
        for (int r = 0; r < NK_TOP10_K; r++) { mx[r] = NK_I32_MIN; ax[r] = 0; }

        for (nk_size_t di = 0; di < document_count; di++) {
            nk_i8_t const *d_row = document_i8 + di * depth_i8_padded;
            __m256i acc = _mm256_setzero_si256();
            for (nk_size_t k = 0; k < depth_i8_padded; k += 32) {
                __m256i d = _mm256_loadu_si256((__m256i const *)(d_row + k));
                __m256i q = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(q_row + k)), xor_mask_u8x32);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q, d), ones_i16x16));
            }
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            nk_i32_t dot = _mm_extract_epi32(s, 0) - 128 * document_metadata[di].sum_i8_i32;
            /* Scalar cascade insert */
            nk_i32_t c = dot; nk_u32_t ci = (nk_u32_t)di;
            for (int r = 0; r < NK_TOP10_K; r++) {
                if (c > mx[r]) { nk_i32_t ts = mx[r]; nk_u32_t ti = ax[r]; mx[r] = c; ax[r] = ci; c = ts; ci = ti; }
            }
        }
        for (int r = 0; r < NK_TOP10_K; r++) {
            topk_indices[qi * NK_TOP10_K + r] = ax[r];
            topk_scores[qi * NK_TOP10_K + r] = mx[r];
        }
    }

    #undef TOP10_UPDATE
}

NK_INTERNAL void nk_maxsim_coarse_argmax_haswell_(        //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_u32_t *best_document_indices) {

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        __m128i running_max_i32x4 = _mm_set1_epi32(NK_I32_MIN);
        __m128i running_argmax_i32x4 = _mm_setzero_si128();

        // 4Q×4D document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x8[query_tile_index][document_tile_index] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i query_biased_u8x32_0 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_u8x32_1 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_u8x32_2 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_u8x32_3 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);

                __m256i document_i8x32, products_i16x16, products_i32x8;

                // Document 0
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_0, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][0] = _mm256_add_epi32(accumulator_tiles_i32x8[0][0], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_1, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][0] = _mm256_add_epi32(accumulator_tiles_i32x8[1][0], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_2, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][0] = _mm256_add_epi32(accumulator_tiles_i32x8[2][0], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_3, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][0] = _mm256_add_epi32(accumulator_tiles_i32x8[3][0], products_i32x8);

                // Document 1
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_0, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][1] = _mm256_add_epi32(accumulator_tiles_i32x8[0][1], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_1, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][1] = _mm256_add_epi32(accumulator_tiles_i32x8[1][1], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_2, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][1] = _mm256_add_epi32(accumulator_tiles_i32x8[2][1], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_3, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][1] = _mm256_add_epi32(accumulator_tiles_i32x8[3][1], products_i32x8);

                // Document 2
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_0, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][2] = _mm256_add_epi32(accumulator_tiles_i32x8[0][2], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_1, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][2] = _mm256_add_epi32(accumulator_tiles_i32x8[1][2], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_2, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][2] = _mm256_add_epi32(accumulator_tiles_i32x8[2][2], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_3, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][2] = _mm256_add_epi32(accumulator_tiles_i32x8[3][2], products_i32x8);

                // Document 3
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_0, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][3] = _mm256_add_epi32(accumulator_tiles_i32x8[0][3], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_1, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][3] = _mm256_add_epi32(accumulator_tiles_i32x8[1][3], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_2, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][3] = _mm256_add_epi32(accumulator_tiles_i32x8[2][3], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32_3, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][3] = _mm256_add_epi32(accumulator_tiles_i32x8[3][3], products_i32x8);
            }

            // Reduce each query's 4 doc accumulators -> __m128i
            __m128i query_0_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[0][0], accumulator_tiles_i32x8[0][1], accumulator_tiles_i32x8[0][2],
                accumulator_tiles_i32x8[0][3]);
            __m128i query_1_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[1][0], accumulator_tiles_i32x8[1][1], accumulator_tiles_i32x8[1][2],
                accumulator_tiles_i32x8[1][3]);
            __m128i query_2_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[2][0], accumulator_tiles_i32x8[2][1], accumulator_tiles_i32x8[2][2],
                accumulator_tiles_i32x8[2][3]);
            __m128i query_3_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[3][0], accumulator_tiles_i32x8[3][1], accumulator_tiles_i32x8[3][2],
                accumulator_tiles_i32x8[3][3]);

            // Bias correction: subtract 128 × sum_quantized for each document
            __m128i bias_correction_i32x4 = _mm_set_epi32(
                128 * document_metadata[document_block_start_index + 3].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 2].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 1].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 0].sum_i8_i32);
            query_0_coarse_dots_i32x4 = _mm_sub_epi32(query_0_coarse_dots_i32x4, bias_correction_i32x4);
            query_1_coarse_dots_i32x4 = _mm_sub_epi32(query_1_coarse_dots_i32x4, bias_correction_i32x4);
            query_2_coarse_dots_i32x4 = _mm_sub_epi32(query_2_coarse_dots_i32x4, bias_correction_i32x4);
            query_3_coarse_dots_i32x4 = _mm_sub_epi32(query_3_coarse_dots_i32x4, bias_correction_i32x4);

            // 4x4 transpose: [query][doc] -> [doc][query] for vectorized argmax
            __m128i transpose_queries_01_low_i32x4 = _mm_unpacklo_epi32(query_0_coarse_dots_i32x4,
                                                                        query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_low_i32x4 = _mm_unpacklo_epi32(query_2_coarse_dots_i32x4,
                                                                        query_3_coarse_dots_i32x4);
            __m128i transpose_queries_01_high_i32x4 = _mm_unpackhi_epi32(query_0_coarse_dots_i32x4,
                                                                         query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_high_i32x4 = _mm_unpackhi_epi32(query_2_coarse_dots_i32x4,
                                                                         query_3_coarse_dots_i32x4);
            __m128i document_0_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_1_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_2_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);
            __m128i document_3_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);

            // Branchless SIMD argmax
            __m128i comparison_mask_i32x4, document_index_i32x4;

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_0_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 0));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_0_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_1_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 1));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_1_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_2_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 2));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_2_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_3_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 3));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_3_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        // Document tail: 4Q×1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            __m256i accumulator_i32x8_0 = _mm256_setzero_si256();
            __m256i accumulator_i32x8_1 = _mm256_setzero_si256();
            __m256i accumulator_i32x8_2 = _mm256_setzero_si256();
            __m256i accumulator_i32x8_3 = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));
                __m256i products_i16x16, products_i32x8;

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_i32x8_0 = _mm256_add_epi32(accumulator_i32x8_0, products_i32x8);

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_i32x8_1 = _mm256_add_epi32(accumulator_i32x8_1, products_i32x8);

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_i32x8_2 = _mm256_add_epi32(accumulator_i32x8_2, products_i32x8);

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_i32x8_3 = _mm256_add_epi32(accumulator_i32x8_3, products_i32x8);
            }

            __m128i reduced_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(accumulator_i32x8_0, accumulator_i32x8_1,
                                                                      accumulator_i32x8_2, accumulator_i32x8_3);
            nk_i32_t bias_correction_i32 = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i coarse_dots_i32x4 = _mm_sub_epi32(reduced_i32x4, _mm_set1_epi32(bias_correction_i32));

            __m128i comparison_mask_i32x4 = _mm_cmpgt_epi32(coarse_dots_i32x4, running_max_i32x4);
            __m128i document_index_i32x4 = _mm_set1_epi32((int)document_index);
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, coarse_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        best_document_indices[query_block_start_index + 0] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 0);
        best_document_indices[query_block_start_index + 1] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 1);
        best_document_indices[query_block_start_index + 2] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 2);
        best_document_indices[query_block_start_index + 3] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 3);
    }

    // Query tail: 1Q×1D
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            __m256i accumulator_i32x8 = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));
                __m256i query_biased_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256((__m256i const *)(query_i8_row + depth_index)), xor_mask_u8x32);
                __m256i products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32, document_i8x32);
                __m256i products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_i32x8 = _mm256_add_epi32(accumulator_i32x8, products_i32x8);
            }

            // Horizontal sum of 8 i32 lanes
            __m128i sum_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_i32x8),
                                              _mm256_extracti128_si256(accumulator_i32x8, 1));
            sum_i32x4 = _mm_add_epi32(sum_i32x4, _mm_shuffle_epi32(sum_i32x4, 0x4E)); // 01001110
            sum_i32x4 = _mm_add_epi32(sum_i32x4, _mm_shuffle_epi32(sum_i32x4, 0xB1)); // 10110001
            nk_i32_t coarse_dot_i32 = _mm_extract_epi32(sum_i32x4, 0) -
                                      128 * document_metadata[document_index].sum_i8_i32;

            if (coarse_dot_i32 > running_max_i32) {
                running_max_i32 = coarse_dot_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        best_document_indices[query_index] = running_argmax_u32;
    }
}

NK_PUBLIC void nk_maxsim_packed_bf16_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                         regions.document_quantized, regions.document_metadata, chunk_size,
                                         document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_bf16((nk_bf16_t const *)(regions.query_originals +
                                            (chunk_start + query_index) * regions.query_original_stride),
                        (nk_bf16_t const *)(regions.document_originals +
                                            best_document_index * regions.document_original_stride),
                        depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

NK_PUBLIC void nk_maxsim_packed_f32_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    if (0 && document_count > 300) {
        /* Inline top-6 with 4Q×4D tiling, refine all 6 candidates */
        for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
            nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
            nk_u32_t topk_indices[256 * NK_TOP6_K];

            nk_maxsim_coarse_top6_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                            regions.document_quantized, regions.document_metadata, chunk_size,
                                            document_count, regions.depth_i8_padded, topk_indices);

            for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
                nk_f32_t const *qf = (nk_f32_t const *)(regions.query_originals +
                                      (chunk_start + query_index) * regions.query_original_stride);
                nk_f64_t best_cosine = -1e30;
                for (int c = 0; c < NK_TOP6_K; c++) {
                    nk_u32_t di = topk_indices[query_index * NK_TOP6_K + c];
                    nk_f64_t dot_result;
                    nk_dot_f32(qf, (nk_f32_t const *)(regions.document_originals +
                                   di * regions.document_original_stride), depth, &dot_result);
                    nk_f64_t cosine = dot_result *
                        (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                        (nk_f64_t)regions.document_metadata[di].inverse_norm_f32;
                    if (cosine > best_cosine) best_cosine = cosine;
                }
                nk_f64_t angular = 1.0 - best_cosine;
                if (angular < 0.0) angular = 0.0;
                total_angular_distance += angular;
            }
        }
    } else {
        /* Standard i8 argmax + single f32 refine for text */
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
                nk_f64_t cosine = dot_result * (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                                  (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
                nk_f64_t angular = 1.0 - cosine;
                if (angular < 0.0) angular = 0.0;
                total_angular_distance += angular;
            }
        }
    }

    *result = total_angular_distance;
}

/**
 *  @brief Per-token MaxSim with Haswell AVX2 coarse screening.
 *  Returns per-query-token angular distances in result[query_count].
 */
NK_PUBLIC void nk_maxsim_packed_per_token_f32_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);

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
                (nk_f32_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride),
                (nk_f32_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f64_t cosine = dot_result *
                              (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f64_t angular = 1.0 - cosine;
            if (angular < 0.0) angular = 0.0;
            result[chunk_start + query_index] = angular;
        }
    }
}

/**
 * 4-bit VPSHUFB MaxSim: near-maddubs speed with Lloyd-Max optimal quantization.
 *
 * Vectors stored as packed 4-bit nibbles (2 dims per byte, 64 bytes for d=128).
 * The i8 region of the packed format holds these nibble-packed bytes.
 * A 16-byte centroid_i8 table maps each nibble value (0-15) to an i8 centroid.
 *
 * The kernel:
 *   1. Loads the 16-byte centroid table into a YMM register (broadcast)
 *   2. For each (query, doc) pair:
 *      a. Load 32 packed bytes (64 nibbles = 64 dims)
 *      b. Split into low/high nibbles
 *      c. VPSHUFB to look up i8 centroid values (32 lookups/cycle!)
 *      d. Interleave back to original dim order
 *      e. maddubs on the expanded i8 values
 *   3. Track argmax + accumulate (sum, sum_sq)
 *
 * Speed: ~1.5× maddubs (the nibble unpack + vpshufb adds ~50%)
 * Accuracy: 64.8% per-token argmax (vs 27.5% NK uniform, 95% 8-bit centroid)
 *
 * centroid_i8_table: 16 int8_t values, the i8-scaled Lloyd-Max centroids for 4-bit.
 * Packed format: i8 region holds nibble-packed bytes (depth_packed = depth/2).
 *   Byte k holds: dim[2k] in low nibble (bits 0-3), dim[2k+1] in high nibble (bits 4-7).
 *
 * Strategy: 4-bit coarse argmax → f32 refine (top-2).
 *   1. Use VPSHUFB to expand 4-bit nibbles and compute i8 dot products for all doc tokens
 *   2. Track top-2 doc token indices by coarse dot score
 *   3. Compute exact f32 dot product for those 2 tokens using original f32 vectors
 *   4. Return angular distance (1 - max(cos1, cos2)) for Serfling bounds
 *
 * This gives 4-bit coarse speed with f32-quality scores for elimination bounds.
 */
NK_PUBLIC void nk_maxsim_packed_4bit_stats_f32_haswell(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result,
    nk_i8_t const *centroid_i8_table) {

    nk_maxsim_packed_regions_t rg = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    /* Broadcast the 16-byte centroid table into both lanes of a YMM register */
    __m128i ctable_128 = _mm_loadu_si128((__m128i const *)centroid_i8_table);
    __m256i ctable = _mm256_broadcastsi128_si256(ctable_128);
    __m256i lo_mask = _mm256_set1_epi8(0x0F);

    /* XOR mask for signed→unsigned conversion (maddubs needs unsigned first arg) */
    __m256i xor_mask = _mm256_set1_epi8((char)0x80);

    nk_size_t depth_packed = depth / 2;  /* 64 bytes for d=128 */

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        nk_u8_t const *q_packed = (nk_u8_t const *)(rg.query_quantized + qi * rg.depth_i8_padded);

        /* Expand query nibbles to i8 centroid values (one-time per query token) */
        nk_i8_t q_expanded[256];  /* max depth, stack allocated */
        for (nk_size_t k = 0; k < depth_packed; k += 32) {
            nk_size_t chunk = depth_packed - k < 32 ? depth_packed - k : 32;
            __m256i packed;
            if (chunk == 32)
                packed = _mm256_loadu_si256((__m256i const *)(q_packed + k));
            else {
                packed = _mm256_setzero_si256();
                for (nk_size_t b = 0; b < chunk; b++)
                    ((nk_u8_t *)&packed)[b] = q_packed[k + b];
            }
            __m256i nibble_lo = _mm256_and_si256(packed, lo_mask);
            __m256i nibble_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask);
            __m256i val_lo = _mm256_shuffle_epi8(ctable, nibble_lo);  /* 32 lookups, 1 cycle! */
            __m256i val_hi = _mm256_shuffle_epi8(ctable, nibble_hi);
            __m256i even = _mm256_unpacklo_epi8(val_lo, val_hi);
            __m256i odd  = _mm256_unpackhi_epi8(val_lo, val_hi);
            _mm256_storeu_si256((__m256i *)(q_expanded + 2*k), even);
            _mm256_storeu_si256((__m256i *)(q_expanded + 2*k + 32), odd);
        }

        /* Phase 1: 4-bit coarse scan → find top-2 doc token indices */
        nk_i32_t best1_dot = -2147483647, best2_dot = -2147483647;
        nk_u32_t best1_idx = 0, best2_idx = 0;

        for (nk_size_t di = 0; di < document_count; di++) {
            nk_u8_t const *d_packed = (nk_u8_t const *)(rg.document_quantized + di * rg.depth_i8_padded);

            __m256i acc = _mm256_setzero_si256();
            __m256i d_sum_acc = _mm256_setzero_si256();
            __m256i ones_i16 = _mm256_set1_epi16(1);
            __m256i ones_i8 = _mm256_set1_epi8(1);

            for (nk_size_t k = 0; k < depth_packed; k += 32) {
                nk_size_t chunk = depth_packed - k < 32 ? depth_packed - k : 32;
                __m256i d_nibbles;
                if (chunk == 32)
                    d_nibbles = _mm256_loadu_si256((__m256i const *)(d_packed + k));
                else {
                    d_nibbles = _mm256_setzero_si256();
                    for (nk_size_t b = 0; b < chunk; b++)
                        ((nk_u8_t *)&d_nibbles)[b] = d_packed[k + b];
                }

                __m256i d_lo = _mm256_shuffle_epi8(ctable, _mm256_and_si256(d_nibbles, lo_mask));
                __m256i d_hi = _mm256_shuffle_epi8(ctable, _mm256_and_si256(_mm256_srli_epi16(d_nibbles, 4), lo_mask));
                __m256i d_even = _mm256_unpacklo_epi8(d_lo, d_hi);
                __m256i d_odd  = _mm256_unpackhi_epi8(d_lo, d_hi);

                /* Bias sum for maddubs correction */
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

            /* Horizontal sum */
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

            /* Track top-2 by coarse 4-bit score */
            if (dot_i32 > best1_dot) {
                best2_dot = best1_dot; best2_idx = best1_idx;
                best1_dot = dot_i32;   best1_idx = (nk_u32_t)di;
            } else if (dot_i32 > best2_dot) {
                best2_dot = dot_i32;   best2_idx = (nk_u32_t)di;
            }
        }

        /* Phase 2: f32 refine on top-2 candidates using original vectors */
        nk_f32_t const *qf = (nk_f32_t const *)(rg.query_originals + qi * rg.query_original_stride);
        nk_f64_t inv_qnorm = (nk_f64_t)rg.query_metadata[qi].inverse_norm_f32;

        nk_f32_t const *d1f = (nk_f32_t const *)(rg.document_originals + best1_idx * rg.document_original_stride);
        nk_f32_t const *d2f = (nk_f32_t const *)(rg.document_originals + best2_idx * rg.document_original_stride);

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

        nk_f64_t cos1 = (nk_f64_t)_mm_cvtss_f32(s1) * inv_qnorm *
                        (nk_f64_t)rg.document_metadata[best1_idx].inverse_norm_f32;
        nk_f64_t cos2 = (nk_f64_t)_mm_cvtss_f32(s2) * inv_qnorm *
                        (nk_f64_t)rg.document_metadata[best2_idx].inverse_norm_f32;
        nk_f64_t best_cosine = cos1 > cos2 ? cos1 : cos2;
        nk_f64_t angular = 1.0 - best_cosine;
        if (angular < 0.0) angular = 0.0;

        total_sum += angular;
        total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

/**
 * Centroid-lookup MaxSim: uses AVX2 gather to compute dot products from
 * Lloyd-Max quantized vectors. Vectors are stored as u8 centroid indices.
 * The centroid table (256 f32 values) maps indices to optimal quantization levels.
 *
 * Accuracy: ~95% per-token argmax (vs 27% for NK i8 uniform on MM).
 * Speed: ~5-10x slower than i8 (gather is expensive). Use for refine, not bulk.
 *
 * query/doc i8 regions contain u8 centroid indices (0-255), not true i8 values.
 * The centroid_table[256] maps these to f32 values.
 */
NK_PUBLIC void nk_maxsim_packed_centroid_stats_f32_haswell(
    void const *query_packed, void const *document_packed,
    nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result,
    float const *centroid_table) {

    nk_maxsim_packed_regions_t rg = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    for (nk_size_t qi = 0; qi < query_count; qi++) {
        /* Query centroid indices (stored as i8, interpret as u8) */
        nk_u8_t const *q_idx = (nk_u8_t const *)(rg.query_quantized + qi * rg.depth_i8_padded);

        nk_f64_t best_dot = -1e30;

        for (nk_size_t di = 0; di < document_count; di++) {
            nk_u8_t const *d_idx = (nk_u8_t const *)(rg.document_quantized + di * rg.depth_i8_padded);

            /* AVX2 gather-based f32 dot product: 8 dims per iteration */
            __m256 acc = _mm256_setzero_ps();
            nk_size_t k = 0;

            for (; k + 8 <= depth; k += 8) {
                /* Load 8 u8 indices, zero-extend to 32-bit for gather */
                __m128i q_u8 = _mm_loadl_epi64((__m128i const *)(q_idx + k));
                __m256i q_i32 = _mm256_cvtepu8_epi32(q_u8);
                __m256 q_f32 = _mm256_i32gather_ps(centroid_table, q_i32, 4);

                __m128i d_u8 = _mm_loadl_epi64((__m128i const *)(d_idx + k));
                __m256i d_i32 = _mm256_cvtepu8_epi32(d_u8);
                __m256 d_f32 = _mm256_i32gather_ps(centroid_table, d_i32, 4);

                acc = _mm256_fmadd_ps(q_f32, d_f32, acc);
            }

            /* Horizontal sum */
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            float dot = _mm_cvtss_f32(s);

            /* Scalar tail */
            for (; k < depth; k++)
                dot += centroid_table[q_idx[k]] * centroid_table[d_idx[k]];

            if ((nk_f64_t)dot > best_dot)
                best_dot = (nk_f64_t)dot;
        }

        /* Convert to angular distance (assumes unit-norm centroid vectors — approximate) */
        nk_f64_t angular = 1.0 - best_dot;
        if (angular < 0.0) angular = 0.0;
        total_sum += angular;
        total_sum_sq += angular * angular;
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

/**
 *  @brief MaxSim with Welford stats: returns (sum, sum_sq) in result[0..1].
 *  Same i8 coarse screening + f32 refinement as the sum kernel, but also
 *  accumulates sum-of-squares for Serfling variance estimation.
 *  Cost: ~identical to sum kernel (one extra FMA per query token).
 */
NK_PUBLIC void nk_maxsim_packed_stats_f32_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_sum = 0.0;
    nk_f64_t total_sum_sq = 0.0;

    /* Stats kernel: use fast argmax for warmup speed.
       Top-6 accuracy comes from the rescore section (sum kernel on final K survivors).
       Top-4 here costs ~1.6× with marginal accuracy gain — disabled. */
    if (0 && document_count > 300) {
        /* Top-6 with fused 6-dot: load Q once, 6 accumulators in one pass.
           Disabled: top-2 is better tradeoff (less overhead, higher vsNK agreement). */
        for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
            nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
            nk_u32_t topk_indices[256 * NK_TOP6_K];

            nk_maxsim_coarse_top6_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                            regions.document_quantized, regions.document_metadata, chunk_size,
                                            document_count, regions.depth_i8_padded, topk_indices);

            for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
                nk_f32_t const *qf = (nk_f32_t const *)(regions.query_originals +
                                      (chunk_start + query_index) * regions.query_original_stride);
                nk_f64_t inv_qnorm = (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32;
                nk_u32_t *cands = &topk_indices[query_index * NK_TOP6_K];

                /* Fused 6-dot: load Q once, compute all 6 dots simultaneously */
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
                nk_f32_t const *d0p = (nk_f32_t const *)(regions.document_originals + cands[0] * regions.document_original_stride);
                nk_f32_t const *d1p = (nk_f32_t const *)(regions.document_originals + cands[1] * regions.document_original_stride);
                nk_f32_t const *d2p = (nk_f32_t const *)(regions.document_originals + cands[2] * regions.document_original_stride);
                nk_f32_t const *d3p = (nk_f32_t const *)(regions.document_originals + cands[3] * regions.document_original_stride);
                nk_f32_t const *d4p = (nk_f32_t const *)(regions.document_originals + cands[4] * regions.document_original_stride);
                nk_f32_t const *d5p = (nk_f32_t const *)(regions.document_originals + cands[5] * regions.document_original_stride);
                for (nk_size_t k = 0; k < depth; k += 8) {
                    __m256 q = _mm256_loadu_ps(qf + k);
                    a0 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d0p + k), a0);
                    a1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1p + k), a1);
                    a2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2p + k), a2);
                    a3 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d3p + k), a3);
                    a4 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d4p + k), a4);
                    a5 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d5p + k), a5);
                }
                /* Horizontal sum each accumulator */
                #define HSUM256(v) ({ \
                    __m128 _lo = _mm256_castps256_ps128(v), _hi = _mm256_extractf128_ps(v, 1); \
                    __m128 _s = _mm_add_ps(_lo, _hi); \
                    _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
                    _mm_cvtss_f32(_mm_add_ss(_s, _mm_shuffle_ps(_s, _s, 1))); \
                })
                nk_f32_t dots[6] = { HSUM256(a0), HSUM256(a1), HSUM256(a2),
                                     HSUM256(a3), HSUM256(a4), HSUM256(a5) };
                #undef HSUM256

                nk_f64_t best_cosine = -1e30;
                for (int c = 0; c < 6; c++) {
                    nk_f64_t cosine = (nk_f64_t)dots[c] * inv_qnorm *
                                      (nk_f64_t)regions.document_metadata[cands[c]].inverse_norm_f32;
                    if (cosine > best_cosine) best_cosine = cosine;
                }
                nk_f64_t angular = 1.0 - best_cosine;
                if (angular < 0.0) angular = 0.0;
                total_sum += angular;
                total_sum_sq += angular * angular;
            }
        }
    } else if (document_count > 0) {
        /* Top-2 with fused 2-dot for ALL doc lengths.
           ~2% overhead, improves i8 argmax accuracy everywhere. */
        for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
            nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
            nk_u32_t best_indices[256];
            nk_u32_t second_indices[256];
            nk_maxsim_coarse_top2_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                            regions.document_quantized, regions.document_metadata, chunk_size,
                                            document_count, regions.depth_i8_padded,
                                            best_indices, second_indices);
            for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
                nk_f32_t const *qf = (nk_f32_t const *)(regions.query_originals +
                                      (chunk_start + query_index) * regions.query_original_stride);
                nk_f32_t const *d1 = (nk_f32_t const *)(regions.document_originals +
                                      best_indices[query_index] * regions.document_original_stride);
                nk_f32_t const *d2 = (nk_f32_t const *)(regions.document_originals +
                                      second_indices[query_index] * regions.document_original_stride);
                __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
                for (nk_size_t k = 0; k < depth; k += 8) {
                    __m256 q = _mm256_loadu_ps(qf + k);
                    acc1 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d1 + k), acc1);
                    acc2 = _mm256_fmadd_ps(q, _mm256_loadu_ps(d2 + k), acc2);
                }
                __m128 lo1 = _mm256_castps256_ps128(acc1), hi1 = _mm256_extractf128_ps(acc1, 1);
                __m128 lo2 = _mm256_castps256_ps128(acc2), hi2 = _mm256_extractf128_ps(acc2, 1);
                __m128 s1 = _mm_add_ps(lo1, hi1), s2 = _mm_add_ps(lo2, hi2);
                s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1, s1, 1));
                s2 = _mm_add_ps(s2, _mm_movehl_ps(s2, s2));
                s2 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
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
    } else {
        /* Standard i8 argmax + f32 refine for text (L_d <= 300) */
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
                nk_f64_t cosine = dot_result * (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                                  (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
                nk_f64_t angular = 1.0 - cosine;
                if (angular < 0.0) angular = 0.0;
                total_sum += angular;
                total_sum_sq += angular * angular;
            }
        }
    }

    result[0] = total_sum;
    result[1] = total_sum_sq;
}

NK_PUBLIC void nk_maxsim_packed_f16_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_haswell_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                         regions.document_quantized, regions.document_metadata, chunk_size,
                                         document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_f16(
                (nk_f16_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride),
                (nk_f16_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_MAXSIM_HASWELL_H
