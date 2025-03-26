/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_neon.h>

// Perform dot product on 16 int8 elements and accumulate into int32 lanes
static inline void InnerProductStep_NEON(const int8_t *&pVect1, const int8_t *&pVect2,
                                         int32x4_t &acc) {
    int8x16_t a = vld1q_s8(pVect1);
    int8x16_t b = vld1q_s8(pVect2);

    int16x8_t a_lo = vmovl_s8(vget_low_s8(a));
    int16x8_t b_lo = vmovl_s8(vget_low_s8(b));
    int16x8_t a_hi = vmovl_s8(vget_high_s8(a));
    int16x8_t b_hi = vmovl_s8(vget_high_s8(b));

    int32x4_t prod_lo_0 = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
    int32x4_t prod_lo_1 = vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo));
    int32x4_t prod_hi_0 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
    int32x4_t prod_hi_1 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi));

    acc = vaddq_s32(acc, prod_lo_0);
    acc = vaddq_s32(acc, prod_lo_1);
    acc = vaddq_s32(acc, prod_hi_0);
    acc = vaddq_s32(acc, prod_hi_1);

    pVect1 += 16;
    pVect2 += 16;
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductImp_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    const size_t chunk_size = 4 * 16;
    const size_t main_size = (dimension / chunk_size) * chunk_size;

    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);

    size_t offset = 0;
    size_t num_main_blocks = dimension / chunk_size;

    for (size_t i = 0; i < num_main_blocks; ++i) {
        InnerProductStep_NEON(pVect1, pVect2, sum0);
        InnerProductStep_NEON(pVect1, pVect2, sum1);
        InnerProductStep_NEON(pVect1, pVect2, sum2);
        InnerProductStep_NEON(pVect1, pVect2, sum3);
        offset += chunk_size;
    }

    if constexpr (additional_steps > 0) {
        for (unsigned char c = 0; c < additional_steps; ++c) {
            InnerProductStep_NEON(pVect1, pVect2, sum0);
            offset += 16;
        }
    }

    if constexpr (partial_chunk) {
        int32_t tail_sum = 0;
        for (size_t i = offset; i < dimension; ++i) {
            int a = static_cast<int>(reinterpret_cast<const int8_t *>(pVect1v)[i]);
            int b = static_cast<int>(reinterpret_cast<const int8_t *>(pVect2v)[i]);
            tail_sum += a * b;
        }
        sum0 = vaddq_s32(sum0, vdupq_n_s32(tail_sum));
    }

    sum0 = vaddq_s32(sum0, sum1);
    sum2 = vaddq_s32(sum2, sum3);
    sum0 = vaddq_s32(sum0, sum2);

    int32_t buffer[4];
    vst1q_s32(buffer, sum0);
    int32_t result = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    return static_cast<float>(result);
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductSIMD_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1.0f -
           INT8_InnerProductImp_NEON<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_CosineSIMD_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float ip =
        INT8_InnerProductImp_NEON<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);

    float norm_v1 =
        *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect1v) + dimension);
    float norm_v2 =
        *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect2v) + dimension);

    return 1.0f - ip / (norm_v1 * norm_v2);
}
