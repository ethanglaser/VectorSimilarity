/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

// Aligned step using svptrue_b8()
static inline void InnerProductStep(const int8_t *&pVect1, const int8_t *&pVect2, svfloat32_t &sum) {
    svbool_t pg = svptrue_b8();

    svint8_t v1_i8 = svld1_s8(pg, pVect1);
    svint8_t v2_i8 = svld1_s8(pg, pVect2);

    // Unpack into int16 for precision by dividing the vector into 2 parts.
    // High and Low
    svint16_t v1_lo = svunpklo_s8(v1_i8);
    svint16_t v2_lo = svunpklo_s8(v2_i8);
    svint16_t v1_hi = svunpkhi_s8(v1_i8);
    svint16_t v2_hi = svunpkhi_s8(v2_i8);

    // Multiply and accumulate on low part
    svint32_t prod_lo = svmul_s32_z(pg, svreinterpret_s32(v1_lo), svreinterpret_s32(v2_lo));
    sum = svadd_f32_z(pg, sum, svcvt_f32_s32(prod_lo));

    // Multiply and accumulate on high part
    svint32_t prod_hi = svmul_s32_z(pg, svreinterpret_s32(v1_hi), svreinterpret_s32(v2_hi));
    sum = svadd_f32_z(pg, sum, svcvt_f32_s32(prod_hi));

    pVect1 += svcntb();
    pVect2 += svcntb();
}

template <bool partial_chunk, unsigned char additional_steps>
float INT8_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

    const size_t vl = svcntb(); // number of int8 per SVE register
    const size_t chunk_size = 4 * vl;
    const size_t main_size = (dimension / chunk_size) * chunk_size;

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t offset = 0;
    size_t num_main_blocks = dimension / chunk_size;

    for (size_t i = 0; i < num_main_blocks; ++i) {
        InnerProductStep(pVect1, pVect2, sum0);
        InnerProductStep(pVect1, pVect2, sum1);
        InnerProductStep(pVect1, pVect2, sum2);
        InnerProductStep(pVect1, pVect2, sum3);
        offset += chunk_size;
    }

    if constexpr (additional_steps > 0) {
        for (unsigned char c = 0; c < additional_steps; ++c) {
            InnerProductStep(pVect1, pVect2, sum0);
            offset += vl;
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b8(offset, dimension);

        svint8_t v1_i8 = svld1_s8(pg, reinterpret_cast<const int8_t *>(pVect1v) + offset);
        svint8_t v2_i8 = svld1_s8(pg, reinterpret_cast<const int8_t *>(pVect2v) + offset);

        svint16_t v1_lo = svunpklo_s8(v1_i8);
        svint16_t v2_lo = svunpklo_s8(v2_i8);
        svint16_t v1_hi = svunpkhi_s8(v1_i8);
        svint16_t v2_hi = svunpkhi_s8(v2_i8);

        svint32_t prod_lo = svmul_s32_z(pg, svreinterpret_s32(v1_lo), svreinterpret_s32(v2_lo));
        sum0 = svadd_f32_z(pg, sum0, svcvt_f32_s32(prod_lo));

        svint32_t prod_hi = svmul_s32_z(pg, svreinterpret_s32(v1_hi), svreinterpret_s32(v2_hi));
        sum0 = svadd_f32_z(pg, sum0, svcvt_f32_s32(prod_hi));
    }

    // Combine the partial sums
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
    sum2 = svadd_f32_z(svptrue_b32(), sum2, sum3);
    sum0 = svadd_f32_z(svptrue_b32(), sum0, sum2);

    // Horizontal sum
    return svaddv_f32(svptrue_b32(), sum0);
}
