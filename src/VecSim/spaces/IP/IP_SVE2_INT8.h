/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"
 #include <arm_sve.h>

 // Perform dot product of 8-bit vectors using svdot_s32 and accumulate into 32-bit lanes
 static inline void InnerProductStep_SVE2(const int8_t *&pVect1, const int8_t *&pVect2,
                                          svint32_t &sum) {
     svbool_t pg = svptrue_b8();
     svint8_t v1 = svld1_s8(pg, pVect1);
     svint8_t v2 = svld1_s8(pg, pVect2);
     sum = svdot_s32(pg, sum, v1, v2);
     pVect1 += svcntb();
     pVect2 += svcntb();
 }

 template <bool partial_chunk, unsigned char additional_steps>
 float INT8_InnerProductImp_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
     const int8_t *pVect1 = reinterpret_cast<const int8_t *>(pVect1v);
     const int8_t *pVect2 = reinterpret_cast<const int8_t *>(pVect2v);

     // Vector length in bytes (number of int8 elements per vector)
     const size_t vl = svcntb();

     // Process 4 full vectors per iteration (unrolled loop)
     const size_t chunk_size = 4 * vl;
     const size_t main_size = (dimension / chunk_size) * chunk_size;

     svint32_t sum0 = svdup_s32(0);
     svint32_t sum1 = svdup_s32(0);
     svint32_t sum2 = svdup_s32(0);
     svint32_t sum3 = svdup_s32(0);

     size_t offset = 0;
     size_t num_main_blocks = dimension / chunk_size;

     for (size_t i = 0; i < num_main_blocks; ++i) {
         InnerProductStep_SVE2(pVect1, pVect2, sum0);
         InnerProductStep_SVE2(pVect1, pVect2, sum1);
         InnerProductStep_SVE2(pVect1, pVect2, sum2);
         InnerProductStep_SVE2(pVect1, pVect2, sum3);
         offset += chunk_size;
     }

     // Handle remainder blocks after main loop
     if constexpr (additional_steps > 0) {
         for (unsigned char c = 0; c < additional_steps; ++c) {
             InnerProductStep_SVE2(pVect1, pVect2, sum0);
             offset += vl;
         }
     }

     // Handle final partial vector (tail) using predicate
     if constexpr (partial_chunk) {
         svbool_t pg = svwhilelt_b8(offset, dimension);
         svint8_t v1 = svld1_s8(pg, reinterpret_cast<const int8_t *>(pVect1v) + offset);
         svint8_t v2 = svld1_s8(pg, reinterpret_cast<const int8_t *>(pVect2v) + offset);
         sum0 = svdot_s32(pg, sum0, v1, v2);
     }

     // Combine the four partial sums
     sum0 = svadd_s32_x(svptrue_b32(), sum0, sum1);
     sum2 = svadd_s32_x(svptrue_b32(), sum2, sum3);
     sum0 = svadd_s32_x(svptrue_b32(), sum0, sum2);

     // Horizontal sum of the lanes
     int32_t result = svaddv_s32(svptrue_b32(), sum0);

     // Return result as float
     return static_cast<float>(result);
 }

 template <bool partial_chunk, unsigned char additional_steps>
 float INT8_InnerProductSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
     return 1.0f - INT8_InnerProductImp_SVE2<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);
 }

 template <bool partial_chunk, unsigned char additional_steps>
 float INT8_CosineSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
     // Inner product between vectors
     float ip = INT8_InnerProductImp_SVE2<partial_chunk, additional_steps>(pVect1v, pVect2v, dimension);

     // Read precomputed norms (stored after the vector)
     float norm_v1 = *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect1v) + dimension);
     float norm_v2 = *reinterpret_cast<const float *>(static_cast<const int8_t *>(pVect2v) + dimension);

     // Return cosine distance
     return 1.0f - ip / (norm_v1 * norm_v2);
 }
