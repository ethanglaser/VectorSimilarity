/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"
#include <cmath>
static inline void powerStep(float *&pVect1, __m512 &sumPowerReg) {

    __m512 v1 = _mm512_loadu_ps(pVect1);

    pVect1 += 16;

    // sumPowerReg = (v1*v1)+sumPowerReg
    sumPowerReg = _mm512_fmadd_ps(v1, v1, sumPowerReg);
}

static inline void divStep(float *&pVect1, __m512 &normFactor) {

    __m512 v1 = _mm512_loadu_ps(pVect1);

    _mm512_storeu_ps(pVect1, _mm512_div_ps(v1, normFactor));

    pVect1 += 16;
}

// residual:  512/32 = 16
template <unsigned char residual> // 0..15
static void FP32_normalizeSIMD16_AVX512(void *pVect1v, size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    const float *pEnd1 = pVect1 + dimension;
    __m512 sumPowerReg = _mm512_setzero_ps();

    // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
    // so mask loading is guaranteed to be safe
    if constexpr (residual) {
        __mmask16 constexpr mask = (1 << residual) - 1;
        __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
        pVect1 += residual;
        sumPowerReg = _mm512_mul_ps(v1, v1);
    }
    do {
        powerStep(pVect1, sumPowerReg);
    } while (pVect1 < pEnd1);

    pVect1 = (float *)pVect1v;

    float sumOfPower = _mm512_reduce_add_ps(sumPowerReg);
    __m512 normFactor = _mm512_sqrt_ps(_mm512_set1_ps(sumOfPower));

    if constexpr (residual) {
        __mmask16 constexpr mask8 = (1 << (residual)) - 1;
        __m512 v1 = _mm512_loadu_ps(pVect1);
        _mm512_mask_storeu_ps(pVect1, mask8, _mm512_div_ps(v1, normFactor));
        pVect1 += residual;
    }
    do {
        divStep(pVect1, normFactor);
    } while (pVect1 < pEnd1);
}