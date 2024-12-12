/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"
#include <cmath>

static inline void powerStep(double *&pVect1, __m512d &sumPowerReg) {

    __m512d v1 = _mm512_loadu_pd(pVect1);

    pVect1 += 8;

    // sumPowerReg = (v1*v1)+sumPowerReg
    sumPowerReg = _mm512_fmadd_pd(v1, v1, sumPowerReg);
}

static inline void divStep(float *&pVect1, __m512d &normFactor) {

    __m512d v1 = _mm512_loadu_pd(pVect1);

    _mm512_storeu_pd(pVect1, _mm512_div_pd(v1, normFactor));

    pVect1 += 8;
}

// residual:  512/64 = 8
template <unsigned char residual> // 0..7
double FP64_normalizeSIMD8_AVX512(const void *pVect1v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    const double *pEnd1 = pVect1 + dimension;
    __m512d sumPowerReg = _mm512_setzero_pd();

    // Deal with remainder first. `dim` is more than 8, so we have at least one 8-double block,
    // so mask loading is guaranteed to be safe
    if constexpr (residual) {
        __mmask8 constexpr mask = (1 << residual) - 1;
        __m512d v1 = _mm512_maskz_loadu_pd(mask, pVect1);
        pVect1 += residual;
        sumPowerReg = _mm512_mul_pd(v1, v1);
    }
    do {
        powerStep(pVect1, sumPowerReg);
    } while (pVect1 < pEnd1);

    pVect1 = (double *)pVect1v;
}