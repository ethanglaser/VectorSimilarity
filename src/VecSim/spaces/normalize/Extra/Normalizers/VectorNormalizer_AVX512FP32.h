#pragma once
#include <cmath>
#include "NormalizeUtils.h"
#include "NaiveNormalize.h"
#include <vector>
struct Naive
{   
    template<typename vectType>
    static bool compareVectors(const std::vector<vectType>& v1, const std::vector<vectType>& v2, int bit_error_margin = 11)
    {
        return v1==v2;
    }

    template <unsigned char residual>
    static  void Normalize(float* const v,size_t dim)
    {
        Naive_Normalizer<residual,float>(v,dim);        
    }
};
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

struct SingleScalar
{
    template<typename vectType>
    static bool compareVectors(const std::vector<vectType>& v1, const std::vector<vectType>& v2, int bit_error_margin = 11)
    {
        return v1==v2;
    }


    static inline void powerStep(float *&pVect1,__m512 &sumPowerReg) {

        __m512 v1 = _mm512_loadu_ps(pVect1);

        pVect1 += 16; 

        //sumPowerReg = (v1*v1)+sumPowerReg
        sumPowerReg =_mm512_fmadd_ps(v1,v1,sumPowerReg);

    }

    static inline void divStep(float *&pVect1,__m512 &normFactor) {

        __m512 v1 = _mm512_loadu_ps(pVect1);

        _mm512_storeu_ps(pVect1,_mm512_div_ps(v1,normFactor));

        pVect1 += 16; 

    }

    //residual:  512/32 = 16
    template <unsigned char residual> // 0..15
    static void Normalize(const void *pVect1v, size_t dimension) {
        float *pVect1 = (float *)pVect1v;
        const float *pEnd1 = pVect1 + dimension;
        __m512 sumPowerReg = _mm512_setzero_ps();

        // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
        // so mask loading is guaranteed to be safe
        if constexpr (residual) {
            __mmask8 constexpr mask = (1 << residual) - 1;
            __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
            pVect1 += residual;
            sumPowerReg = _mm512_mul_ps(v1, v1);
        }
        do {
            powerStep(pVect1, sumPowerReg);
        } while (pVect1 < pEnd1);

        pVect1 = (float *)pVect1v;


        float sumOfPower = _mm512_reduce_add_ps(sumPowerReg);
        __m512 normFactor = _mm512_set1_ps(sqrtf(sumOfPower));

        if constexpr (residual) {
            __mmask8 constexpr mask8 = (1 << (residual)) - 1;
            __m512 v1 = _mm512_loadu_ps(pVect1);
            _mm512_mask_storeu_ps(pVect1,mask8,_mm512_div_ps(v1,normFactor));
            pVect1 += residual;
        }
        do {
            divStep(pVect1, sumPowerReg);
        } while (pVect1 < pEnd1);
    }
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

struct SingleSqrt
{
    template<typename vectType>
    static bool compareVectors(const std::vector<vectType>& v1, const std::vector<vectType>& v2, int bit_error_margin = 11)
    {
        return v1==v2;
    }


    static inline void powerStep(float *&pVect1,__m512 &sumPowerReg) {

        __m512 v1 = _mm512_loadu_ps(pVect1);

        pVect1 += 16; 

        //sumPowerReg = (v1*v1)+sumPowerReg
        sumPowerReg =_mm512_fmadd_ps(v1,v1,sumPowerReg);

    }

    static inline void divStep(float *&pVect1,__m512 &normFactor) {

        __m512 v1 = _mm512_loadu_ps(pVect1);

        _mm512_storeu_ps(pVect1,_mm512_div_ps(v1,normFactor));

        pVect1 += 16; 

    }

    //residual:  512/32 = 16
    template <unsigned char residual> // 0..15
    static void Normalize(const void *pVect1v, size_t dimension) {
        float *pVect1 = (float *)pVect1v;
        const float *pEnd1 = pVect1 + dimension;
        __m512 sumPowerReg = _mm512_setzero_ps();

        // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
        // so mask loading is guaranteed to be safe
        if constexpr (residual) {
            __mmask8 constexpr mask = (1 << residual) - 1;
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
            __mmask8 constexpr mask8 = (1 << (residual)) - 1;
            __m512 v1 = _mm512_loadu_ps(pVect1);
            _mm512_mask_storeu_ps(pVect1,mask8,_mm512_div_ps(v1,normFactor));
            pVect1 += residual;
        }
        do {
            divStep(pVect1, sumPowerReg);
        } while (pVect1 < pEnd1);
    }
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

struct SingleRsqrt
{
    template<typename vectType>
    static bool compareVectors(const std::vector<vectType>& v1, const std::vector<vectType>& v2, int bit_error_margin = 11)
    {
        if (v1.size() != v2.size()) {
        std::cerr << "Vector sizes are different!\n";
        return false;
    }

        // Calculate the epsilon for the given bit error margin
        float epsilon = std::ldexp(1.0f, -bit_error_margin); // equivalent to 2^(-bit_error_margin)

        for (size_t i = 0; i < v1.size(); ++i) {
            if (std::fabs(v1[i] - v2[i]) > epsilon) {
                std::cerr << "Mismatch at index " << i << ": " << v1[i] << " vs " << v2[i] << "\n";
                return false;
            }
        }

        return true;
    }


    static inline void powerStep(float *&pVect1,__m512 &sumPowerReg) {

        __m512 v1 = _mm512_loadu_ps(pVect1);

        pVect1 += 16; 

        //sumPowerReg = (v1*v1)+sumPowerReg
        sumPowerReg =_mm512_fmadd_ps(v1,v1,sumPowerReg);

    }

    static inline void divStep(float *&pVect1,__m512 &normFactor) {

        __m512 v1 = _mm512_loadu_ps(pVect1);

        _mm512_storeu_ps(pVect1,_mm512_mul_ps(v1,normFactor));

        pVect1 += 16; 

    }

    //residual:  512/32 = 16
    template <unsigned char residual> // 0..15
    static void Normalize(const void *pVect1v, size_t dimension) {
        float *pVect1 = (float *)pVect1v;
        const float *pEnd1 = pVect1 + dimension;
        __m512 sumPowerReg = _mm512_setzero_ps();

        // Deal with remainder first. `dim` is more than 16, so we have at least one 16-float block,
        // so mask loading is guaranteed to be safe
        if constexpr (residual) {
            __mmask8 constexpr mask = (1 << residual) - 1;
            __m512 v1 = _mm512_maskz_loadu_ps(mask, pVect1);
            pVect1 += residual;
            sumPowerReg = _mm512_mul_ps(v1, v1);
        }
        do {
            powerStep(pVect1, sumPowerReg);
        } while (pVect1 < pEnd1);

        pVect1 = (float *)pVect1v;


        float sumOfPower = _mm512_reduce_add_ps(sumPowerReg);
        __m512 normFactor = _mm512_rsqrt14_ps(_mm512_set1_ps(sumOfPower));

        if constexpr (residual) {
            __mmask8 constexpr mask8 = (1 << (residual)) - 1;
            __m512 v1 = _mm512_loadu_ps(pVect1);
            _mm512_mask_storeu_ps(pVect1,mask8,_mm512_mul_ps(v1,normFactor));
            pVect1 += residual;
        }
        do {
            divStep(pVect1, sumPowerReg);
        } while (pVect1 < pEnd1);
    }
};

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

