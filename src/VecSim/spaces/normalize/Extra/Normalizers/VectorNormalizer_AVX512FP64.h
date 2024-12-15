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
    static  void Normalize(double* const v,size_t dim)
    {
        Naive_Normalizer<residual,double>(v,dim);        
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


    static inline void powerStep(double *&pVect1,__m512d &sumPowerReg) {

        __m512d v1 = _mm512_loadu_pd(pVect1);

        pVect1 += 8; 

        //sumPowerReg = (v1*v1)+sumPowerReg
        sumPowerReg =_mm512_fmadd_pd(v1,v1,sumPowerReg);

    }

    static inline void divStep(double *&pVect1,__m512d &normFactor) {

        __m512d v1 = _mm512_loadu_pd(pVect1);

        _mm512_storeu_pd(pVect1,_mm512_div_pd (v1,normFactor));

        pVect1 += 8; 

    }

    //residual:  512/64 = 8
    template <unsigned char residual> // 0..7
    static void Normalize(const void *pVect1v, size_t dimension) {
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


        double sumOfPower = _mm512_reduce_add_pd(sumPowerReg);
        __m512d normFactor = _mm512_sqrt_pd(_mm512_set1_pd(sumOfPower));

        if constexpr (residual) {
            __mmask8 constexpr mask8 = (1 << (residual)) - 1;
            __m512d v1 = _mm512_loadu_pd(pVect1);
            _mm512_mask_storeu_pd(pVect1,mask8,_mm512_div_pd(v1,normFactor));
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


struct SingleScalar
{
    template<typename vectType>
    static bool compareVectors(const std::vector<vectType>& v1, const std::vector<vectType>& v2, int bit_error_margin = 11)
    {
        return v1==v2;
    }


    static inline void powerStep(double *&pVect1,__m512d &sumPowerReg) {

        __m512d v1 = _mm512_loadu_pd(pVect1);

        pVect1 += 8; 

        //sumPowerReg = (v1*v1)+sumPowerReg
        sumPowerReg =_mm512_fmadd_pd(v1,v1,sumPowerReg);

    }

    static inline void divStep(double *&pVect1,__m512d &normFactor) {

        __m512d v1 = _mm512_loadu_pd(pVect1);

        _mm512_storeu_pd(pVect1,_mm512_div_pd (v1,normFactor));

        pVect1 += 8; 

    }

    //residual:  512/64 = 8
    template <unsigned char residual> // 0..7
    static void Normalize(const void *pVect1v, size_t dimension) {
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


        double sumOfPower = _mm512_reduce_add_pd(sumPowerReg);
        __m512d normFactor = _mm512_set1_pd(sqrt(sumOfPower));

        if constexpr (residual) {
            __mmask8 constexpr mask8 = (1 << (residual)) - 1;
            __m512d v1 = _mm512_loadu_pd(pVect1);
            _mm512_mask_storeu_pd(pVect1,mask8,_mm512_div_pd(v1,normFactor));
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

    static inline void powerStep(double *&pVect1,__m512d &sumPowerReg) {

        __m512d v1 = _mm512_loadu_pd(pVect1);

        pVect1 += 8; 

        //sumPowerReg = (v1*v1)+sumPowerReg
        sumPowerReg =_mm512_fmadd_pd(v1,v1,sumPowerReg);

    }

    static inline void divStep(double *&pVect1,__m512d &normFactor) {

        __m512d v1 = _mm512_loadu_pd(pVect1);

        _mm512_storeu_pd(pVect1,_mm512_mul_pd (v1,normFactor));

        pVect1 += 8; 

    }

    //residual:  512/64 = 8
    template <unsigned char residual> // 0..7
    static void Normalize(const void *pVect1v, size_t dimension) {
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


        double sumOfPower = _mm512_reduce_add_pd(sumPowerReg);
        __m512d normFactor = _mm512_rsqrt14_pd(_mm512_set1_pd(sumOfPower));

        if constexpr (residual) {
            __mmask8 constexpr mask8 = (1 << (residual)) - 1;
            __m512d v1 = _mm512_loadu_pd(pVect1);
            _mm512_mask_storeu_pd(pVect1,mask8,_mm512_div_pd(v1,normFactor));
            pVect1 += residual;
        }
        do {
            divStep(pVect1, sumPowerReg);
        } while (pVect1 < pEnd1);
    }
};