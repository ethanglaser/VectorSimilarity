
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/norm.h"
#include "VecSim/spaces/normalize/normalize_naive.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/SSE3.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

spaces::normalizeVector_f<float>
spaces::norm_FP32_GetNormalizeFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    spaces::normalizeVector_f<float> ret_normalize_func = spaces::normalizeVector_imp<float>;
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_normalize_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 8 doubles
        return Choose_FP32_normalize_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        return Choose_FP32_normalize_implementation_AVX(dim);
    }
#endif
#endif // __x86_64__
    return ret_normalize_func;
}

spaces::normalizeVector_f<double>
spaces::norm_FP64_GetNormalizeFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    spaces::normalizeVector_f<double> ret_normalize_func = spaces::normalizeVector_imp<double>;
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 8) {
        return ret_normalize_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);

#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        return Choose_FP64_normalize_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx) {
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 8 floats
        return Choose_FP64_normalize_implementation_AVX(dim);
    }
#endif
#endif // __x86_64__
    return ret_normalize_func;
}