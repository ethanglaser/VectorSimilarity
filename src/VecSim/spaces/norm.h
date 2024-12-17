/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/spaces/spaces.h"

namespace spaces {
normalizeVector_f<float> norm_FP32_GetNormalizeFunc(size_t dim, unsigned char *alignment = nullptr,
                                                    const void *arch_opt = nullptr);
normalizeVector_f<double> norm_FP64_GetNormalizeFunc(size_t dim, unsigned char *alignment = nullptr,
                                                     const void *arch_opt = nullptr);
} // namespace spaces
