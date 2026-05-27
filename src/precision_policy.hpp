#ifndef PRECISION_POLICY_HPP
#define PRECISION_POLICY_HPP

#include <stddef.h>

enum VariantKind {
    VAR_MKL_FP64 = 0,
    VAR_MKL_FP32 = 1,
    VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE = 2
};

inline const char* variant_name(VariantKind v) {
    switch (v) {
    case VAR_MKL_FP64:
        return "mkl_fp64";
    case VAR_MKL_FP32:
        return "mkl_fp32";
    case VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE:
        return "mkl_mixed_factor_fp32_storage_fp64_compute";
    default:
        return "unknown";
    }
}

inline size_t variant_value_storage_size(VariantKind v) {
    if (v == VAR_MKL_FP32) {
        return sizeof(float);
    }
    return sizeof(double);
}

inline size_t variant_factor_storage_size(VariantKind v) {
    if (v == VAR_MKL_FP32 ||
        v == VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE) {
        return sizeof(float);
    }
    return sizeof(double);
}

inline size_t variant_factor_compute_read_size(VariantKind v) {
    if (v == VAR_MKL_FP32) {
        return sizeof(float);
    }
    return sizeof(double);
}

inline size_t variant_output_storage_size(VariantKind v) {
    if (v == VAR_MKL_FP32) {
        return sizeof(float);
    }
    return sizeof(double);
}

#endif
