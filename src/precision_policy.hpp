#ifndef PRECISION_POLICY_HPP
#define PRECISION_POLICY_HPP

#include <stddef.h>

enum VariantKind {
    VAR_FACTOR_FP64_COMPUTE_FP64 = 0,
    VAR_FACTOR_FP32_COMPUTE_FP64 = 1,
    VAR_FACTOR_FP32_ONFLY_COMPUTE_FP64 = 2,
    VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64 = 3,
    VAR_FACTOR_FP32_2DBLOCKED_COMPUTE_FP64 = 4,
    VAR_FACTOR_FP32_COMPUTE_FP32 = 5,
    VAR_VALUE_FP32_FACTOR_FP64_COMPUTE_FP64 = 6,
    VAR_CSR_FP64_FACTOR_FP64_BACKEND = 7,
    VAR_CSR_FACTOR_FP32_GLOBAL_UPCAST_FP64_BACKEND = 8,
    VAR_CSR_FACTOR_FP32_TILED_ACCESSOR_FP64_BACKEND = 9,
    VAR_CSR_FP32_FACTOR_FP32_BACKEND = 10
};

inline const char* variant_name(VariantKind v) {
    switch (v) {
    case VAR_FACTOR_FP64_COMPUTE_FP64:
        return "factor_fp64_compute_fp64";
    case VAR_FACTOR_FP32_COMPUTE_FP64:
        return "factor_fp32_compute_fp64";
    case VAR_FACTOR_FP32_ONFLY_COMPUTE_FP64:
        return "factor_fp32_onfly_compute_fp64";
    case VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64:
        return "factor_fp32_blocked_compute_fp64";
    case VAR_FACTOR_FP32_2DBLOCKED_COMPUTE_FP64:
        return "factor_fp32_2dblocked_compute_fp64";
    case VAR_FACTOR_FP32_COMPUTE_FP32:
        return "factor_fp32_compute_fp32";
    case VAR_VALUE_FP32_FACTOR_FP64_COMPUTE_FP64:
        return "value_fp32_factor_fp64_compute_fp64";
    case VAR_CSR_FP64_FACTOR_FP64_BACKEND:
        return "mkl_fp64";
    case VAR_CSR_FACTOR_FP32_GLOBAL_UPCAST_FP64_BACKEND:
        return "mkl_mixed_factor_fp32_storage_fp64_compute";
    case VAR_CSR_FACTOR_FP32_TILED_ACCESSOR_FP64_BACKEND:
        return "csr_factor_fp32_tiled_accessor_fp64_backend";
    case VAR_CSR_FP32_FACTOR_FP32_BACKEND:
        return "mkl_fp32";
    default:
        return "unknown";
    }
}

inline size_t variant_value_storage_size(VariantKind v) {
    if (v == VAR_VALUE_FP32_FACTOR_FP64_COMPUTE_FP64 ||
        v == VAR_CSR_FP32_FACTOR_FP32_BACKEND) {
        return sizeof(float);
    }
    return sizeof(double);
}

inline size_t variant_factor_storage_size(VariantKind v) {
    if (v == VAR_FACTOR_FP32_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_ONFLY_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_2DBLOCKED_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_COMPUTE_FP32 ||
        v == VAR_CSR_FACTOR_FP32_GLOBAL_UPCAST_FP64_BACKEND ||
        v == VAR_CSR_FACTOR_FP32_TILED_ACCESSOR_FP64_BACKEND ||
        v == VAR_CSR_FP32_FACTOR_FP32_BACKEND) {
        return sizeof(float);
    }
    return sizeof(double);
}

inline size_t variant_factor_compute_read_size(VariantKind v) {
    if (v == VAR_FACTOR_FP32_ONFLY_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_2DBLOCKED_COMPUTE_FP64 ||
        v == VAR_FACTOR_FP32_COMPUTE_FP32 ||
        v == VAR_CSR_FACTOR_FP32_TILED_ACCESSOR_FP64_BACKEND ||
        v == VAR_CSR_FP32_FACTOR_FP32_BACKEND) {
        return sizeof(float);
    }
    return sizeof(double);
}

inline size_t variant_output_storage_size(VariantKind v) {
    if (v == VAR_FACTOR_FP32_COMPUTE_FP32 ||
        v == VAR_CSR_FP32_FACTOR_FP32_BACKEND) {
        return sizeof(float);
    }
    return sizeof(double);
}

#endif
