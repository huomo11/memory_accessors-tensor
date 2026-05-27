#include "csv_writer.hpp"
#include "csr_matrix.hpp"
#include "dense_matrix.hpp"
#include "metrics.hpp"
#include "precision_policy.hpp"
#include "sparse_tensor.hpp"
#include "spmm_backend.hpp"
#include "timer.hpp"

#include <errno.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Options {
    std::string output;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    std::vector<uint64_t> nnzs;
    std::vector<int> ranks;
    std::vector<int> modes;
    int threads;
    int repeats;
    uint64_t seed;
    std::vector<VariantKind> selected_variants;

    Options()
        : output("results/mkl_three_variants.csv"), dim0(16), dim1(12), dim2(10),
          threads(1), repeats(1), seed(42) {}
};

static bool parse_long_checked(const char* text, long* value) {
    char* end = 0;
    errno = 0;
    long v = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = v;
    return true;
}

static bool parse_ull_checked(const char* text, uint64_t* value) {
    char* end = 0;
    errno = 0;
    unsigned long long v = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = (uint64_t)v;
    return true;
}

static std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    size_t i;
    for (i = 0; i < s.size(); ++i) {
        if (s[i] == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(s[i]);
        }
    }
    out.push_back(cur);
    return out;
}

static bool parse_int_list(const std::string& text, std::vector<int>* out) {
    std::vector<std::string> parts = split_commas(text);
    out->clear();
    size_t i;
    for (i = 0; i < parts.size(); ++i) {
        long v = 0;
        if (!parse_long_checked(parts[i].c_str(), &v)) {
            return false;
        }
        out->push_back((int)v);
    }
    return true;
}

static bool parse_ull_list(const std::string& text, std::vector<uint64_t>* out) {
    std::vector<std::string> parts = split_commas(text);
    out->clear();
    size_t i;
    for (i = 0; i < parts.size(); ++i) {
        uint64_t v = 0;
        if (!parse_ull_checked(parts[i].c_str(), &v)) {
            return false;
        }
        out->push_back(v);
    }
    return true;
}

static bool variant_from_name(const std::string& name, VariantKind* variant) {
    if (name == "mkl_fp64") {
        *variant = VAR_MKL_FP64;
    } else if (name == "mkl_fp32") {
        *variant = VAR_MKL_FP32;
    } else if (name == "mkl_mixed_factor_fp32_storage_fp64_compute") {
        *variant = VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE;
    } else {
        return false;
    }
    return true;
}

static bool parse_variant_list(const std::string& text, std::vector<VariantKind>* out) {
    std::vector<std::string> parts = split_commas(text);
    out->clear();
    size_t i;
    for (i = 0; i < parts.size(); ++i) {
        VariantKind variant = VAR_MKL_FP64;
        if (!variant_from_name(parts[i], &variant)) {
            return false;
        }
        out->push_back(variant);
    }
    return true;
}

static bool has_variant(const std::vector<VariantKind>& variants, VariantKind target) {
    size_t i;
    for (i = 0; i < variants.size(); ++i) {
        if (variants[i] == target) {
            return true;
        }
    }
    return false;
}

static bool parse_dims(const std::string& text, uint32_t* d0, uint32_t* d1, uint32_t* d2) {
    std::vector<std::string> parts = split_commas(text);
    if (parts.size() != 3) {
        return false;
    }
    long vals[3];
    int i;
    for (i = 0; i < 3; ++i) {
        if (!parse_long_checked(parts[(size_t)i].c_str(), &vals[i]) || vals[i] <= 0) {
            return false;
        }
    }
    *d0 = (uint32_t)vals[0];
    *d1 = (uint32_t)vals[1];
    *d2 = (uint32_t)vals[2];
    return true;
}

static void usage() {
    std::cerr << "Usage: ttm_bench --output out.csv --dims I,J,K --nnz N "
              << "--ranks r1,r2 --modes 0,1,2 --threads 1 --repeats R "
              << "--seed S --variants mkl_fp64,mkl_fp32,mkl_mixed_factor_fp32_storage_fp64_compute\n";
}

static bool parse_args(int argc, char** argv, Options* opt) {
    int i = 1;
    while (i < argc) {
        if (i + 1 >= argc) {
            return false;
        }
        std::string key(argv[i]);
        std::string val(argv[i + 1]);
        if (key == "--output") {
            opt->output = val;
        } else if (key == "--dims") {
            if (!parse_dims(val, &opt->dim0, &opt->dim1, &opt->dim2)) {
                return false;
            }
        } else if (key == "--nnz") {
            if (!parse_ull_list(val, &opt->nnzs)) {
                return false;
            }
        } else if (key == "--ranks") {
            if (!parse_int_list(val, &opt->ranks)) {
                return false;
            }
        } else if (key == "--modes") {
            if (!parse_int_list(val, &opt->modes)) {
                return false;
            }
        } else if (key == "--threads") {
            long tmp = 0;
            if (!parse_long_checked(val.c_str(), &tmp) || tmp < 1) {
                return false;
            }
            opt->threads = (int)tmp;
        } else if (key == "--repeats") {
            long tmp = 0;
            if (!parse_long_checked(val.c_str(), &tmp) || tmp < 1) {
                return false;
            }
            opt->repeats = (int)tmp;
        } else if (key == "--seed") {
            if (!parse_ull_checked(val.c_str(), &opt->seed)) {
                return false;
            }
        } else if (key == "--variants") {
            if (!parse_variant_list(val, &opt->selected_variants)) {
                return false;
            }
        } else {
            return false;
        }
        i += 2;
    }
    if (opt->nnzs.empty()) {
        opt->nnzs.push_back(1000);
    }
    if (opt->ranks.empty()) {
        opt->ranks.push_back(4);
    }
    if (opt->modes.empty()) {
        opt->modes.push_back(0);
    }
    if (opt->selected_variants.empty()) {
        opt->selected_variants.push_back(VAR_MKL_FP64);
        opt->selected_variants.push_back(VAR_MKL_FP32);
        opt->selected_variants.push_back(VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE);
    }
    return true;
}

static DenseMatrix<double> generate_factor_double(size_t rows, size_t cols, uint64_t seed) {
    DenseMatrix<double> f(rows, cols);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    size_t i;
    for (i = 0; i < f.size(); ++i) {
        f.values()[i] = dist(rng);
    }
    return f;
}

template <typename OutT, typename InT>
static DenseMatrix<OutT> convert_factor(const DenseMatrix<InT>& in) {
    DenseMatrix<OutT> out(in.rows(), in.cols());
    size_t i;
    for (i = 0; i < in.size(); ++i) {
        out.values()[i] = (OutT)in.values()[i];
    }
    return out;
}

static MetricsRow make_metric_base(const Options& opt, const CsrMatrix& csr,
                                   VariantKind variant, int rank, int mode,
                                   int repeat, double format_ms, SpMMBackend backend) {
    MetricsRow row;
    uint64_t nnz = (uint64_t)csr.nnz();
    uint64_t factor_elems = (uint64_t)csr.ncols * (uint64_t)rank;
    uint64_t output_elems = (uint64_t)csr.nrows * (uint64_t)rank;
    size_t value_size = variant_value_storage_size(variant);
    size_t factor_size = variant_factor_storage_size(variant);
    size_t factor_compute_size = variant_factor_compute_read_size(variant);
    size_t output_size = variant_output_storage_size(variant);

    row.format_prepare_ms = format_ms;
    row.index_storage_bytes = ((uint64_t)csr.nrows + 1ULL) * (uint64_t)sizeof(size_t) +
        nnz * (uint64_t)sizeof(uint32_t);
    row.value_storage_bytes = nnz * (uint64_t)value_size;
    row.factor_storage_bytes = factor_elems * (uint64_t)factor_size;
    row.output_storage_bytes = output_elems * (uint64_t)output_size;
    row.index_logical_read_bytes = nnz * (uint64_t)sizeof(uint32_t);
    row.value_logical_read_bytes = nnz * (uint64_t)value_size;
    row.factor_logical_read_bytes = nnz * (uint64_t)rank * (uint64_t)factor_size;
    row.factor_compute_logical_read_bytes = nnz * (uint64_t)rank * (uint64_t)factor_compute_size;
    {
        uint64_t streamed = nnz * (uint64_t)rank * (uint64_t)output_size;
        row.output_logical_write_bytes = row.output_storage_bytes > streamed
            ? row.output_storage_bytes : streamed;
    }
    row.rank = rank;
    row.nnz = nnz;
    row.mode = mode;
    row.thread_count = opt.threads;
    row.seed = opt.seed;
    row.variant = variant_name(variant);
    row.dim0 = opt.dim0;
    row.dim1 = opt.dim1;
    row.dim2 = opt.dim2;
    row.repeat = repeat;
    row.backend = spmm_backend_name(backend);
    row.csr_nrows = csr.nrows;
    row.csr_ncols = csr.ncols;
    row.csr_nnz = nnz;
    return row;
}

static MetricsRow run_mkl_fp64(const Options& opt, const CsrMatrix& csr,
                               const DenseMatrix<double>& factor64, int rank,
                               int mode, int repeat, double format_ms,
                               SpMMBackend backend, std::vector<double>* out) {
    MetricsRow row = make_metric_base(opt, csr, VAR_MKL_FP64, rank, mode, repeat,
                                      format_ms, backend);
    Timer timer;
    backend_csr_spmm_fp64(backend, csr, factor64, (size_t)rank, out, false);
    row.compute_ms = timer.elapsed_ms();
    row.upcast_prepare_ms = 0.0;
    row.kernel_ms = row.compute_ms;
    row.total_ms = row.format_prepare_ms + row.kernel_ms;
    row.rel_error = 0.0;
    return row;
}

static MetricsRow run_mkl_fp32(const Options& opt, const CsrMatrix& csr64,
                               const CsrMatrixF& csr32,
                               const DenseMatrix<double>& factor64,
                               const std::vector<double>& baseline,
                               int rank, int mode, int repeat,
                               double format_ms, SpMMBackend backend) {
    MetricsRow row = make_metric_base(opt, csr64, VAR_MKL_FP32, rank, mode, repeat,
                                      format_ms, backend);
    DenseMatrix<float> factor32 = convert_factor<float>(factor64);
    std::vector<float> out32;
    Timer timer;
    backend_csr_spmm_fp32(backend, csr32, factor32, (size_t)rank, &out32, false);
    row.compute_ms = timer.elapsed_ms();
    row.upcast_prepare_ms = 0.0;
    row.kernel_ms = row.compute_ms;
    row.total_ms = row.format_prepare_ms + row.kernel_ms;
    std::vector<double> out64;
    copy_as_double(out32, &out64);
    row.rel_error = relative_frobenius_error(out64, baseline);
    return row;
}

static MetricsRow run_mkl_mixed(const Options& opt, const CsrMatrix& csr,
                                const DenseMatrix<double>& factor64,
                                const std::vector<double>& baseline,
                                int rank, int mode, int repeat,
                                double format_ms, SpMMBackend backend) {
    MetricsRow row = make_metric_base(opt, csr,
                                      VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE,
                                      rank, mode, repeat, format_ms, backend);
    DenseMatrix<float> factor32 = convert_factor<float>(factor64);
    Timer timer;
    DenseMatrix<double> factor_workspace = convert_factor<double>(factor32);
    row.upcast_prepare_ms = timer.elapsed_ms();
    std::vector<double> out64;
    timer.reset();
    backend_csr_spmm_fp64(backend, csr, factor_workspace, (size_t)rank, &out64, false);
    row.compute_ms = timer.elapsed_ms();
    row.kernel_ms = row.upcast_prepare_ms + row.compute_ms;
    row.total_ms = row.format_prepare_ms + row.kernel_ms;
    row.rel_error = relative_frobenius_error(out64, baseline);
    return row;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage();
        return 2;
    }

    CsvWriter csv(opt.output);
    if (!csv.good()) {
        std::cerr << "Failed to open output CSV: " << opt.output << "\n";
        return 1;
    }

    SpMMBackend backend = choose_default_backend();
    size_t ni;
    for (ni = 0; ni < opt.nnzs.size(); ++ni) {
        SparseTensorCOO x = generate_random_tensor(opt.dim0, opt.dim1, opt.dim2,
                                                   (size_t)opt.nnzs[ni], opt.seed + (uint64_t)ni);
        size_t mi;
        for (mi = 0; mi < opt.modes.size(); ++mi) {
            int mode = opt.modes[mi];
            if (mode < 0 || mode > 2) {
                std::cerr << "Invalid mode: " << mode << "\n";
                return 2;
            }

            Timer prep_timer;
            CsrMatrix csr64 = build_csr_fp64_from_tensor(x, mode);
            CsrMatrixF csr32;
            if (has_variant(opt.selected_variants, VAR_MKL_FP32)) {
                csr32 = build_csr_fp32_from_tensor(x, mode);
            }
            double format_ms = prep_timer.elapsed_ms();

            size_t ri;
            for (ri = 0; ri < opt.ranks.size(); ++ri) {
                int rank = opt.ranks[ri];
                if (rank <= 0) {
                    std::cerr << "Invalid rank: " << rank << "\n";
                    return 2;
                }
                DenseMatrix<double> factor64 =
                    generate_factor_double((size_t)csr64.ncols, (size_t)rank,
                                           opt.seed + (uint64_t)mode * 1000003ULL + (uint64_t)rank);

                int rep;
                for (rep = 0; rep < opt.repeats; ++rep) {
                    std::vector<double> baseline;
                    MetricsRow base = run_mkl_fp64(opt, csr64, factor64, rank, mode,
                                                   rep, format_ms, backend, &baseline);

                    if (has_variant(opt.selected_variants, VAR_MKL_FP64)) {
                        csv.write(base);
                    }
                    if (has_variant(opt.selected_variants, VAR_MKL_FP32)) {
                        csv.write(run_mkl_fp32(opt, csr64, csr32, factor64, baseline,
                                               rank, mode, rep, format_ms, backend));
                    }
                    if (has_variant(opt.selected_variants,
                                    VAR_MKL_MIXED_FACTOR_FP32_STORAGE_FP64_COMPUTE)) {
                        csv.write(run_mkl_mixed(opt, csr64, factor64, baseline,
                                                rank, mode, rep, format_ms, backend));
                    }
                }
            }
        }
    }

    return 0;
}
