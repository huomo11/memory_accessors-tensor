#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include <stddef.h>
#include <stdint.h>
#include <vector>

template <typename T>
class DenseMatrix {
public:
    DenseMatrix() : rows_(0), cols_(0) {}

    DenseMatrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    T* data() { return data_.empty() ? 0 : &data_[0]; }
    const T* data() const { return data_.empty() ? 0 : &data_[0]; }

    T& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    std::vector<T>& values() { return data_; }
    const std::vector<T>& values() const { return data_; }

private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;
};

#endif
