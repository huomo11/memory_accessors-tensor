#pragma once

#include <cmath>
#include <vector>

inline double checksum_dense(const std::vector<double>& x) {
  double sum = 0.0;
  for (std::size_t n = 0; n < x.size(); ++n) {
    sum += x[n] * (1.0 + static_cast<double>((n % 17) + 1) * 1.0e-6);
  }
  return sum;
}
