#pragma once

// Check shared by the backend tests of svd, trunc_svd, eig and eigh, whose
// sigma / lambda_mat outputs TCAPI V1 makes second-order diagonal tensors.

#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

// Requires `t` to be a square second-order tensor, checks that every
// off-diagonal element is exactly zero and that every diagonal element's
// imaginary part is (near) zero, and returns the real parts of the diagonal for
// the caller to check.
template <typename Ctx, typename Ten> std::vector<double> require_diagonal(Ctx& ctx, const Ten& t) {
  const auto shape = tci::shape(ctx, t);
  REQUIRE(shape.size() == 2);
  REQUIRE(shape[0] == shape[1]);
  const auto n = static_cast<cytnx::cytnx_uint64>(shape[0]);
  std::vector<double> diagonal;
  std::size_t nonzero_off_diagonal = 0;
  for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
    for (cytnx::cytnx_uint64 j = 0; j < n; ++j) {
      const auto elem = tci::get_elem(ctx, t, {i, j});
      if (i == j) {
        CHECK(static_cast<double>(std::imag(elem)) == doctest::Approx(0.0));
        diagonal.push_back(static_cast<double>(std::real(elem)));
      } else if (std::abs(elem) != 0.0) {
        ++nonzero_off_diagonal;
      }
    }
  }
  CHECK(nonzero_off_diagonal == 0);
  return diagonal;
}
