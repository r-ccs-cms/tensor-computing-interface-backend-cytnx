#include "tci/tensor_linear_algebra.h"
#include "tci/tensor_linear_algebra_impl.h"

#include <cytnx.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include "tci/cytnx_tensor_traits.h"
#include "tci/variant_helpers.h"

namespace tci {

  // Template specializations for linear algebra functions using Cytnx

  // norm implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // contract implementations moved to header for template visibility

  // svd implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // normalize implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // scale implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  template <> void linear_combine(context_handle_t<cytnx::Tensor>& ctx,
                                  const List<cytnx::Tensor>& ins, cytnx::Tensor& out) {
    if (ins.empty()) return;

    out = ins[0].clone();
    for (size_t i = 1; i < ins.size(); ++i) {
      out = out + ins[i];
    }
  }

  template <> void linear_combine(context_handle_t<cytnx::Tensor>& ctx,
                                  const List<cytnx::Tensor>& ins,
                                  const List<elem_t<cytnx::Tensor>>& coefs, cytnx::Tensor& out) {
    if (ins.empty() || coefs.empty()) return;

    // Convert variant to Cytnx-compatible type
    auto coef0 = tci::to_complex128(coefs[0]);
    out = ins[0] * coef0;
    for (size_t i = 1; i < std::min(ins.size(), coefs.size()); ++i) {
      auto coefi = tci::to_complex128(coefs[i]);
      out = out + (ins[i] * coefi);
    }
  }

  // diag implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // trace implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // trunc_svd implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // qr implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // lq implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // eigvals implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // eigvalsh implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // eig implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // eigh implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // ===== Matrix Exponential =====

  template <>
  void exp(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
           const rank_t<cytnx::Tensor> num_of_bds_as_row) {
    auto shape = inout.shape();
    if (shape.size() < num_of_bds_as_row) {
      throw std::invalid_argument("exp: num_of_bds_as_row exceeds tensor rank");
    }

    cytnx::cytnx_uint64 row_dim = 1, col_dim = 1;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("exp: matrix must be square");
    }

    auto original_shape = shape;
    inout.reshape_({static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Use Cytnx ExpM for general matrix exponential
    inout = cytnx::linalg::ExpM(inout);

    inout.reshape_(original_shape);
  }

  template <>
  void exp(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
           const rank_t<cytnx::Tensor> num_of_bds_as_row, cytnx::Tensor& out) {
    auto shape = in.shape();
    if (shape.size() < num_of_bds_as_row) {
      throw std::invalid_argument("exp: num_of_bds_as_row exceeds tensor rank");
    }

    cytnx::cytnx_uint64 row_dim = 1, col_dim = 1;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("exp: matrix must be square");
    }

    cytnx::Tensor matrix = in.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Use Cytnx ExpM for general matrix exponential
    out = cytnx::linalg::ExpM(matrix);

    out.reshape_(shape);
  }

}  // namespace tci
