#pragma once

#include <cytnx.hpp>
#include <stdexcept>
#include <vector>

#include "tci/tensor_linear_algebra.h"
#include "tci/tensor_traits.h"
#include "tci/cytnx_tensor_traits.h"

namespace tci {

  // Backend implementations for cytnx::Tensor (type-erased runtime version)
  // These implementations are moved from source/tensor_linear_algebra.cpp
  // to make them visible to frontend (CytnxTensor) for delegation

  /**
   * @brief Truncated SVD implementation for cytnx::Tensor (Backend)
   *
   * This is the single source of truth for trunc_svd logic.
   * Frontend (CytnxTensor) should delegate to this implementation.
   */
  template <>
  inline void trunc_svd(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                        const rank_t<cytnx::Tensor> num_of_bds_as_row, cytnx::Tensor& u,
                        real_ten_t<cytnx::Tensor>& s_diag, cytnx::Tensor& v_dag,
                        real_t<cytnx::Tensor>& trunc_err,
                        const bond_dim_t<cytnx::Tensor> chi_max,
                        const real_t<cytnx::Tensor> s_min) {
    (void)ctx;

    auto shape = a.shape();
    if (shape.size() < num_of_bds_as_row) {
      throw std::invalid_argument("trunc_svd: num_of_bds_as_row exceeds tensor rank");
    }

    cytnx::cytnx_uint64 row_dim = 1, col_dim = 1;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    cytnx::Tensor matrix = a.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                     static_cast<cytnx::cytnx_int64>(col_dim)});

    // Use Cytnx Svd_truncate with parameters:
    // keepdim = chi_max (maximum singular values to keep)
    // err = s_min (threshold for small singular values)
    // is_UvT = true (return U and V†)
    // return_err = 0 (do NOT return truncation error - causes ASAN issues)
    // mindim = 1 (minimum number of singular values to keep)
    std::vector<cytnx::Tensor> svd_result = cytnx::linalg::Svd_truncate(
        matrix, static_cast<cytnx::cytnx_uint64>(chi_max), s_min, true, 0, 1);

    if (svd_result.size() < 3) {
      throw std::runtime_error("trunc_svd: unexpected result size from Svd_truncate");
    }

    s_diag = svd_result[0];  // Singular values
    u = svd_result[1];       // Left unitary matrix U
    v_dag = svd_result[2];   // Right unitary matrix V†

    // Calculate truncation error manually
    // trunc_err is set to the minimum kept singular value (following Cytnx convention)
    auto s_shape = s_diag.shape();
    if (s_shape.size() > 0 && s_shape[0] > 0) {
      // Get the minimum singular value (last element)
      if (s_diag.dtype() == cytnx::Type.Double) {
        auto* s_data = s_diag.ptr_as<double>();
        trunc_err = s_data[s_shape[0] - 1];
      } else if (s_diag.dtype() == cytnx::Type.Float) {
        auto* s_data = s_diag.ptr_as<float>();
        trunc_err = static_cast<double>(s_data[s_shape[0] - 1]);
      } else {
        trunc_err = 0.0;
      }
    } else {
      trunc_err = 0.0;
    }

    // Reshape results to match original tensor structure
    // Always reshape U to tensor structure (not just when != 2D)
    auto u_shape_vec = u.shape();
    std::vector<cytnx::cytnx_int64> new_u_shape;

    // For U matrix: row dimensions from original tensor + kept singular values
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      new_u_shape.push_back(static_cast<cytnx::cytnx_int64>(shape[i]));
    }
    new_u_shape.push_back(u_shape_vec[1]);  // Number of kept singular values

    u.reshape_(new_u_shape);

    // Always reshape V† to tensor structure (not just when != 2D)
    auto v_dag_shape_vec = v_dag.shape();
    std::vector<cytnx::cytnx_int64> new_v_dag_shape;

    // For V† matrix: kept singular values + column dimensions from original tensor
    new_v_dag_shape.push_back(v_dag_shape_vec[0]);  // Number of kept singular values
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      new_v_dag_shape.push_back(static_cast<cytnx::cytnx_int64>(shape[i]));
    }

    v_dag.reshape_(new_v_dag_shape);
  }

}  // namespace tci
