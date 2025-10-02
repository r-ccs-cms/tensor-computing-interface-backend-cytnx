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

  template <>
  real_t<cytnx::Tensor> norm(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    // Use Cytnx's built-in Norm function for Frobenius norm
    auto norm_result = cytnx::linalg::Norm(a);

    // Extract the scalar value with explicit cast
    if (norm_result.dtype() == cytnx::Type.Double) {
      return static_cast<double>(norm_result.at({0}).real());
    } else if (norm_result.dtype() == cytnx::Type.ComplexDouble) {
      return static_cast<double>(norm_result.at({0}).real());  // Norm should always be real
    } else {
      // Convert to double first
      auto converted = norm_result.astype(cytnx::Type.Double);
      return static_cast<double>(converted.at({0}).real());
    }
  }

  // contract implementations moved to header for template visibility

  // svd implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  template <>
  elem_t<cytnx::Tensor> normalize(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout) {
    // Calculate norm
    auto original_norm = norm(ctx, inout);

    // Normalize by dividing by norm
    if (original_norm > 0.0) {
      inout = inout / cytnx::cytnx_complex128(original_norm, 0.0);
    }

    return cytnx::cytnx_complex128(original_norm, 0.0);
  }

  template <> elem_t<cytnx::Tensor> normalize(context_handle_t<cytnx::Tensor>& ctx,
                                              const cytnx::Tensor& in, cytnx::Tensor& out) {
    out = in.clone();
    return normalize(ctx, out);
  }

  template <> void scale(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
                         const elem_t<cytnx::Tensor> s) {
    // Convert variant to Cytnx-compatible type
    auto complex_val = tci::to_complex128(s);
    inout = inout * complex_val;
  }

  template <> void scale(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                         const elem_t<cytnx::Tensor> s, cytnx::Tensor& out) {
    // Convert variant to Cytnx-compatible type
    auto complex_val = tci::to_complex128(s);
    out = in * complex_val;
  }

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

  template <> void diag(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout) {
    auto shape = inout.shape();

    if (shape.size() == 1) {
      // rank-1 → rank-2 diagonal matrix
      auto n = shape[0];
      cytnx::Tensor diag_matrix = cytnx::zeros({n, n}, inout.dtype(), ctx);

      // Set diagonal elements
      for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
        auto val = inout.at({i});
        diag_matrix.at({i, i}) = val;
      }

      inout = std::move(diag_matrix);
    } else if (shape.size() == 2 && shape[0] == shape[1]) {
      // rank-2 square matrix → rank-1 diagonal vector
      auto n = shape[0];
      cytnx::Tensor diag_vector = cytnx::zeros({n}, inout.dtype(), ctx);

      // Extract diagonal elements
      for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
        auto val = inout.at({i, i});
        diag_vector.at({i}) = val;
      }

      inout = std::move(diag_vector);
    } else {
      throw std::invalid_argument("diag: input must be rank-1 vector or square rank-2 matrix");
    }
  }

  template <>
  void diag(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in, cytnx::Tensor& out) {
    out = in.clone();
    diag(ctx, out);
  }

  template <> void trace(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
                         const bond_idx_pairs_t<cytnx::Tensor>& bdidx_pairs) {
    if (bdidx_pairs.empty()) {
      return;  // No trace to perform
    }

    auto shape = inout.shape();

    // Validate bond index pairs
    for (const auto& [idx1, idx2] : bdidx_pairs) {
      if (idx1 >= shape.size() || idx2 >= shape.size()) {
        throw std::invalid_argument("trace: bond index out of range");
      }
      if (shape[idx1] != shape[idx2]) {
        throw std::invalid_argument("trace: bond dimensions must match for tracing");
      }
    }

    // For now, implement simple matrix trace (rank-2 case)
    if (shape.size() == 2 && bdidx_pairs.size() == 1) {
      auto [idx1, idx2] = bdidx_pairs[0];
      if ((idx1 == 0 && idx2 == 1) || (idx1 == 1 && idx2 == 0)) {
        // Standard matrix trace
        cytnx::cytnx_complex128 trace_sum = cytnx::cytnx_complex128(0.0, 0.0);
        auto n = std::min(shape[0], shape[1]);

        for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
          auto elem = inout.at({i, i});
          trace_sum += cytnx::cytnx_complex128(static_cast<double>(elem.real()),
                                               static_cast<double>(elem.imag()));
        }

        // Create scalar tensor with trace result without zero-dimension initialization issues
        cytnx::Tensor scalar = cytnx::zeros({1}, inout.dtype(), ctx);
        scalar.at({0}) = trace_sum;
        scalar.reshape_({});
        inout = std::move(scalar);
        return;
      }
    }

    // General case: use Cytnx's built-in Trace function
    cytnx::Tensor result = inout.clone();

    // Process each bond pair sequentially using Cytnx Trace
    for (const auto& [idx1, idx2] : bdidx_pairs) {
      auto current_shape = result.shape();

      if (idx1 >= current_shape.size() || idx2 >= current_shape.size()) {
        throw std::invalid_argument("trace: bond index out of range");
      }
      if (current_shape[idx1] != current_shape[idx2]) {
        throw std::invalid_argument("trace: bond dimensions must match for tracing");
      }

      // Use Cytnx's Trace function for proper tensor trace
      result = cytnx::linalg::Trace(result, static_cast<cytnx::cytnx_uint64>(idx1),
                                             static_cast<cytnx::cytnx_uint64>(idx2));

      // For multiple pairs, we would need to adjust indices after each trace
      // For now, handle only the first pair
      break;
    }

    inout = std::move(result);
  }

  template <> void trace(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                         const bond_idx_pairs_t<cytnx::Tensor>& bdidx_pairs, cytnx::Tensor& out) {
    out = in.clone();
    trace(ctx, out, bdidx_pairs);
  }

  // trunc_svd implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // qr implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  // lq implementation moved to include/tci/tensor_linear_algebra_impl.h
  // (Backend Unification Pattern)

  template <> void eigvals(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                           const rank_t<cytnx::Tensor> num_of_bds_as_row,
                           cplx_ten_t<cytnx::Tensor>& w_diag) {
    auto shape = a.shape();

    // Calculate row and column dimensions based on num_of_bds_as_row
    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row && i < shape.size(); ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigvals: matrix must be square");
    }

    // Reshape tensor to matrix form
    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eig_result = cytnx::linalg::Eig(matrix);  // [eigenvalues, eigenvectors]
    w_diag = eig_result[0];

    if (w_diag.shape().size() != 1) {
      w_diag.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }

    if (w_diag.dtype() != cytnx::Type.ComplexDouble) {
      w_diag = w_diag.astype(cytnx::Type.ComplexDouble);
    }
  }

  template <> void eigvalsh(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                            const rank_t<cytnx::Tensor> num_of_bds_as_row,
                            real_ten_t<cytnx::Tensor>& w_diag) {
    auto shape = a.shape();

    // Calculate row and column dimensions based on num_of_bds_as_row
    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row && i < shape.size(); ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigvalsh: matrix must be square");
    }

    // Reshape tensor to matrix form
    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eigh_result = cytnx::linalg::Eigh(matrix);
    w_diag = eigh_result[0];

    if (w_diag.shape().size() != 1) {
      w_diag.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }
  }

  template <> void eig(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                       const rank_t<cytnx::Tensor> num_of_bds_as_row,
                       cplx_ten_t<cytnx::Tensor>& w_diag, cplx_ten_t<cytnx::Tensor>& v) {
    auto shape = a.shape();

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row && i < shape.size(); ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eig: matrix must be square");
    }

    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eig_result = cytnx::linalg::Eig(matrix);  // [eigenvalues, eigenvectors]
    w_diag = eig_result[0];
    v = eig_result[1];

    if (w_diag.shape().size() != 1) {
      w_diag.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }
    if (w_diag.dtype() != cytnx::Type.ComplexDouble) {
      w_diag = w_diag.astype(cytnx::Type.ComplexDouble);
    }

    if (v.shape().size() != 2) {
      v.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                  static_cast<cytnx::cytnx_int64>(row_dim)});
    }
    if (v.dtype() != cytnx::Type.ComplexDouble) {
      v = v.astype(cytnx::Type.ComplexDouble);
    }
  }

  template <> void eigh(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                        const rank_t<cytnx::Tensor> num_of_bds_as_row,
                        real_ten_t<cytnx::Tensor>& w_diag, cytnx::Tensor& v) {
    auto shape = a.shape();

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row && i < shape.size(); ++i) {
      row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      col_dim *= shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigh: matrix must be square");
    }

    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eigh_result = cytnx::linalg::Eigh(matrix);  // [eigenvalues, eigenvectors]
    w_diag = eigh_result[0];
    v = eigh_result[1];

    if (w_diag.shape().size() != 1) {
      w_diag.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }

    if (v.shape().size() != 2) {
      v.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                  static_cast<cytnx::cytnx_int64>(row_dim)});
    }
  }

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
