#include "tci/tensor_linear_algebra.h"

// #define TCI_DEBUG_CONTRACT

#include <cytnx.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include "tci/cytnx_tensor_traits.h"

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

  namespace {
    // Abnormal NCON analysis: detect and handle mixed positive/negative output labels
    struct NCONAnalysis {
      std::vector<cytnx::cytnx_uint64> contract_axes_a, contract_axes_b;
      std::vector<cytnx::cytnx_uint64> free_axes_a, free_axes_b;
      std::vector<cytnx::cytnx_uint64> output_permutation;
      bool is_abnormal_ncon = false;

      NCONAnalysis(const List<bond_label_t<cytnx::Tensor>>& bd_labs_a,
                   const List<bond_label_t<cytnx::Tensor>>& bd_labs_b,
                   const List<bond_label_t<cytnx::Tensor>>& bd_labs_c) {
        analyze(bd_labs_a, bd_labs_b, bd_labs_c);
      }

    private:
      void analyze(const List<bond_label_t<cytnx::Tensor>>& bd_labs_a,
                   const List<bond_label_t<cytnx::Tensor>>& bd_labs_b,
                   const List<bond_label_t<cytnx::Tensor>>& bd_labs_c) {
        // Find contracted indices (appear in both a and b, not in c)
        std::set<cytnx::cytnx_int64> labels_a(bd_labs_a.begin(), bd_labs_a.end());
        std::set<cytnx::cytnx_int64> labels_b(bd_labs_b.begin(), bd_labs_b.end());
        std::set<cytnx::cytnx_int64> labels_c(bd_labs_c.begin(), bd_labs_c.end());

        // Check for abnormal NCON: output labels mixing positive and negative
        bool has_positive = false, has_negative = false;
        for (const auto& label : bd_labs_c) {
          if (label > 0) has_positive = true;
          if (label < 0) has_negative = true;
        }
        is_abnormal_ncon = has_positive && has_negative;

        // Find contract axes
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (labels_b.count(label) && !labels_c.count(label)) {
            contract_axes_a.push_back(static_cast<cytnx::cytnx_uint64>(i));
          } else if (labels_c.count(label)) {
            free_axes_a.push_back(static_cast<cytnx::cytnx_uint64>(i));
          }
        }

        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          auto label = bd_labs_b[i];
          if (labels_a.count(label) && !labels_c.count(label)) {
            contract_axes_b.push_back(static_cast<cytnx::cytnx_uint64>(i));
          } else if (labels_c.count(label)) {
            free_axes_b.push_back(static_cast<cytnx::cytnx_uint64>(i));
          }
        }

        // Calculate output permutation to match bd_labs_c order
        calculate_output_permutation(bd_labs_a, bd_labs_b, bd_labs_c);
      }

      void calculate_output_permutation(const List<bond_label_t<cytnx::Tensor>>& bd_labs_a,
                                        const List<bond_label_t<cytnx::Tensor>>& bd_labs_b,
                                        const List<bond_label_t<cytnx::Tensor>>& bd_labs_c) {
        // Create mapping from output label to desired position
        std::map<cytnx::cytnx_int64, size_t> target_pos;
        for (size_t i = 0; i < bd_labs_c.size(); ++i) {
          target_pos[bd_labs_c[i]] = i;
        }

        // Build permutation array
        output_permutation.clear();
        size_t current_pos = 0;

        // Add free axes from tensor a
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (target_pos.count(label)) {
            while (output_permutation.size() <= target_pos[label]) {
              output_permutation.push_back(-1);
            }
            output_permutation[target_pos[label]] = current_pos;
          }
          if (std::find(contract_axes_a.begin(), contract_axes_a.end(), i)
              == contract_axes_a.end()) {
            current_pos++;
          }
        }

        // Add free axes from tensor b
        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          auto label = bd_labs_b[i];
          if (target_pos.count(label)) {
            while (output_permutation.size() <= target_pos[label]) {
              output_permutation.push_back(-1);
            }
            output_permutation[target_pos[label]] = current_pos;
          }
          if (std::find(contract_axes_b.begin(), contract_axes_b.end(), i)
              == contract_axes_b.end()) {
            current_pos++;
          }
        }
      }
    };
  }  // namespace

  template <> void contract(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                            const List<bond_label_t<cytnx::Tensor>>& bd_labs_a,
                            const cytnx::Tensor& b,
                            const List<bond_label_t<cytnx::Tensor>>& bd_labs_b, cytnx::Tensor& c,
                            const List<bond_label_t<cytnx::Tensor>>& bd_labs_c) {
    const auto rank_a = a.shape().size();
    const auto rank_b = b.shape().size();

    bool treat_as_label_mode
        = (bd_labs_a.size() == rank_a) && (bd_labs_b.size() == rank_b);

    const auto in_range = [](size_t rank, const List<bond_label_t<cytnx::Tensor>>& axes_list) {
      for (auto axis : axes_list) {
        if (axis < 0 || static_cast<size_t>(axis) >= rank) {
          return false;
        }
      }
      return true;
    };

    if (!in_range(rank_a, bd_labs_a) || !in_range(rank_b, bd_labs_b)) {
      treat_as_label_mode = true;
    }

    if (bd_labs_a.size() > rank_a || bd_labs_b.size() > rank_b) {
      treat_as_label_mode = true;
    }

    if (treat_as_label_mode) {
      NCONAnalysis analysis(bd_labs_a, bd_labs_b, bd_labs_c);

      if (analysis.contract_axes_a.empty() && analysis.contract_axes_b.empty()) {
        auto flatten = [](const cytnx::Tensor& tensor, bool row_vector) {
          cytnx::Tensor flat = tensor.clone();
          cytnx::cytnx_uint64 total
              = std::accumulate(tensor.shape().begin(), tensor.shape().end(),
                                static_cast<cytnx::cytnx_uint64>(1),
                                std::multiplies<cytnx::cytnx_uint64>());
          if (row_vector) {
            flat.reshape_({1, static_cast<cytnx::cytnx_int64>(total)});
          } else {
            flat.reshape_({static_cast<cytnx::cytnx_int64>(total), 1});
          }
          return flat;
        };

        auto a_flat = flatten(a, false);
        auto b_flat = flatten(b, true);
        cytnx::Tensor outer_matrix = cytnx::linalg::Matmul(a_flat, b_flat);

        std::vector<cytnx::cytnx_uint64> target_shape;
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        target_shape.insert(target_shape.end(), a_shape.begin(), a_shape.end());
        target_shape.insert(target_shape.end(), b_shape.begin(), b_shape.end());

        cytnx::Tensor result = outer_matrix.reshape(target_shape);
        if (!analysis.output_permutation.empty() && analysis.is_abnormal_ncon) {
          result = result.permute(analysis.output_permutation);
        }
        c = std::move(result);
        return;
      }

      if (analysis.contract_axes_a.empty() || analysis.contract_axes_b.empty()) {
        throw std::invalid_argument("contract: no valid contraction axes found");
      }

      try {
        cytnx::Tensor result = cytnx::linalg::Tensordot(a, b, analysis.contract_axes_a,
                                                        analysis.contract_axes_b);
        if (!analysis.output_permutation.empty() && analysis.is_abnormal_ncon) {
          result = result.permute(analysis.output_permutation);
        }
        c = std::move(result);
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string("contract: Tensordot failed - ") + e.what());
      }
      return;
    }

    auto convert_axes = [](size_t rank,
                           const List<bond_label_t<cytnx::Tensor>>& axes_list,
                           const char* which) {
      std::vector<cytnx::cytnx_uint64> axes;
      axes.reserve(axes_list.size());
      std::vector<bool> seen(rank, false);
      for (auto axis : axes_list) {
        if (axis < 0 || static_cast<size_t>(axis) >= rank) {
          std::ostringstream oss;
          oss << "contract: axis index out of range for " << which << " (rank=" << rank
              << ", requested=" << axis << ")";
          oss << " | axes list=";
          for (auto v : axes_list) oss << v << ' ';
          throw std::out_of_range(oss.str());
        }
        auto idx = static_cast<size_t>(axis);
        if (seen[idx]) {
          throw std::invalid_argument("contract: duplicate axis index detected");
        }
        seen[idx] = true;
        axes.push_back(static_cast<cytnx::cytnx_uint64>(idx));
      }
      return axes;
    };

    auto contract_axes_a = convert_axes(rank_a, bd_labs_a, "first tensor");
    auto contract_axes_b = convert_axes(rank_b, bd_labs_b, "second tensor");

#ifdef TCI_DEBUG_CONTRACT
    std::cerr << "contract axis-mode | rank_a=" << rank_a << " rank_b=" << rank_b << "\n";
    std::cerr << "  axes_a:";
    for (auto v : contract_axes_a) std::cerr << ' ' << v;
    std::cerr << "\n  axes_b:";
    for (auto v : contract_axes_b) std::cerr << ' ' << v;
    std::cerr << "\n";
#endif

    if (contract_axes_a.size() != contract_axes_b.size()) {
      throw std::invalid_argument("contract: axis lists must have equal length");
    }

    auto collect_free_axes = [](size_t rank,
                                const std::vector<cytnx::cytnx_uint64>& contracted) {
      std::vector<bool> used(rank, false);
      for (auto idx : contracted) used[idx] = true;
      std::vector<cytnx::cytnx_uint64> free_axes;
      free_axes.reserve(rank - contracted.size());
      for (size_t i = 0; i < rank; ++i) {
        if (!used[i]) free_axes.push_back(static_cast<cytnx::cytnx_uint64>(i));
      }
      return free_axes;
    };

    const auto free_axes_a = collect_free_axes(rank_a, contract_axes_a);
    const auto free_axes_b = collect_free_axes(rank_b, contract_axes_b);
    const size_t total_free = free_axes_a.size() + free_axes_b.size();

    std::vector<cytnx::cytnx_uint64> permute_order;

    // Only check output axis specification if it's non-empty
    if (!bd_labs_c.empty()) {
      if (bd_labs_c.size() != total_free) {
        throw std::invalid_argument("contract: output axis specification mismatch");
      }

      std::vector<bool> seen(total_free, false);
      permute_order.reserve(total_free);
      for (auto axis : bd_labs_c) {
        if (axis < 0 || static_cast<size_t>(axis) >= total_free || seen[axis]) {
          throw std::invalid_argument("contract: invalid output axis permutation");
        }
        seen[axis] = true;
        permute_order.push_back(static_cast<cytnx::cytnx_uint64>(axis));
      }
    }

    try {
      cytnx::Tensor result
          = cytnx::linalg::Tensordot(a, b, contract_axes_a, contract_axes_b);

      if (!permute_order.empty()) {
        result = result.permute(permute_order);
      }

      c = std::move(result);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("contract: Tensordot failed - ") + e.what());
    }
  }

  template <> void contract(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                            const std::string_view bd_labs_str_a, const cytnx::Tensor& b,
                            const std::string_view bd_labs_str_b, cytnx::Tensor& c,
                            const std::string_view bd_labs_str_c) {
    // Convert string notation to bond label lists
    List<bond_label_t<cytnx::Tensor>> bd_labs_a, bd_labs_b, bd_labs_c;

    // Parse string labels (each character represents a bond)
    for (char ch : bd_labs_str_a) {
      bd_labs_a.push_back(static_cast<bond_label_t<cytnx::Tensor>>(ch));
    }
    for (char ch : bd_labs_str_b) {
      bd_labs_b.push_back(static_cast<bond_label_t<cytnx::Tensor>>(ch));
    }
    for (char ch : bd_labs_str_c) {
      bd_labs_c.push_back(static_cast<bond_label_t<cytnx::Tensor>>(ch));
    }

    // Use the label version of contract
    contract(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);
  }

  template <> void svd(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                       const rank_t<cytnx::Tensor> num_of_bds_as_row, cytnx::Tensor& u,
                       real_ten_t<cytnx::Tensor>& s_diag, cytnx::Tensor& v_dag) {
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

    // Reshape tensor to matrix form for SVD
    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Perform SVD using Cytnx. Return order is [S, U, V^T].
    auto svd_result = cytnx::linalg::Svd(matrix);

    s_diag = svd_result[0];  // Singular values as a rank-1 tensor
    u = svd_result[1];       // Left singular vectors
    v_dag = svd_result[2];   // Right singular vectors (already transposed)

    // Reshape U to match original bond structure
    std::vector<cytnx::cytnx_uint64> u_shape;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      u_shape.push_back(shape[i]);
    }
    u_shape.push_back(s_diag.shape()[0]);  // Add bond dimension
    u.reshape_(u_shape);

    // Reshape V† to match original bond structure
    std::vector<cytnx::cytnx_uint64> v_shape;
    v_shape.push_back(s_diag.shape()[0]);  // Bond dimension first
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      v_shape.push_back(shape[i]);
    }
    v_dag.reshape_(v_shape);
  }

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
    inout = inout * s;
  }

  template <> void scale(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                         const elem_t<cytnx::Tensor> s, cytnx::Tensor& out) {
    out = in * s;
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

    out = ins[0] * coefs[0];
    for (size_t i = 1; i < std::min(ins.size(), coefs.size()); ++i) {
      out = out + (ins[i] * coefs[i]);
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
        elem_t<cytnx::Tensor> trace_sum = 0.0;
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

  template <> void trunc_svd(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
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
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Use Cytnx Svd_truncate with parameters:
    // keepdim = chi_max (maximum singular values to keep)
    // err = s_min (threshold for small singular values)
    // is_UvT = true (return U and V†)
    // return_err = 1 (return truncation error)
    // mindim = 1 (minimum number of singular values to keep)
    std::vector<cytnx::Tensor> svd_result = cytnx::linalg::Svd_truncate(
        matrix, static_cast<cytnx::cytnx_uint64>(chi_max), s_min, true, 1, 1);

    if (svd_result.size() < 4) {
      throw std::runtime_error("trunc_svd: unexpected result size from Svd_truncate");
    }

    s_diag = svd_result[0];  // Singular values
    u = svd_result[1];       // Left unitary matrix U
    v_dag = svd_result[2];   // Right unitary matrix V†

    // Extract truncation error (last element in result)
    cytnx::Tensor err_tensor = svd_result[3];
    if (err_tensor.dtype() == cytnx::Type.Double) {
      trunc_err = err_tensor.item<double>();
    } else if (err_tensor.dtype() == cytnx::Type.Float) {
      trunc_err = static_cast<double>(err_tensor.item<float>());
    } else {
      trunc_err = 0.0;  // Default if we can't extract the error
    }

    // Reshape results to match original tensor structure
    if (u.shape().size() != 2) {
      auto u_shape_vec = u.shape();
      std::vector<cytnx::cytnx_int64> new_u_shape;

      // For U matrix: row dimensions from original tensor + kept singular values
      for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
        new_u_shape.push_back(static_cast<cytnx::cytnx_int64>(shape[i]));
      }
      new_u_shape.push_back(u_shape_vec[1]);  // Number of kept singular values

      u.reshape_(new_u_shape);
    }

    if (v_dag.shape().size() != 2) {
      auto v_dag_shape_vec = v_dag.shape();
      std::vector<cytnx::cytnx_int64> new_v_dag_shape;

      // For V† matrix: kept singular values + column dimensions from original tensor
      new_v_dag_shape.push_back(v_dag_shape_vec[0]);  // Number of kept singular values
      for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
        new_v_dag_shape.push_back(static_cast<cytnx::cytnx_int64>(shape[i]));
      }

      v_dag.reshape_(new_v_dag_shape);
    }
  }

  template <> void qr(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                      const rank_t<cytnx::Tensor> num_of_bds_as_row, cytnx::Tensor& q,
                      cytnx::Tensor& r) {
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

    // Reshape tensor to matrix form for QR
    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Perform QR decomposition using Cytnx
    auto qr_result = cytnx::linalg::Qr(matrix);

    // Extract results
    q = qr_result[0];  // Q matrix (orthogonal)
    r = qr_result[1];  // R matrix (upper triangular)

    // Reshape Q to match original bond structure
    std::vector<cytnx::cytnx_uint64> q_shape;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      q_shape.push_back(shape[i]);
    }
    q_shape.push_back(std::min(row_dim, col_dim));  // Add bond dimension
    q.reshape_(q_shape);

    // Reshape R to match bond structure
    std::vector<cytnx::cytnx_uint64> r_shape;
    r_shape.push_back(std::min(row_dim, col_dim));  // Bond dimension first
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      r_shape.push_back(shape[i]);
    }
    r.reshape_(r_shape);
  }

  template <> void lq(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                      const rank_t<cytnx::Tensor> num_of_bds_as_row, cytnx::Tensor& l,
                      cytnx::Tensor& q) {
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

    // Reshape tensor to matrix form
    cytnx::Tensor matrix = a.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Transpose for QR decomposition (A = LQ ⇔ A^T = Q^TR^T)
    auto matrix_t = matrix.permute({1, 0});  // swap dimensions for 2D transpose

    // Perform QR decomposition on transpose
    auto qr_result = cytnx::linalg::Qr(matrix_t);
    auto q_temp = qr_result[0];  // Q from QR of A^T
    auto r_temp = qr_result[1];  // R from QR of A^T

    // For LQ: A = LQ, so A^T = Q^TR^T
    // Therefore: Q = (Q_temp)^T, L = (R_temp)^T
    q = q_temp.permute({1, 0});
    l = r_temp.permute({1, 0});

    // Reshape L to match original bond structure
    std::vector<cytnx::cytnx_uint64> l_shape;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
      l_shape.push_back(shape[i]);
    }
    l_shape.push_back(std::min(row_dim, col_dim));  // Add bond dimension
    l.reshape_(l_shape);

    // Reshape Q to match bond structure
    std::vector<cytnx::cytnx_uint64> q_shape;
    q_shape.push_back(std::min(row_dim, col_dim));  // Bond dimension first
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
      q_shape.push_back(shape[i]);
    }
    q.reshape_(q_shape);
  }

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
