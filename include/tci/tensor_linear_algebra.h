#pragma once

#include <string_view>
#include <cytnx.hpp>
#include <map>
#include <set>
#include <sstream>
#include <numeric>
#include <algorithm>

#include "tci/tensor_traits.h"
#include "tci/cytnx_tensor_traits.h"

namespace tci {

  /**
   * @brief Calculate Frobenius norm of tensor
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @return real_t<TenT> Frobenius norm
   */
  template <typename TenT> real_t<TenT> norm(context_handle_t<TenT>& ctx, const TenT& a);

  /**
   * @brief Normalize tensor by its Frobenius norm (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to normalize
   * @return elem_t<TenT> Original norm before normalization
   */
  template <typename TenT> elem_t<TenT> normalize(context_handle_t<TenT>& ctx, TenT& inout);

  /**
   * @brief Normalize tensor by its Frobenius norm (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param out Normalized tensor
   * @return elem_t<TenT> Original norm before normalization
   */
  template <typename TenT>
  elem_t<TenT> normalize(context_handle_t<TenT>& ctx, const TenT& in, TenT& out);

  /**
   * @brief Scale tensor by a factor (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to scale
   * @param s Scaling factor
   */
  template <typename TenT>
  void scale(context_handle_t<TenT>& ctx, TenT& inout, const elem_t<TenT> s);

  /**
   * @brief Scale tensor by a factor (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param s Scaling factor
   * @param out Output scaled tensor
   */
  template <typename TenT>
  void scale(context_handle_t<TenT>& ctx, const TenT& in, const elem_t<TenT> s, TenT& out);

  /**
   * @brief Diagonal operation (rank-1 ⇔ rank-2 diagonal matrix conversion) - in-place
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to transform (rank-1 → rank-2 diagonal or rank-2 → rank-1)
   */
  template <typename TenT> void diag(context_handle_t<TenT>& ctx, TenT& inout);

  /**
   * @brief Diagonal operation (rank-1 ⇔ rank-2 diagonal matrix conversion) - out-of-place
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor (rank-1 or rank-2)
   * @param out Output tensor (rank-2 diagonal or rank-1)
   */
  template <typename TenT> void diag(context_handle_t<TenT>& ctx, const TenT& in, TenT& out);

  /**
   * @brief Partial trace operation - in-place
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to trace
   * @param bdidx_pairs Bond index pairs to trace over
   */
  template <typename TenT>
  void trace(context_handle_t<TenT>& ctx, TenT& inout, const bond_idx_pairs_t<TenT>& bdidx_pairs);

  /**
   * @brief Partial trace operation - out-of-place
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param bdidx_pairs Bond index pairs to trace over
   * @param out Output traced tensor
   */
  template <typename TenT> void trace(context_handle_t<TenT>& ctx, const TenT& in,
                                      const bond_idx_pairs_t<TenT>& bdidx_pairs, TenT& out);

  /**
   * @brief Tensor contraction with bond labels (version 1)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a First tensor
   * @param bd_labs_a Bond labels for tensor a
   * @param b Second tensor
   * @param bd_labs_b Bond labels for tensor b
   * @param c Output tensor
   * @param bd_labs_c Bond labels for tensor c
   */
  template <typename TenT> void contract(context_handle_t<TenT>& ctx, const TenT& a,
                                         const List<bond_label_t<TenT>>& bd_labs_a, const TenT& b,
                                         const List<bond_label_t<TenT>>& bd_labs_b, TenT& c,
                                         const List<bond_label_t<TenT>>& bd_labs_c);

  /**
   * @brief Tensor contraction with string notation (version 2)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a First tensor
   * @param bd_labs_str_a Bond labels string for tensor a (e.g., "ijk")
   * @param b Second tensor
   * @param bd_labs_str_b Bond labels string for tensor b (e.g., "kjl")
   * @param c Output tensor
   * @param bd_labs_str_c Bond labels string for tensor c (e.g., "il")
   */
  template <typename TenT> void contract(context_handle_t<TenT>& ctx, const TenT& a,
                                         const std::string_view bd_labs_str_a, const TenT& b,
                                         const std::string_view bd_labs_str_b, TenT& c,
                                         const std::string_view bd_labs_str_c);

  /**
   * @brief Linear combination of tensors (with uniform coefficients)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param ins List of input tensors
   * @param out Output tensor (linear combination)
   */
  template <typename TenT>
  void linear_combine(context_handle_t<TenT>& ctx, const List<TenT>& ins, TenT& out);

  /**
   * @brief Linear combination of tensors (with specified coefficients)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param ins List of input tensors
   * @param coefs List of coefficients
   * @param out Output tensor (linear combination)
   */
  template <typename TenT> void linear_combine(context_handle_t<TenT>& ctx, const List<TenT>& ins,
                                               const List<elem_t<TenT>>& coefs, TenT& out);

  /**
   * @brief Singular Value Decomposition
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param u Left unitary matrix
   * @param s_diag Singular values (diagonal)
   * @param v_dag Right unitary matrix (conjugate transpose)
   */
  template <typename TenT> void svd(context_handle_t<TenT>& ctx, const TenT& a,
                                    const rank_t<TenT> num_of_bds_as_row, TenT& u,
                                    real_ten_t<TenT>& s_diag, TenT& v_dag);

  /**
   * @brief Truncated SVD with maximum bond dimension
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param u Left unitary matrix
   * @param s_diag Singular values (diagonal)
   * @param v_dag Right unitary matrix (conjugate transpose)
   * @param trunc_err Truncation error (output)
   * @param chi_max Maximum bond dimension
   * @param s_min Minimum singular value threshold
   */
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const rank_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const bond_dim_t<TenT> chi_max, const real_t<TenT> s_min);

  /**
   * @brief QR decomposition
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param q Orthogonal matrix
   * @param r Upper triangular matrix
   */
  template <typename TenT> void qr(context_handle_t<TenT>& ctx, const TenT& a,
                                   const rank_t<TenT> num_of_bds_as_row, TenT& q, TenT& r);

  /**
   * @brief LQ decomposition
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param l Lower triangular matrix
   * @param q Orthogonal matrix
   */
  template <typename TenT> void lq(context_handle_t<TenT>& ctx, const TenT& a,
                                   const rank_t<TenT> num_of_bds_as_row, TenT& l, TenT& q);

  /**
   * @brief Compute eigenvalues (general matrix)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor (square matrix)
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param w_diag Eigenvalues
   */
  template <typename TenT> void eigvals(context_handle_t<TenT>& ctx, const TenT& a,
                                        const rank_t<TenT> num_of_bds_as_row,
                                        cplx_ten_t<TenT>& w_diag);

  /**
   * @brief Compute eigenvalues (symmetric/hermitian matrix)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor (symmetric/hermitian matrix)
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param w_diag Real eigenvalues
   */
  template <typename TenT> void eigvalsh(context_handle_t<TenT>& ctx, const TenT& a,
                                         const rank_t<TenT> num_of_bds_as_row,
                                         real_ten_t<TenT>& w_diag);

  /**
   * @brief Compute eigenvalues and eigenvectors (general matrix)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor (square matrix)
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param w_diag Eigenvalues
   * @param v Eigenvectors
   */
  template <typename TenT> void eig(context_handle_t<TenT>& ctx, const TenT& a,
                                    const rank_t<TenT> num_of_bds_as_row, cplx_ten_t<TenT>& w_diag,
                                    cplx_ten_t<TenT>& v);

  /**
   * @brief Compute eigenvalues and eigenvectors (symmetric/hermitian matrix)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor (symmetric/hermitian matrix)
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param w_diag Real eigenvalues
   * @param v Eigenvectors
   */
  template <typename TenT> void eigh(context_handle_t<TenT>& ctx, const TenT& a,
                                     const rank_t<TenT> num_of_bds_as_row, real_ten_t<TenT>& w_diag,
                                     TenT& v);

  /**
   * @brief Matrix exponential (general matrix) - in-place version
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Input/output tensor (square matrix)
   * @param num_of_bds_as_row Number of bonds to treat as rows
   */
  template <typename TenT> void exp(context_handle_t<TenT>& ctx, TenT& inout,
                                    const rank_t<TenT> num_of_bds_as_row);

  /**
   * @brief Matrix exponential (general matrix) - out-of-place version
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor (square matrix)
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param out Output tensor
   */
  template <typename TenT> void exp(context_handle_t<TenT>& ctx, const TenT& in,
                                    const rank_t<TenT> num_of_bds_as_row, TenT& out);

  // Implementation details for contract (moved from .cpp for template visibility)
  namespace detail {
    // NCON analysis: determine contraction and permutation from labels
    struct NCONAnalysis {
      std::vector<cytnx::cytnx_uint64> contract_axes_a, contract_axes_b;
      std::vector<cytnx::cytnx_uint64> free_axes_a, free_axes_b;
      std::vector<cytnx::cytnx_uint64> output_permutation;

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
        std::set<cytnx::cytnx_int64> labels_b(bd_labs_b.begin(), bd_labs_b.end());
        std::set<cytnx::cytnx_int64> labels_c(bd_labs_c.begin(), bd_labs_c.end());

        // Process tensor a: for each contracted axis, find corresponding axis in b
        // This ensures contract_axes_a[i] and contract_axes_b[i] have the same label
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (labels_b.count(label) && !labels_c.count(label)) {
            // This axis will be contracted
            contract_axes_a.push_back(static_cast<cytnx::cytnx_uint64>(i));
            // Find corresponding axis in b with same label
            for (size_t j = 0; j < bd_labs_b.size(); ++j) {
              if (bd_labs_b[j] == label) {
                contract_axes_b.push_back(static_cast<cytnx::cytnx_uint64>(j));
                break;
              }
            }
          } else if (labels_c.count(label)) {
            free_axes_a.push_back(static_cast<cytnx::cytnx_uint64>(i));
          }
        }

        // Process tensor b: find free axes (not contracted)
        std::set<cytnx::cytnx_uint64> contracted_b_set(contract_axes_b.begin(), contract_axes_b.end());
        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          if (contracted_b_set.count(i) == 0) {
            auto label = bd_labs_b[i];
            if (labels_c.count(label)) {
              free_axes_b.push_back(static_cast<cytnx::cytnx_uint64>(i));
            }
          }
        }

        // Calculate output permutation to match bd_labs_c order
        calculate_output_permutation(bd_labs_a, bd_labs_b, bd_labs_c);
      }

      void calculate_output_permutation(const List<bond_label_t<cytnx::Tensor>>& bd_labs_a,
                                        const List<bond_label_t<cytnx::Tensor>>& bd_labs_b,
                                        const List<bond_label_t<cytnx::Tensor>>& bd_labs_c) {
        output_permutation.clear();

        if (bd_labs_c.empty()) {
          return; // No output reordering needed
        }

        // Build list of free axes in natural order (tensor a first, then tensor b)
        std::vector<cytnx::cytnx_int64> natural_order;

        // Add free axes from tensor a
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (std::find(contract_axes_a.begin(), contract_axes_a.end(), i) == contract_axes_a.end()) {
            natural_order.push_back(label);
          }
        }

        // Add free axes from tensor b
        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          auto label = bd_labs_b[i];
          if (std::find(contract_axes_b.begin(), contract_axes_b.end(), i) == contract_axes_b.end()) {
            natural_order.push_back(label);
          }
        }

        // Create permutation: for each position in bd_labs_c, find where it is in natural_order
        output_permutation.resize(bd_labs_c.size());
        for (size_t i = 0; i < bd_labs_c.size(); ++i) {
          auto desired_label = bd_labs_c[i];
          auto it = std::find(natural_order.begin(), natural_order.end(), desired_label);
          if (it != natural_order.end()) {
            size_t natural_pos = std::distance(natural_order.begin(), it);
            output_permutation[i] = static_cast<cytnx::cytnx_uint64>(natural_pos);
          } else {
            throw std::invalid_argument("contract: output label not found in free axes");
          }
        }
      }
    };
  }  // namespace detail

  // Template specializations for contract (moved from .cpp for template visibility)
  template <> inline void contract(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
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
      detail::NCONAnalysis analysis(bd_labs_a, bd_labs_b, bd_labs_c);

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
        if (!analysis.output_permutation.empty()) {
          // Filter out invalid permutation indices
          std::vector<cytnx::cytnx_uint64> valid_perm;
          for (auto idx : analysis.output_permutation) {
            if (idx < static_cast<cytnx::cytnx_uint64>(result.shape().size())) {
              valid_perm.push_back(idx);
            }
          }
          if (valid_perm.size() == result.shape().size()) {
            result = result.permute(valid_perm);
          }
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
        if (!analysis.output_permutation.empty()) {
          // Filter out invalid permutation indices
          std::vector<cytnx::cytnx_uint64> valid_perm;
          for (auto idx : analysis.output_permutation) {
            if (idx < static_cast<cytnx::cytnx_uint64>(result.shape().size())) {
              valid_perm.push_back(idx);
            }
          }
          if (valid_perm.size() == result.shape().size()) {
            result = result.permute(valid_perm);
          }
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

  template <> inline void contract(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
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

}  // namespace tci