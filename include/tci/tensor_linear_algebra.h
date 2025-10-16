#pragma once

#include <algorithm>
#include <cytnx.hpp>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string_view>

#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"

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
   * @return real_t<TenT> Original norm before normalization
   */
  template <typename TenT> real_t<TenT> normalize(context_handle_t<TenT>& ctx, TenT& inout);

  /**
   * @brief Normalize tensor by its Frobenius norm (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param out Normalized tensor
   * @return real_t<TenT> Original norm before normalization
   */
  template <typename TenT>
  real_t<TenT> normalize(context_handle_t<TenT>& ctx, const TenT& in, TenT& out);

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
   * @brief Truncated SVD with target truncation error
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param u Left unitary matrix
   * @param s_diag Singular values (diagonal)
   * @param v_dag Right unitary matrix (conjugate transpose)
   * @param trunc_err Truncation error (output)
   * @param target_trunc_err Target truncation error (chi_min=1, chi_max=∞ implicitly)
   * @param s_min Minimum singular value threshold
   */
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const rank_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const real_t<TenT> target_trunc_err, const real_t<TenT> s_min);

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
   * @param chi_max Maximum bond dimension (chi_min=1, target_trunc_err=0 implicitly)
   * @param s_min Minimum singular value threshold
   */
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const rank_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const bond_dim_t<TenT> chi_max, const real_t<TenT> s_min);

  /**
   * @brief Truncated SVD with full control
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param num_of_bds_as_row Number of bonds to treat as rows
   * @param u Left unitary matrix
   * @param s_diag Singular values (diagonal)
   * @param v_dag Right unitary matrix (conjugate transpose)
   * @param trunc_err Truncation error (output)
   * @param chi_min Minimum bond dimension
   * @param chi_max Maximum bond dimension
   * @param target_trunc_err Target truncation error
   * @param s_min Minimum singular value threshold
   */
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const rank_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const bond_dim_t<TenT> chi_min, const bond_dim_t<TenT> chi_max,
                 const real_t<TenT> target_trunc_err, const real_t<TenT> s_min);

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
  template <typename TenT>
  void exp(context_handle_t<TenT>& ctx, TenT& inout, const rank_t<TenT> num_of_bds_as_row);

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
    template <typename TenT> struct NCONAnalysis {
      std::vector<cytnx::cytnx_uint64> contract_axes_a, contract_axes_b;
      std::vector<cytnx::cytnx_uint64> free_axes_a, free_axes_b;
      std::vector<cytnx::cytnx_uint64> output_permutation;

      NCONAnalysis(const List<bond_label_t<TenT>>& bd_labs_a,
                   const List<bond_label_t<TenT>>& bd_labs_b,
                   const List<bond_label_t<TenT>>& bd_labs_c) {
        analyze(bd_labs_a, bd_labs_b, bd_labs_c);
      }

    private:
      void analyze(const List<bond_label_t<TenT>>& bd_labs_a,
                   const List<bond_label_t<TenT>>& bd_labs_b,
                   const List<bond_label_t<TenT>>& bd_labs_c) {
        // Find contracted indices (appear in both a and b, not in c)
        std::set<bond_label_t<TenT>> labels_b(bd_labs_b.begin(), bd_labs_b.end());
        std::set<bond_label_t<TenT>> labels_c(bd_labs_c.begin(), bd_labs_c.end());

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
        std::set<cytnx::cytnx_uint64> contracted_b_set(contract_axes_b.begin(),
                                                       contract_axes_b.end());
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

      void calculate_output_permutation(const List<bond_label_t<TenT>>& bd_labs_a,
                                        const List<bond_label_t<TenT>>& bd_labs_b,
                                        const List<bond_label_t<TenT>>& bd_labs_c) {
        output_permutation.clear();

        if (bd_labs_c.empty()) {
          return;  // No output reordering needed
        }

        // Build list of free axes in natural order (tensor a first, then tensor b)
        std::vector<bond_label_t<TenT>> natural_order;

        // Add free axes from tensor a
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (std::find(contract_axes_a.begin(), contract_axes_a.end(), i)
              == contract_axes_a.end()) {
            natural_order.push_back(label);
          }
        }

        // Add free axes from tensor b
        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          auto label = bd_labs_b[i];
          if (std::find(contract_axes_b.begin(), contract_axes_b.end(), i)
              == contract_axes_b.end()) {
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

}  // namespace tci
