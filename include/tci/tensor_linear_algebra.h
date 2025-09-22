#pragma once

#include "tci/tensor_traits.h"
#include <string_view>

namespace tci {

/**
 * @brief Calculate Frobenius norm of tensor
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param a Input tensor
 * @return real_t<TenT> Frobenius norm
 */
template <typename TenT>
real_t<TenT> norm(
    context_handle_t<TenT> &ctx,
    const TenT &a
);

/**
 * @brief Normalize tensor by its Frobenius norm (in-place version)
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param inout Tensor to normalize
 * @return elem_t<TenT> Original norm before normalization
 */
template <typename TenT>
elem_t<TenT> normalize(
    context_handle_t<TenT> &ctx,
    TenT &inout
);

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
elem_t<TenT> normalize(
    context_handle_t<TenT> &ctx,
    const TenT &in,
    TenT &out
);

/**
 * @brief Scale tensor by a factor (in-place version)
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param inout Tensor to scale
 * @param s Scaling factor
 */
template <typename TenT>
void scale(
    context_handle_t<TenT> &ctx,
    TenT &inout,
    const elem_t<TenT> s
);

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
void scale(
    context_handle_t<TenT> &ctx,
    const TenT &in,
    const elem_t<TenT> s,
    TenT &out
);

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
template <typename TenT>
void contract(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const List<bond_label_t<TenT>> &bd_labs_a,
    const TenT &b,
    const List<bond_label_t<TenT>> &bd_labs_b,
    TenT &c,
    const List<bond_label_t<TenT>> &bd_labs_c
);

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
template <typename TenT>
void contract(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const std::string_view bd_labs_str_a,
    const TenT &b,
    const std::string_view bd_labs_str_b,
    TenT &c,
    const std::string_view bd_labs_str_c
);

/**
 * @brief Linear combination of tensors (with uniform coefficients)
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param ins List of input tensors
 * @param out Output tensor (linear combination)
 */
template <typename TenT>
void linear_combine(
    context_handle_t<TenT> &ctx,
    const List<TenT> &ins,
    TenT &out
);

/**
 * @brief Linear combination of tensors (with specified coefficients)
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param ins List of input tensors
 * @param coefs List of coefficients
 * @param out Output tensor (linear combination)
 */
template <typename TenT>
void linear_combine(
    context_handle_t<TenT> &ctx,
    const List<TenT> &ins,
    const List<elem_t<TenT>> &coefs,
    TenT &out
);

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
template <typename TenT>
void svd(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    TenT &u,
    real_ten_t<TenT> &s_diag,
    TenT &v_dag
);

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
void trunc_svd(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    TenT &u,
    real_ten_t<TenT> &s_diag,
    TenT &v_dag,
    real_t<TenT> &trunc_err,
    const bond_dim_t<TenT> chi_max,
    const real_t<TenT> s_min
);

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
template <typename TenT>
void qr(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    TenT &q,
    TenT &r
);

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
template <typename TenT>
void lq(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    TenT &l,
    TenT &q
);

/**
 * @brief Compute eigenvalues (general matrix)
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param a Input tensor (square matrix)
 * @param num_of_bds_as_row Number of bonds to treat as rows
 * @param w_diag Eigenvalues
 */
template <typename TenT>
void eigvals(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    cplx_ten_t<TenT> &w_diag
);

/**
 * @brief Compute eigenvalues (symmetric/hermitian matrix)
 *
 * @tparam TenT Tensor type
 * @param ctx Context handle for the tensor library
 * @param a Input tensor (symmetric/hermitian matrix)
 * @param num_of_bds_as_row Number of bonds to treat as rows
 * @param w_diag Real eigenvalues
 */
template <typename TenT>
void eigvalsh(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    real_ten_t<TenT> &w_diag
);

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
template <typename TenT>
void eig(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    cplx_ten_t<TenT> &w_diag,
    cplx_ten_t<TenT> &v
);

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
template <typename TenT>
void eigh(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    const rank_t<TenT> num_of_bds_as_row,
    real_ten_t<TenT> &w_diag,
    TenT &v
);

} // namespace tci