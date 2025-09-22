#include "tci/tensor_linear_algebra.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>

namespace tci {

// Template specializations for linear algebra functions using Cytnx

template <>
real_t<cytnx::Tensor> norm(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    // Use Cytnx's built-in Norm function for Frobenius norm
    auto norm_result = cytnx::linalg::Norm(a);

    // Extract the scalar value with explicit cast
    if (norm_result.dtype() == cytnx::Type.Double) {
        return static_cast<double>(norm_result.at({0}).real());
    } else if (norm_result.dtype() == cytnx::Type.ComplexDouble) {
        return static_cast<double>(norm_result.at({0}).real()); // Norm should always be real
    } else {
        // Convert to double first
        auto converted = norm_result.astype(cytnx::Type.Double);
        return static_cast<double>(converted.at({0}).real());
    }
}

// TODO: Implement tensor contraction with Cytnx API
// Temporarily commented out due to API compatibility issues
/*
template <>
void contract(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const std::string_view bd_labs_str_a,
    const cytnx::Tensor &b,
    const std::string_view bd_labs_str_b,
    cytnx::Tensor &c,
    const std::string_view bd_labs_str_c
) {
    // Placeholder: just clone the first tensor
    c = a.clone();
}
*/

template <>
void svd(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const rank_t<cytnx::Tensor> num_of_bds_as_row,
    cytnx::Tensor &u,
    real_ten_t<cytnx::Tensor> &s_diag,
    cytnx::Tensor &v_dag
) {
    // TODO: Implement SVD with Cytnx API
    // For now, create placeholder results
    u = a.clone();
    s_diag = a.clone();
    v_dag = a.clone();
}

template <>
elem_t<cytnx::Tensor> normalize(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout
) {
    // Calculate norm
    auto original_norm = norm(ctx, inout);

    // Normalize by dividing by norm
    if (original_norm > 0.0) {
        inout = inout / cytnx::cytnx_complex128(original_norm, 0.0);
    }

    return cytnx::cytnx_complex128(original_norm, 0.0);
}

template <>
elem_t<cytnx::Tensor> normalize(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    cytnx::Tensor &out
) {
    out = in.clone();
    return normalize(ctx, out);
}

template <>
void scale(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout,
    const elem_t<cytnx::Tensor> s
) {
    inout = inout * s;
}

template <>
void scale(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    const elem_t<cytnx::Tensor> s,
    cytnx::Tensor &out
) {
    out = in * s;
}

template <>
void linear_combine(
    context_handle_t<cytnx::Tensor> &ctx,
    const List<cytnx::Tensor> &ins,
    cytnx::Tensor &out
) {
    if (ins.empty()) return;

    out = ins[0].clone();
    for (size_t i = 1; i < ins.size(); ++i) {
        out = out + ins[i];
    }
}

template <>
void linear_combine(
    context_handle_t<cytnx::Tensor> &ctx,
    const List<cytnx::Tensor> &ins,
    const List<elem_t<cytnx::Tensor>> &coefs,
    cytnx::Tensor &out
) {
    if (ins.empty() || coefs.empty()) return;

    out = ins[0] * coefs[0];
    for (size_t i = 1; i < std::min(ins.size(), coefs.size()); ++i) {
        out = out + (ins[i] * coefs[i]);
    }
}

} // namespace tci