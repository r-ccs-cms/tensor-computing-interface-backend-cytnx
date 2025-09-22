#include "tci/tensor_manipulation.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>

namespace tci {

// Template specializations for tensor manipulation functions using Cytnx

template <>
void set_elem(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &a,
    const elem_coors_t<cytnx::Tensor> &coors,
    const elem_t<cytnx::Tensor> el
) {
    // Convert coordinates to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
        cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }

    // Set element in Cytnx tensor
    a.at(cytnx_coors) = static_cast<cytnx::cytnx_complex128>(el);
}

template <>
void reshape(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout,
    const shape_t<cytnx::Tensor> &new_shape
) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(new_shape.size());
    for (const auto& dim : new_shape) {
        cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    inout.reshape_(cytnx_shape);
}

template <>
void transpose(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout,
    const List<bond_idx_t<cytnx::Tensor>> &new_order
) {
    // Convert to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
        cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }

    inout = inout.permute(cytnx_order);
}

template <>
void reshape(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    const shape_t<cytnx::Tensor> &new_shape,
    cytnx::Tensor &out
) {
    out = in.clone();
    reshape(ctx, out, new_shape);
}

template <>
void transpose(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    const List<bond_idx_t<cytnx::Tensor>> &new_order,
    cytnx::Tensor &out
) {
    // Convert to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
        cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }

    out = in.permute(cytnx_order);
}

template <>
void cplx_conj(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout
) {
    inout = cytnx::linalg::Conj(inout);
}

template <>
void cplx_conj(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    cytnx::Tensor &out
) {
    out = cytnx::linalg::Conj(in);
}

template <>
void to_cplx(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    cplx_ten_t<cytnx::Tensor> &out
) {
    if (in.dtype() == cytnx::Type.ComplexDouble || in.dtype() == cytnx::Type.ComplexFloat) {
        // Already complex, just copy
        out = in.clone();
    } else {
        // Convert real to complex
        out = in.astype(cytnx::Type.ComplexDouble);
    }
}

template <>
cplx_ten_t<cytnx::Tensor> to_cplx(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in
) {
    cplx_ten_t<cytnx::Tensor> result;
    to_cplx(ctx, in, result);
    return result;
}

template <>
void real(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    real_ten_t<cytnx::Tensor> &out
) {
    // TODO: Implement real part extraction
    // For now, just copy the tensor
    out = in.clone();
}

template <>
real_ten_t<cytnx::Tensor> real(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in
) {
    real_ten_t<cytnx::Tensor> result;
    real(ctx, in, result);
    return result;
}

template <>
void imag(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    real_ten_t<cytnx::Tensor> &out
) {
    // TODO: Implement imaginary part extraction
    // For now, just create a zero tensor
    out = cytnx::zeros(in.shape(), cytnx::Type.Double, ctx);
}

template <>
real_ten_t<cytnx::Tensor> imag(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in
) {
    real_ten_t<cytnx::Tensor> result;
    imag(ctx, in, result);
    return result;
}

} // namespace tci