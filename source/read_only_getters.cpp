#include "tci/read_only_getters.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>

namespace tci {

// Template specializations for Cytnx::Tensor

template <>
rank_t<cytnx::Tensor> rank(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    return static_cast<rank_t<cytnx::Tensor>>(a.shape().size());
}

template <>
shape_t<cytnx::Tensor> shape(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    auto cytnx_shape = a.shape();
    shape_t<cytnx::Tensor> result;
    result.reserve(cytnx_shape.size());
    for (const auto& dim : cytnx_shape) {
        result.push_back(static_cast<bond_dim_t<cytnx::Tensor>>(dim));
    }
    return result;
}

template <>
ten_size_t<cytnx::Tensor> size(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    auto cytnx_shape = a.shape();
    ten_size_t<cytnx::Tensor> total_size = 1;
    for (const auto& dim : cytnx_shape) {
        total_size *= static_cast<ten_size_t<cytnx::Tensor>>(dim);
    }
    return total_size;
}

template <>
std::size_t size_bytes(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    // Calculate based on element type and total size
    auto total_elements = size(ctx, a);

    // Get the actual element size based on dtype
    std::size_t element_size;
    if (a.dtype() == cytnx::Type.ComplexDouble) {
        element_size = sizeof(cytnx::cytnx_complex128);
    } else if (a.dtype() == cytnx::Type.ComplexFloat) {
        element_size = sizeof(cytnx::cytnx_complex64);
    } else if (a.dtype() == cytnx::Type.Double) {
        element_size = sizeof(cytnx::cytnx_double);
    } else if (a.dtype() == cytnx::Type.Float) {
        element_size = sizeof(cytnx::cytnx_float);
    } else if (a.dtype() == cytnx::Type.Int64) {
        element_size = sizeof(cytnx::cytnx_int64);
    } else if (a.dtype() == cytnx::Type.Int32) {
        element_size = sizeof(cytnx::cytnx_int32);
    } else if (a.dtype() == cytnx::Type.Int16) {
        element_size = sizeof(cytnx::cytnx_int16);
    } else if (a.dtype() == cytnx::Type.Uint64) {
        element_size = sizeof(cytnx::cytnx_uint64);
    } else if (a.dtype() == cytnx::Type.Uint32) {
        element_size = sizeof(cytnx::cytnx_uint32);
    } else if (a.dtype() == cytnx::Type.Uint16) {
        element_size = sizeof(cytnx::cytnx_uint16);
    } else {
        // Default to complex128 for unknown types
        element_size = sizeof(cytnx::cytnx_complex128);
    }

    return total_elements * element_size;
}

template <>
void get_elem(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const elem_coors_t<cytnx::Tensor> &coors,
    elem_t<cytnx::Tensor> &elem
) {
    // Convert coordinates to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
        cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }

    // Get element from Cytnx tensor
    auto cytnx_scalar = a.at(cytnx_coors);

    // Convert Cytnx scalar to complex128 using explicit cast
    if (a.dtype() == cytnx::Type.ComplexDouble) {
        // For complex types, get real and imaginary parts
        double real_part = static_cast<double>(cytnx_scalar.real());
        double imag_part = static_cast<double>(cytnx_scalar.imag());
        elem = cytnx::cytnx_complex128(real_part, imag_part);
    } else if (a.dtype() == cytnx::Type.Double) {
        // For real types, imaginary part is zero
        double real_part = static_cast<double>(cytnx_scalar.real());
        elem = cytnx::cytnx_complex128(real_part, 0.0);
    } else {
        // For other types, convert to double first
        double real_part = static_cast<double>(cytnx_scalar.real());
        elem = cytnx::cytnx_complex128(real_part, 0.0);
    }
}

template <>
elem_t<cytnx::Tensor> get_elem(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const elem_coors_t<cytnx::Tensor> &coors
) {
    elem_t<cytnx::Tensor> elem;
    get_elem(ctx, a, coors, elem);
    return elem;
}

} // namespace tci