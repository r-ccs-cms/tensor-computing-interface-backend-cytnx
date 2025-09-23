#include "tci/construction_destruction.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>

namespace tci {

// Template specializations for Cytnx::Tensor

template <>
void allocate(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape,
    cytnx::Tensor &a
) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
        cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    // Create uninitialized tensor
    a = cytnx::Tensor(cytnx_shape, cytnx::Type.ComplexDouble, ctx);
}

template <>
cytnx::Tensor allocate(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape
) {
    cytnx::Tensor result;
    allocate(ctx, shape, result);
    return result;
}

template <>
void zeros(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape,
    cytnx::Tensor &a
) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
        cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    // Create zero-filled tensor
    a = cytnx::zeros(cytnx_shape, cytnx::Type.ComplexDouble, ctx);
}

template <>
cytnx::Tensor zeros(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape
) {
    cytnx::Tensor result;
    zeros(ctx, shape, result);
    return result;
}

template <>
void eye(
    context_handle_t<cytnx::Tensor> &ctx,
    const bond_dim_t<cytnx::Tensor> N,
    cytnx::Tensor &a
) {
    a = cytnx::eye(static_cast<cytnx::cytnx_uint64>(N), cytnx::Type.ComplexDouble, ctx);
}

template <>
cytnx::Tensor eye(
    context_handle_t<cytnx::Tensor> &ctx,
    const bond_dim_t<cytnx::Tensor> N
) {
    cytnx::Tensor result;
    eye(ctx, N, result);
    return result;
}

template <>
void copy(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &orig,
    cytnx::Tensor &dist
) {
    dist = orig.clone();
}

template <>
cytnx::Tensor copy(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &orig
) {
    return orig.clone();
}

template <>
void clear(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &a
) {
    // Create empty tensor
    a = cytnx::Tensor();
}

template <>
void fill(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape,
    const elem_t<cytnx::Tensor> v,
    cytnx::Tensor &a
) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
        cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    // Create tensor filled with value v
    a = cytnx::Tensor(cytnx_shape, cytnx::Type.ComplexDouble, ctx);
    // Fill with the specified value
    a.fill(v);
}

template <>
cytnx::Tensor fill(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape,
    const elem_t<cytnx::Tensor> v
) {
    cytnx::Tensor result;
    fill(ctx, shape, v, result);
    return result;
}

template <>
void move(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &from,
    cytnx::Tensor &to
) {
    to = std::move(from);
    // Ensure from is in a valid empty state
    from = cytnx::Tensor();
}

template <>
cytnx::Tensor move(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &from
) {
    cytnx::Tensor result = std::move(from);
    from = cytnx::Tensor();
    return result;
}

// Explicit specialization for cytnx::Tensor - in-place version
template <>
void assign_from_container(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape,
    std::vector<elem_t<cytnx::Tensor>>::iterator init_elems_begin,
    std::function<std::ptrdiff_t(const elem_coors_t<cytnx::Tensor> &)> &&coors2idx,
    cytnx::Tensor &a
) {
    // Create tensor with the specified shape
    allocate(ctx, shape, a);

    // Generate all coordinate combinations and assign values
    std::function<void(elem_coors_t<cytnx::Tensor>, std::size_t)> assign_recursive;
    assign_recursive = [&](elem_coors_t<cytnx::Tensor> current_coords, std::size_t dim) {
        if (dim == shape.size()) {
            // Base case: all dimensions set, assign the element
            auto index = coors2idx(current_coords);
            auto value = *(init_elems_begin + index);

            // Convert coordinates to cytnx format and set element
            std::vector<cytnx::cytnx_uint64> cytnx_coords;
            cytnx_coords.reserve(current_coords.size());
            for (const auto& coord : current_coords) {
                cytnx_coords.push_back(static_cast<cytnx::cytnx_uint64>(coord));
            }

            a.at<elem_t<cytnx::Tensor>>(cytnx_coords) = value;
        } else {
            // Recursive case: iterate through current dimension
            for (bond_dim_t<cytnx::Tensor> i = 0; i < shape[dim]; ++i) {
                current_coords.push_back(i);
                assign_recursive(current_coords, dim + 1);
                current_coords.pop_back();
            }
        }
    };

    assign_recursive({}, 0);
}

// Explicit specialization for cytnx::Tensor - out-of-place version
template <>
cytnx::Tensor assign_from_container(
    context_handle_t<cytnx::Tensor> &ctx,
    const shape_t<cytnx::Tensor> &shape,
    std::vector<elem_t<cytnx::Tensor>>::iterator init_elems_begin,
    std::function<std::ptrdiff_t(const elem_coors_t<cytnx::Tensor> &)> &&coors2idx
) {
    cytnx::Tensor result;
    assign_from_container(ctx, shape, init_elems_begin, std::move(coors2idx), result);
    return result;
}

} // namespace tci