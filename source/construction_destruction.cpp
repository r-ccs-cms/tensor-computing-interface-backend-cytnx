#include "tci/construction_destruction.h"

#include <cytnx.hpp>
#include <functional>
#include <complex>

#include "tci/cytnx_tensor_traits.h"
#include "tci/variant_helpers.h"

namespace tci {

  // Template specializations for Cytnx::Tensor

  template <> void allocate(context_handle_t<cytnx::Tensor>& ctx,
                            const shape_t<cytnx::Tensor>& shape, cytnx::Tensor& a) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    // Create uninitialized tensor
    a = cytnx::Tensor(cytnx_shape, cytnx::Type.ComplexDouble, ctx);
  }

  template <> cytnx::Tensor allocate(context_handle_t<cytnx::Tensor>& ctx,
                                     const shape_t<cytnx::Tensor>& shape) {
    cytnx::Tensor result;
    allocate(ctx, shape, result);
    return result;
  }

  template <> void zeros(context_handle_t<cytnx::Tensor>& ctx, const shape_t<cytnx::Tensor>& shape,
                         cytnx::Tensor& a) {
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
  cytnx::Tensor zeros(context_handle_t<cytnx::Tensor>& ctx, const shape_t<cytnx::Tensor>& shape) {
    cytnx::Tensor result;
    zeros(ctx, shape, result);
    return result;
  }


  template <> void eye(context_handle_t<cytnx::Tensor>& ctx, const bond_dim_t<cytnx::Tensor> N,
                       cytnx::Tensor& a) {
    a = cytnx::eye(static_cast<cytnx::cytnx_uint64>(N), cytnx::Type.ComplexDouble, ctx);
  }

  template <>
  cytnx::Tensor eye(context_handle_t<cytnx::Tensor>& ctx, const bond_dim_t<cytnx::Tensor> N) {
    cytnx::Tensor result;
    eye(ctx, N, result);
    return result;
  }

  template <>
  void copy(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& orig, cytnx::Tensor& dist) {
    dist = orig.clone();
  }

  template <> cytnx::Tensor copy(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& orig) {
    return orig.clone();
  }

  template <> void clear(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& a) {
    // Create empty tensor
    a = cytnx::Tensor();
  }

  template <> void fill(context_handle_t<cytnx::Tensor>& ctx, const shape_t<cytnx::Tensor>& shape,
                        const elem_t<cytnx::Tensor> v, cytnx::Tensor& a) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    // Create tensor filled with value v
    a = cytnx::Tensor(cytnx_shape, cytnx::Type.ComplexDouble, ctx);
    // Fill with the specified value (convert variant to compatible type)
    auto complex_val = tci::to_complex128(v);
    a.fill(complex_val);
  }

  template <> cytnx::Tensor fill(context_handle_t<cytnx::Tensor>& ctx,
                                 const shape_t<cytnx::Tensor>& shape,
                                 const elem_t<cytnx::Tensor> v) {
    cytnx::Tensor result;
    fill(ctx, shape, v, result);
    return result;
  }

  template <>
  void move(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& from, cytnx::Tensor& to) {
    to = std::move(from);
    // Ensure from is in a valid empty state
    from = cytnx::Tensor();
  }

  template <> cytnx::Tensor move(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& from) {
    cytnx::Tensor result = std::move(from);
    from = cytnx::Tensor();
    return result;
  }


}  // namespace tci
