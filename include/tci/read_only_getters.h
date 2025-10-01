#pragma once

#include "tci/tensor_traits.h"
#include <cytnx.hpp>
#include "tci/cytnx_tensor_traits.h"

namespace tci {

  /**
   * @brief Get the rank (number of dimensions) of a tensor
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @return rank_t<TenT> The rank of the tensor
   */
  template <typename TenT> rank_t<TenT> rank(context_handle_t<TenT>& ctx, const TenT& a);

  /**
   * @brief Get the shape (dimensions) of a tensor
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @return shape_t<TenT> The shape of the tensor
   */
  template <typename TenT> shape_t<TenT> shape(context_handle_t<TenT>& ctx, const TenT& a);

  /**
   * @brief Get the total number of elements in a tensor
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @return ten_size_t<TenT> The total number of elements
   */
  template <typename TenT> ten_size_t<TenT> size(context_handle_t<TenT>& ctx, const TenT& a);

  /**
   * @brief Get the memory usage of a tensor in bytes
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @return std::size_t Memory usage in bytes
   */
  template <typename TenT> std::size_t size_bytes(context_handle_t<TenT>& ctx, const TenT& a);

  /**
   * @brief Get the value of a specific element (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param coors Coordinates of the element
   * @param elem Output element value
   */
  template <typename TenT> void get_elem(context_handle_t<TenT>& ctx, const TenT& a,
                                         const elem_coors_t<TenT>& coors, elem_t<TenT>& elem);

  /**
   * @brief Get the value of a specific element (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param coors Coordinates of the element
   * @return elem_t<TenT> The element value
   */
  template <typename TenT> elem_t<TenT> get_elem(context_handle_t<TenT>& ctx, const TenT& a,
                                                 const elem_coors_t<TenT>& coors);

  // Implementation for size (moved from .cpp for template visibility)
  template <>
  inline ten_size_t<cytnx::Tensor> size(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    auto cytnx_shape = a.shape();
    ten_size_t<cytnx::Tensor> total_size = 1;
    for (const auto& dim : cytnx_shape) {
      total_size *= static_cast<ten_size_t<cytnx::Tensor>>(dim);
    }
    return total_size;
  }

  // Implementation for size_bytes (moved from .cpp for template visibility)
  template <>
  inline std::size_t size_bytes(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    auto total_elements = tci::size(ctx, a);
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
      element_size = sizeof(cytnx::cytnx_complex128);
    }
    return total_elements * element_size;
  }

}  // namespace tci