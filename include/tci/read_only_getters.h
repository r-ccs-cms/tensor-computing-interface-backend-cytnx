#pragma once

#include <cytnx.hpp>

#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"

namespace tci {

  /**
   * @brief Get the order (number of dimensions) of a tensor
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @return order_t<TenT> The order of the tensor
   */
  template <typename TenT> order_t<TenT> order(context_handle_t<TenT>& ctx, const TenT& a);

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

}  // namespace tci
