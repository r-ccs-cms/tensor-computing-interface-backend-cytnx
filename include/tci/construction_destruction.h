#pragma once

#include "tci/tensor_traits.h"
#include "tci/cytnx_tensor_traits.h"
#include <functional>
#include <complex>
#include <utility>

namespace tci {

  /**
   * @brief Allocate memory for a tensor with given shape (in-place version)
   *
   * Memory is allocated but not initialized for performance reasons.
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor to allocate
   * @param a Output tensor
   */
  template <typename TenT>
  void allocate(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, TenT& a);

  /**
   * @brief Allocate memory for a tensor with given shape (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor to allocate
   * @return TenT Allocated tensor
   */
  template <typename TenT> TenT allocate(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape);

  /**
   * @brief Create a tensor filled with zeros (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param a Output tensor
   */
  template <typename TenT>
  void zeros(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, TenT& a);

  /**
   * @brief Create a tensor filled with zeros (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @return TenT Zero-filled tensor
   */
  template <typename TenT> TenT zeros(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape);

  /**
   * @brief Create a tensor from a container/range (in-place version)
   *
   * @tparam TenT Tensor type
   * @tparam RandomIt Random access iterator type
   * @tparam Func Function type for coordinate to index mapping
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param init_elems_begin Iterator to beginning of elements
   * @param coors2idx Function to convert coordinates to linear index
   * @param a Output tensor
   */
  template <typename TenT, typename RandomIt, typename Func>
  void assign_from_container(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                             RandomIt init_elems_begin, Func&& coors2idx, TenT& a);

  /**
   * @brief Create a tensor from a container/range (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @tparam RandomIt Random access iterator type
   * @tparam Func Function type for coordinate to index mapping
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param init_elems_begin Iterator to beginning of elements
   * @param coors2idx Function to convert coordinates to linear index
   * @return TenT Created tensor
   */
  template <typename TenT, typename RandomIt, typename Func>
  TenT assign_from_container(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                             RandomIt init_elems_begin, Func&& coors2idx);

  /**
   * @brief Create a tensor with random values (in-place version)
   *
   * @tparam TenT Tensor type
   * @tparam RandNumGen Random number generator type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param gen Random number generator
   * @param a Output tensor
   */
  template <typename TenT, typename RandNumGen>
  void random(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, RandNumGen&& gen, TenT& a) {
    allocate(ctx, shape, a);
    auto& storage = a.storage();
    const auto total = storage.size();

    // Fill tensor with random values using perfect forwarding to support any callable type
    // std::invoke enables compatibility with lambdas, function pointers, functors, and std::function
    for (cytnx::cytnx_uint64 idx = 0; idx < total; ++idx) {
      // 'template' keyword required for dependent template name resolution in template context
      storage.template at<elem_t<TenT>>(idx) = static_cast<elem_t<TenT>>(std::invoke(gen));
    }
  }

  /**
   * @brief Create a tensor with random values (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @tparam RandNumGen Random number generator type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param gen Random number generator
   * @return TenT Random-filled tensor
   */
  template <typename TenT, typename RandNumGen>
  TenT random(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, RandNumGen&& gen) {
    TenT result;
    // Forward generator to in-place version to avoid duplication and maintain perfect forwarding
    random(ctx, shape, std::forward<RandNumGen>(gen), result);
    return result;
  }


  /**
   * @brief Create an identity matrix (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param N Size of the square matrix
   * @param a Output tensor
   */
  template <typename TenT> void eye(context_handle_t<TenT>& ctx, const bond_dim_t<TenT> N, TenT& a);

  /**
   * @brief Create an identity matrix (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param N Size of the square matrix
   * @return TenT Identity matrix
   */
  template <typename TenT> TenT eye(context_handle_t<TenT>& ctx, const bond_dim_t<TenT> N);

  /**
   * @brief Fill tensor with a constant value (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param v Fill value
   * @param a Output tensor
   */
  template <typename TenT>
  void fill(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, const elem_t<TenT> v, TenT& a);

  /**
   * @brief Fill tensor with a constant value (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param v Fill value
   * @return TenT Filled tensor
   */
  template <typename TenT>
  TenT fill(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, const elem_t<TenT> v);

  /**
   * @brief Deep copy a tensor (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param orig Original tensor to copy
   * @param dist Destination tensor
   */
  template <typename TenT> void copy(context_handle_t<TenT>& ctx, const TenT& orig, TenT& dist);

  /**
   * @brief Deep copy a tensor (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param orig Original tensor to copy
   * @return TenT Copied tensor
   */
  template <typename TenT> TenT copy(context_handle_t<TenT>& ctx, const TenT& orig);

  /**
   * @brief Move tensor contents (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param from Source tensor (will be empty after move)
   * @param to Destination tensor
   */
  template <typename TenT> void move(context_handle_t<TenT>& ctx, TenT& from, TenT& to);

  /**
   * @brief Move tensor contents (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param from Source tensor (will be empty after move)
   * @return TenT Moved tensor
   */
  template <typename TenT> TenT move(context_handle_t<TenT>& ctx, TenT& from);

  /**
   * @brief Clear tensor contents
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Tensor to clear
   */
  template <typename TenT> void clear(context_handle_t<TenT>& ctx, TenT& a);

}  // namespace tci
