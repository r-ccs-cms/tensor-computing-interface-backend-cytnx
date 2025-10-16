#pragma once

#include <complex>
#include <functional>
#include <type_traits>
#include <utility>

#include "tci/cytnx_tensor_traits.h"
#include "tci/read_only_getters.h"
#include "tci/tensor_manipulation.h"
#include "tci/tensor_traits.h"

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
   * @brief Create a tensor from a range
   *
   * @tparam TenT Tensor type
   * @tparam RandomIt Random access iterator type
   * @tparam Func Function type for coordinate to index mapping
   * @param ctx Context handle for the tensor library
   * @param shape Shape of the tensor
   * @param first Iterator to beginning of elements
   * @param coors2idx Function to convert coordinates to linear index
   * @return TenT Created tensor
   */
  template <typename TenT, typename RandomIt, typename Func>
  TenT assign_from_range(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                         RandomIt first, Func&& coors2idx);

  /**
   * @brief Create a tensor from a container/range (deprecated - use assign_from_range)
   *
   * @deprecated Use assign_from_range instead. This API will be removed in the next major version
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
  [[deprecated("Use assign_from_range instead. This API will be removed in the next major version")]]
  TenT assign_from_container(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                             RandomIt init_elems_begin, Func&& coors2idx);

  // Template function implementations for assign_from_range
  // Generic implementation that works with any tensor type

  template <typename TenT, typename RandomIt, typename Func>
  TenT assign_from_range(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                         RandomIt first, Func&& coors2idx) {
    TenT a;
    // Create tensor with the specified shape
    allocate(ctx, shape, a);

    // Generate all coordinate combinations and assign values
    std::function<void(elem_coors_t<TenT>, std::size_t)> assign_recursive;
    assign_recursive = [&](elem_coors_t<TenT> current_coords, std::size_t dim) {
      if (dim == shape.size()) {
        // Base case: all dimensions set, assign the element
        auto index = std::invoke(coors2idx, current_coords);
        auto value = *(first + index);
        elem_t<TenT> elem_val = static_cast<elem_t<TenT>>(value);

        set_elem(ctx, a, current_coords, elem_val);
      } else {
        // Recursive case: iterate through current dimension
        for (bond_dim_t<TenT> i = 0; i < shape[dim]; ++i) {
          current_coords.push_back(i);
          assign_recursive(current_coords, dim + 1);
          current_coords.pop_back();
        }
      }
    };

    assign_recursive({}, 0);
    return a;
  }

  // Deprecated: Forward to assign_from_range
  template <typename TenT, typename RandomIt, typename Func>
  TenT assign_from_container(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                             RandomIt init_elems_begin, Func&& coors2idx) {
    return assign_from_range<TenT, RandomIt, Func>(ctx, shape, init_elems_begin, std::forward<Func>(coors2idx));
  }

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

    // Use coordinate-based approach to avoid storage compatibility issues
    auto total_size = size(ctx, a);
    elem_coors_t<TenT> coords(shape.size(), 0);

    for (size_t flat_idx = 0; flat_idx < total_size; ++flat_idx) {
      // Generate random value and convert to appropriate elem_t
      auto raw_val = std::invoke(gen);
      elem_t<TenT> elem_val = static_cast<elem_t<TenT>>(raw_val);

      set_elem(ctx, a, coords, elem_val);

      // Advance to next coordinate (row-major order)
      for (int dim = static_cast<int>(coords.size()) - 1; dim >= 0; --dim) {
        if (++coords[dim] < shape[dim]) {
          break;
        }
        coords[dim] = 0;
      }
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
