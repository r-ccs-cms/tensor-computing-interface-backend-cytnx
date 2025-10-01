#pragma once

#include "tci/tensor_traits.h"
#include "tci/read_only_getters.h"
#include "tci/variant_helpers.h"

namespace tci {

  /**
   * @brief Create context handle (PDF spec compatible)
   *
   * @tparam ContextHandleT Context handle type
   * @param implementation_defined_params Implementation-specific parameters
   * @return ContextHandleT Created context handle
   */
  template <typename ContextHandleT> ContextHandleT create_context(/* implementation-defined */);

  /**
   * @brief Create context handle (in-place version)
   *
   * @tparam ContextHandleT Context handle type
   * @param ctx Context handle to initialize
   */
  template <typename ContextHandleT> void create_context(ContextHandleT& ctx);

  /**
   * @brief Destroy context handle
   *
   * @tparam ContextHandleT Context handle type
   * @param ctx Context handle to destroy
   */
  template <typename ContextHandleT> void destroy_context(ContextHandleT& ctx);

  /**
   * @brief Copy tensor elements to a container/range
   *
   * @tparam TenT Tensor type
   * @tparam RandomIt Random access iterator type
   * @tparam Func Function type for coordinate to index mapping
   * @param ctx Context handle for the tensor library
   * @param a Input tensor
   * @param first Iterator to beginning of destination range
   * @param coors2idx Function to convert coordinates to linear index
   */
  template <typename TenT, typename RandomIt, typename Func>
  void to_container(context_handle_t<TenT>& ctx, const TenT& a, RandomIt first, Func&& coors2idx) {
    const auto ten_shape = shape(ctx, a);
    const auto total_size = size(ctx, a);

    // Create coordinate vector for iteration
    elem_coors_t<TenT> coors(ten_shape.size(), 0);

    for (size_t flat_idx = 0; flat_idx < total_size; ++flat_idx) {
      // Get element at current coordinates
      auto elem = get_elem(ctx, a, coors);

      // Use lambda to convert coordinates to container index
      auto container_idx = std::invoke(coors2idx, coors);

      // Store element in container (convert variant to appropriate type)
      *(first + container_idx) = static_cast<typename std::iterator_traits<RandomIt>::value_type>(
          tci::to_complex128(elem));

      // Advance to next coordinate (row-major order)
      for (int dim = static_cast<int>(coors.size()) - 1; dim >= 0; --dim) {
        if (++coors[dim] < ten_shape[dim]) {
          break;
        }
        coors[dim] = 0;
      }
    }
  }

  /**
   * @brief Print tensor contents in human-readable format
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Tensor to print
   */
  template <typename TenT> void show(context_handle_t<TenT>& ctx, const TenT& a);

  /**
   * @brief Check if two tensors are equal within tolerance
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a First tensor
   * @param b Second tensor
   * @param epsilon Tolerance for comparison
   * @return bool True if tensors are equal within tolerance
   */
  template <typename TenT>
  bool eq(context_handle_t<TenT>& ctx, const TenT& a, const TenT& b, const elem_t<TenT> epsilon);

  /**
   * @brief Convert tensor between different types
   *
   * @tparam Ten1T First tensor type
   * @tparam Ten2T Second tensor type
   * @param ctx1 Context handle for first tensor type
   * @param t1 Input tensor
   * @param ctx2 Context handle for second tensor type
   * @param t2 Output tensor
   */
  template <typename Ten1T, typename Ten2T> void convert(context_handle_t<Ten1T>& ctx1,
                                                         const Ten1T& t1,
                                                         context_handle_t<Ten2T>& ctx2, Ten2T& t2);

}  // namespace tci

// Include implementation for CytnxTensor type conversions
#include "tci/detail/convert_impl.h"