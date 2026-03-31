#pragma once

#include <type_traits>
#include <utility>

#include "tci/tensor_traits.h"

namespace tci {

  namespace detail {

    template <typename...> struct dependent_false_helper : std::false_type {};

    template <typename... Args> inline constexpr bool dependent_false_v
        = dependent_false_helper<Args...>::value;

    template <typename TenT, typename Storage, typename Enable = void> struct storage_adapter {
      template <typename StorageLike>
      static void load(context_handle_t<TenT>&, StorageLike&&, TenT&) {
        static_assert(dependent_false_v<TenT, StorageLike>,
                      "Unsupported storage type for tci::load");
      }

      template <typename StorageLike>
      static TenT load_out(context_handle_t<TenT>& ctx, StorageLike&& strg) {
        TenT result{};
        load(ctx, std::forward<StorageLike>(strg), result);
        return result;
      }

      template <typename StorageLike>
      static void save(context_handle_t<TenT>&, const TenT&, StorageLike&) {
        static_assert(dependent_false_v<TenT, StorageLike>,
                      "Unsupported storage type for tci::save");
      }
    };

  }  // namespace detail

  /**
   * @brief Load tensor from storage (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @tparam Storage Storage type (e.g., file path, memory buffer)
   * @param ctx Context handle for the tensor library
   * @param strg Storage object to load from
   * @return TenT Loaded tensor
   */
  template <typename TenT, typename Storage>
  inline TenT load(context_handle_t<TenT>& ctx, Storage&& strg) {
    return detail::storage_adapter<TenT, std::decay_t<Storage>>::load_out(
        ctx, std::forward<Storage>(strg));
  }

  /**
   * @brief Save tensor to storage
   *
   * @tparam TenT Tensor type
   * @tparam Storage Storage type (e.g., file path, memory buffer)
   * @param ctx Context handle for the tensor library
   * @param a Tensor to save
   * @param strg Storage object to save to
   */
  template <typename TenT, typename Storage>
  inline void save(context_handle_t<TenT>& ctx, const TenT& a, Storage& strg) {
    detail::storage_adapter<TenT, std::decay_t<Storage>>::save(ctx, a, strg);
  }

}  // namespace tci

#include "tci/detail/cytnx_io_adapter.h"
