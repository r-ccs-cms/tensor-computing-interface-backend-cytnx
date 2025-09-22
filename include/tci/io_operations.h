#pragma once

#include "tci/tensor_traits.h"

namespace tci {

/**
 * @brief Load tensor from storage (in-place version)
 *
 * @tparam TenT Tensor type
 * @tparam Storage Storage type (e.g., file path, memory buffer)
 * @param ctx Context handle for the tensor library
 * @param strg Storage object to load from
 * @param a Output tensor
 */
template <typename TenT, typename Storage>
void load(
    context_handle_t<TenT> &ctx,
    Storage &&strg,
    TenT &a
);

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
TenT load(
    context_handle_t<TenT> &ctx,
    Storage &&strg
);

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
void save(
    context_handle_t<TenT> &ctx,
    const TenT &a,
    Storage &strg
);

} // namespace tci