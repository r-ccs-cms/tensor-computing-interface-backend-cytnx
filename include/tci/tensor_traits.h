#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <complex>
#include <cstddef>

namespace tci {

// Forward declaration of tensor_traits
template <typename TenT>
struct tensor_traits;

// Auxiliary types as specified in the TCI standard

// List type - currently restricted to std::vector
template <typename T>
using List = std::vector<T>;

// Pair type - currently restricted to std::pair
template <typename T, typename U>
using Pair = std::pair<T, U>;

// Map type - currently restricted to std::unordered_map
template <typename T, typename U>
using Map = std::unordered_map<T, U>;

// Convenience type aliases for extracting types from tensor_traits
template <typename TenT>
using ten_t = typename tensor_traits<TenT>::ten_t;

template <typename TenT>
using rank_t = typename tensor_traits<TenT>::rank_t;

template <typename TenT>
using shape_t = typename tensor_traits<TenT>::shape_t;

template <typename TenT>
using bond_dim_t = typename tensor_traits<TenT>::bond_dim_t;

template <typename TenT>
using bond_idx_t = typename tensor_traits<TenT>::bond_idx_t;

template <typename TenT>
using bond_label_t = typename tensor_traits<TenT>::bond_label_t;

template <typename TenT>
using ten_size_t = typename tensor_traits<TenT>::ten_size_t;

template <typename TenT>
using elem_t = typename tensor_traits<TenT>::elem_t;

template <typename TenT>
using elem_coor_t = typename tensor_traits<TenT>::elem_coor_t;

template <typename TenT>
using elem_coors_t = typename tensor_traits<TenT>::elem_coors_t;

template <typename TenT>
using real_t = typename tensor_traits<TenT>::real_t;

template <typename TenT>
using real_ten_t = typename tensor_traits<TenT>::real_ten_t;

template <typename TenT>
using cplx_t = typename tensor_traits<TenT>::cplx_t;

template <typename TenT>
using cplx_ten_t = typename tensor_traits<TenT>::cplx_ten_t;

template <typename TenT>
using context_handle_t = typename tensor_traits<TenT>::context_handle_t;

// Composite type aliases as specified in TCI standard
template <typename TenT>
using bond_idx_pairs_t = List<Pair<bond_idx_t<TenT>, bond_idx_t<TenT>>>;

template <typename TenT>
using bond_idx_elem_coor_pair_map = Map<bond_idx_t<TenT>, Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>;

/**
 * @brief Primary template for tensor_traits
 *
 * This struct must be specialized for each tensor library implementation.
 * It provides type information and context handling for the underlying tensor type.
 *
 * @tparam TenT The tensor type from the underlying library
 */
template <typename TenT>
struct tensor_traits {
    // Member types that must be defined in specializations:
    // - ten_t: The actual type of TenT
    // - rank_t: Type for tensor rank (integer)
    // - shape_t: Type for tensor shape (List<bond_dim_t>)
    // - bond_dim_t: Type for bond dimension (integer)
    // - bond_idx_t: Type for bond index (integer)
    // - bond_label_t: Type for bond label (signed integer)
    // - ten_size_t: Type for tensor size (integer)
    // - elem_t: Type for tensor elements (real or complex floating point)
    // - elem_coor_t: Type for element coordinate (integer)
    // - elem_coors_t: Type for element coordinates (List<elem_coor_t>)
    // - real_t: Type for real floating point numbers
    // - real_ten_t: Type for real tensors
    // - cplx_t: Type for complex floating point numbers
    // - cplx_ten_t: Type for complex tensors
    // - context_handle_t: Type for library context handle

    static_assert(sizeof(TenT) == 0, "tensor_traits must be specialized for each tensor type");
};

} // namespace tci