#pragma once

#include "tci/tensor_traits.h"
#include "tci/cytnx_tensor_traits.h"

namespace tci {

  /**
   * @brief Set the value of a specific element
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param a Tensor to modify
   * @param coors Coordinates of the element
   * @param el New element value
   */
  template <typename TenT> void set_elem(context_handle_t<TenT>& ctx, TenT& a,
                                         const elem_coors_t<TenT>& coors, const elem_t<TenT> el);

  /**
   * @brief Reshape tensor (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to reshape
   * @param new_shape New shape for the tensor
   */
  template <typename TenT>
  void reshape(context_handle_t<TenT>& ctx, TenT& inout, const shape_t<TenT>& new_shape);

  /**
   * @brief Reshape tensor (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param new_shape New shape for the tensor
   * @param out Output tensor
   */
  template <typename TenT> void reshape(context_handle_t<TenT>& ctx, const TenT& in,
                                        const shape_t<TenT>& new_shape, TenT& out);

  /**
   * @brief Transpose tensor (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to transpose
   * @param new_order New order of bonds
   */
  template <typename TenT>
  void transpose(context_handle_t<TenT>& ctx, TenT& inout, const List<bond_idx_t<TenT>>& new_order);

  /**
   * @brief Transpose tensor (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param new_order New order of bonds
   * @param out Output tensor
   */
  template <typename TenT> void transpose(context_handle_t<TenT>& ctx, const TenT& in,
                                          const List<bond_idx_t<TenT>>& new_order, TenT& out);

  /**
   * @brief Complex conjugate (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to conjugate
   */
  template <typename TenT> void cplx_conj(context_handle_t<TenT>& ctx, TenT& inout);

  /**
   * @brief Complex conjugate (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param out Output tensor
   */
  template <typename TenT> void cplx_conj(context_handle_t<TenT>& ctx, const TenT& in, TenT& out);

  /**
   * @brief Convert to complex format (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param out Complex tensor output
   */
  template <typename TenT>
  void to_cplx(context_handle_t<TenT>& ctx, const TenT& in, cplx_ten_t<TenT>& out);

  /**
   * @brief Convert to complex format (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @return cplx_ten_t<TenT> Complex tensor
   */
  template <typename TenT> cplx_ten_t<TenT> to_cplx(context_handle_t<TenT>& ctx, const TenT& in);

  /**
   * @brief Extract real part (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param out Real tensor output
   */
  template <typename TenT>
  void real(context_handle_t<TenT>& ctx, const TenT& in, real_ten_t<TenT>& out);

  /**
   * @brief Extract real part (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @return real_ten_t<TenT> Real tensor
   */
  template <typename TenT> real_ten_t<TenT> real(context_handle_t<TenT>& ctx, const TenT& in);

  /**
   * @brief Extract imaginary part (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param out Real tensor output (imaginary parts)
   */
  template <typename TenT>
  void imag(context_handle_t<TenT>& ctx, const TenT& in, real_ten_t<TenT>& out);

  /**
   * @brief Extract imaginary part (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @return real_ten_t<TenT> Real tensor containing imaginary parts
   */
  template <typename TenT> real_ten_t<TenT> imag(context_handle_t<TenT>& ctx, const TenT& in);

  /**
   * @brief Concatenate tensors along a bond
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param ins List of input tensors
   * @param concat_bdidx Bond index for concatenation
   * @param out Output concatenated tensor
   */
  template <typename TenT> void concatenate(context_handle_t<TenT>& ctx, const List<TenT>& ins,
                                            const bond_idx_t<TenT> concat_bdidx, TenT& out);

  /**
   * @brief Stack tensors along a new bond
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param ins List of input tensors
   * @param stack_bdidx Bond index for stacking
   * @param out Output stacked tensor
   */
  template <typename TenT> void stack(context_handle_t<TenT>& ctx, const List<TenT>& ins,
                                      const bond_idx_t<TenT> stack_bdidx, TenT& out);

  /**
   * @brief Apply function to each element (modifying version)
   *
   * @tparam TenT Tensor type
   * @tparam Func Function type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to process
   * @param f Function to apply to each element
   */
  template <typename TenT, typename Func>
  void for_each(context_handle_t<TenT>& ctx, TenT& inout, Func&& f);

  /**
   * @brief Apply function to each element (const version)
   *
   * @tparam TenT Tensor type
   * @tparam Func Function type
   * @param ctx Context handle for the tensor library
   * @param in Tensor to process
   * @param f Function to apply to each element
   */
  template <typename TenT, typename Func>
  void for_each(context_handle_t<TenT>& ctx, const TenT& in, Func&& f);

  /**
   * @brief Expand tensor dimensions (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to expand
   * @param bond_idx_increment_map Map of bond indices to increments
   */
  template <typename TenT>
  void expand(context_handle_t<TenT>& ctx, TenT& inout,
              const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& bond_idx_increment_map);

  /**
   * @brief Expand tensor dimensions (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param bond_idx_increment_map Map of bond indices to increments
   * @param out Output tensor
   */
  template <typename TenT>
  void expand(context_handle_t<TenT>& ctx, const TenT& in,
              const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& bond_idx_increment_map, TenT& out);

  /**
   * @brief Shrink tensor dimensions (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to shrink
   * @param bd_idx_el_coor_pair_map Map of bond indices to coordinate pairs
   */
  template <typename TenT>
  void shrink(context_handle_t<TenT>& ctx, TenT& inout,
              const bond_idx_elem_coor_pair_map<TenT>& bd_idx_el_coor_pair_map);

  /**
   * @brief Shrink tensor dimensions (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param bd_idx_el_coor_pair_map Map of bond indices to coordinate pairs
   * @param out Output tensor
   */
  template <typename TenT>
  void shrink(context_handle_t<TenT>& ctx, const TenT& in,
              const bond_idx_elem_coor_pair_map<TenT>& bd_idx_el_coor_pair_map, TenT& out);

  /**
   * @brief Extract sub-tensor (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to extract from
   * @param coor_pairs Coordinate pairs for extraction ranges
   */
  template <typename TenT>
  void extract_sub(context_handle_t<TenT>& ctx, TenT& inout,
                   const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs);

  /**
   * @brief Extract sub-tensor (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param coor_pairs Coordinate pairs for extraction ranges
   * @param out Output tensor
   */
  template <typename TenT>
  void extract_sub(context_handle_t<TenT>& ctx, const TenT& in,
                   const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs, TenT& out);

  /**
   * @brief Replace sub-tensor (in-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to modify
   * @param sub Sub-tensor to insert
   * @param begin_pt Starting coordinates
   */
  template <typename TenT> void replace_sub(context_handle_t<TenT>& ctx, TenT& inout,
                                            const TenT& sub, const elem_coors_t<TenT>& begin_pt);

  /**
   * @brief Replace sub-tensor (out-of-place version)
   *
   * @tparam TenT Tensor type
   * @param ctx Context handle for the tensor library
   * @param in Input tensor
   * @param sub Sub-tensor to insert
   * @param begin_pt Starting coordinates
   * @param out Output tensor
   */
  template <typename TenT> void replace_sub(context_handle_t<TenT>& ctx, const TenT& in,
                                            const TenT& sub, const elem_coors_t<TenT>& begin_pt,
                                            TenT& out);

  /**
   * @brief Apply function to each element with coordinates (modifying version)
   *
   * @tparam TenT Tensor type
   * @tparam Func Function type
   * @param ctx Context handle for the tensor library
   * @param inout Tensor to process
   * @param f Function to apply to each element with coordinates
   */
  template <typename TenT, typename Func>
  void for_each_with_coors(context_handle_t<TenT>& ctx, TenT& inout, Func&& f);

  /**
   * @brief Apply function to each element with coordinates (const version)
   *
   * @tparam TenT Tensor type
   * @tparam Func Function type
   * @param ctx Context handle for the tensor library
   * @param in Tensor to process
   * @param f Function to apply to each element with coordinates
   */
  template <typename TenT, typename Func>
  void for_each_with_coors(context_handle_t<TenT>& ctx, const TenT& in, Func&& f);

  // Template function implementations for for_each
  // These must be in the header for template instantiation

  template <typename Func>
  void for_each(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout, Func&& f) {
    auto& storage = inout.storage();
    const auto total = storage.size();

    for (cytnx::cytnx_uint64 idx = 0; idx < total; ++idx) {
      auto& elem = storage.at<elem_t<cytnx::Tensor>>(idx);
      f(elem);
    }
  }

  template <typename Func>
  void for_each(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in, Func&& f) {
    const auto& storage = in.storage();
    const auto total = storage.size();

    for (cytnx::cytnx_uint64 idx = 0; idx < total; ++idx) {
      const auto& elem = storage.at<elem_t<cytnx::Tensor>>(idx);
      f(elem);
    }
  }

}  // namespace tci