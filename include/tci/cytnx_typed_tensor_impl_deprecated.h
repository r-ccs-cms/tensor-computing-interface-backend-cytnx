#pragma once

// Backward-compatibility wrappers for the old ElemT-parameterized API.
//
// Each function here is identical to its TenT-based counterpart in
// cytnx_typed_tensor_impl.h except that the template parameter is ElemT and
// the concrete tensor type CytnxTensor<ElemT> is written out explicitly.
// All wrappers are marked [[deprecated]] to guide callers toward the
// canonical TenT-parameterized API.
//
// To migrate: replace every occurrence of
//   tci::foo<ElemT>(ctx, ...)
// with
//   tci::foo<CytnxTensor<ElemT>>(ctx, ...)   // explicit instantiation, OR
//   tci::foo(ctx, ...)                        // let the compiler deduce TenT

#include "tci/cytnx_typed_tensor_impl.h"
#include "tci/miscellaneous.h"

namespace tci {

// Shared deprecation message used by every wrapper in this file.
#define TCI_DEPRECATED_ELEMT_API \
  "Use TenT-based API: replace template parameter ElemT with TenT = CytnxTensor<ElemT>"

// ============================================================
// Construction and Destruction
// ============================================================

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const shape_t<CytnxTensor<ElemT>>& shape, CytnxTensor<ElemT>& a) {
  allocate<CytnxTensor<ElemT>>(ctx, shape, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                             const shape_t<CytnxTensor<ElemT>>& shape) {
  return allocate<CytnxTensor<ElemT>>(ctx, shape);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void zeros(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const shape_t<CytnxTensor<ElemT>>& shape, CytnxTensor<ElemT>& a) {
  zeros<CytnxTensor<ElemT>>(ctx, shape, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> zeros(context_handle_t<CytnxTensor<ElemT>>& ctx,
                          const shape_t<CytnxTensor<ElemT>>& shape) {
  return zeros<CytnxTensor<ElemT>>(ctx, shape);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void eye(context_handle_t<CytnxTensor<ElemT>>& ctx,
         bond_dim_t<CytnxTensor<ElemT>> dim, CytnxTensor<ElemT>& a) {
  eye<CytnxTensor<ElemT>>(ctx, dim, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> eye(context_handle_t<CytnxTensor<ElemT>>& ctx,
                        bond_dim_t<CytnxTensor<ElemT>> N) {
  return eye<CytnxTensor<ElemT>>(ctx, N);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void fill(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const shape_t<CytnxTensor<ElemT>>& shape,
          elem_t<CytnxTensor<ElemT>> value, CytnxTensor<ElemT>& a) {
  fill<CytnxTensor<ElemT>>(ctx, shape, value, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> fill(context_handle_t<CytnxTensor<ElemT>>& ctx,
                         const shape_t<CytnxTensor<ElemT>>& shape,
                         elem_t<CytnxTensor<ElemT>> value) {
  return fill<CytnxTensor<ElemT>>(ctx, shape, value);
}

template <typename ElemT, typename RandNumGen>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> random(context_handle_t<CytnxTensor<ElemT>>& ctx,
                           const shape_t<CytnxTensor<ElemT>>& shape,
                           RandNumGen&& gen) {
  return random<CytnxTensor<ElemT>>(ctx, shape, std::forward<RandNumGen>(gen));
}

template <typename ElemT, typename RandNumGen>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void random(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const shape_t<CytnxTensor<ElemT>>& shape,
            RandNumGen&& gen, CytnxTensor<ElemT>& a) {
  random<CytnxTensor<ElemT>>(ctx, shape, std::forward<RandNumGen>(gen), a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> copy(context_handle_t<CytnxTensor<ElemT>>& ctx,
                         const CytnxTensor<ElemT>& orig) {
  return copy<CytnxTensor<ElemT>>(ctx, orig);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void copy(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& orig, CytnxTensor<ElemT>& dist) {
  copy<CytnxTensor<ElemT>>(ctx, orig, dist);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
CytnxTensor<ElemT> move(context_handle_t<CytnxTensor<ElemT>>& ctx,
                         CytnxTensor<ElemT>& from) {
  return move<CytnxTensor<ElemT>>(ctx, from);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void move(context_handle_t<CytnxTensor<ElemT>>& ctx,
          CytnxTensor<ElemT>& from, CytnxTensor<ElemT>& to) {
  move<CytnxTensor<ElemT>>(ctx, from, to);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void clear(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& a) {
  clear<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT, typename RandomIt, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void assign_from_container(context_handle_t<CytnxTensor<ElemT>>& ctx,
                           const shape_t<CytnxTensor<ElemT>>& shape,
                           RandomIt init_elems_begin, Func&& coors2idx,
                           CytnxTensor<ElemT>& a) {
  assign_from_container<CytnxTensor<ElemT>>(ctx, shape, init_elems_begin,
                                            std::forward<Func>(coors2idx), a);
}

// ============================================================
// Context Management
// ============================================================

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void create_context(context_handle_t<CytnxTensor<ElemT>>& ctx) {
  create_context<CytnxTensor<ElemT>>(ctx);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void destroy_context(context_handle_t<CytnxTensor<ElemT>>& ctx) {
  destroy_context<CytnxTensor<ElemT>>(ctx);
}

// ============================================================
// Read-only Getters
// ============================================================

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& a,
              const elem_coors_t<CytnxTensor<ElemT>>& coors,
              elem_t<CytnxTensor<ElemT>>& elem) {
  get_elem<CytnxTensor<ElemT>>(ctx, a, coors, elem);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
elem_t<CytnxTensor<ElemT>> get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                     const CytnxTensor<ElemT>& a,
                                     const elem_coors_t<CytnxTensor<ElemT>>& coors) {
  return get_elem<CytnxTensor<ElemT>>(ctx, a, coors);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
shape_t<CytnxTensor<ElemT>> shape(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                   const CytnxTensor<ElemT>& a) {
  return shape<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
order_t<CytnxTensor<ElemT>> order(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                   const CytnxTensor<ElemT>& a) {
  return order<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
order_t<CytnxTensor<ElemT>> rank(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                  const CytnxTensor<ElemT>& a) {
  return rank<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
ten_size_t<CytnxTensor<ElemT>> size(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                     const CytnxTensor<ElemT>& a) {
  return size<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
std::size_t size_bytes(context_handle_t<CytnxTensor<ElemT>>& ctx,
                       const CytnxTensor<ElemT>& a) {
  return size_bytes<CytnxTensor<ElemT>>(ctx, a);
}

// ============================================================
// Tensor Manipulation
// ============================================================

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void set_elem(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& a,
              const elem_coors_t<CytnxTensor<ElemT>>& coors,
              elem_t<CytnxTensor<ElemT>> elem) {
  set_elem<CytnxTensor<ElemT>>(ctx, a, coors, elem);
}

template <typename ElemT, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
              Func&& f) {
  for_each<CytnxTensor<ElemT>>(ctx, inout, std::forward<Func>(f));
}

template <typename ElemT, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& in, Func&& f) {
  for_each<CytnxTensor<ElemT>>(ctx, in, std::forward<Func>(f));
}

template <typename ElemT, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void for_each_with_coors(context_handle_t<CytnxTensor<ElemT>>& ctx,
                         CytnxTensor<ElemT>& inout, Func&& f) {
  for_each_with_coors<CytnxTensor<ElemT>>(ctx, inout, std::forward<Func>(f));
}

template <typename ElemT, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void for_each_with_coors(context_handle_t<CytnxTensor<ElemT>>& ctx,
                         const CytnxTensor<ElemT>& in, Func&& f) {
  for_each_with_coors<CytnxTensor<ElemT>>(ctx, in, std::forward<Func>(f));
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void reshape(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
             const shape_t<CytnxTensor<ElemT>>& new_shape) {
  reshape<CytnxTensor<ElemT>>(ctx, inout, new_shape);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void reshape(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const CytnxTensor<ElemT>& in,
             const shape_t<CytnxTensor<ElemT>>& new_shape, CytnxTensor<ElemT>& out) {
  reshape<CytnxTensor<ElemT>>(ctx, in, new_shape, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void transpose(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
               const std::vector<bond_idx_t<CytnxTensor<ElemT>>>& new_order) {
  transpose<CytnxTensor<ElemT>>(ctx, inout, new_order);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void transpose(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& in,
               const std::vector<bond_idx_t<CytnxTensor<ElemT>>>& new_order,
               CytnxTensor<ElemT>& out) {
  transpose<CytnxTensor<ElemT>>(ctx, in, new_order, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void cplx_conj(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout) {
  cplx_conj<CytnxTensor<ElemT>>(ctx, inout);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void cplx_conj(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& in, CytnxTensor<ElemT>& out) {
  cplx_conj<CytnxTensor<ElemT>>(ctx, in, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void real(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& in, real_ten_t<CytnxTensor<ElemT>>& out) {
  real<CytnxTensor<ElemT>>(ctx, in, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
real_ten_t<CytnxTensor<ElemT>> real(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                     const CytnxTensor<ElemT>& in) {
  return real<CytnxTensor<ElemT>>(ctx, in);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void imag(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& in, real_ten_t<CytnxTensor<ElemT>>& out) {
  imag<CytnxTensor<ElemT>>(ctx, in, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
real_ten_t<CytnxTensor<ElemT>> imag(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                     const CytnxTensor<ElemT>& in) {
  return imag<CytnxTensor<ElemT>>(ctx, in);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void to_cplx(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const CytnxTensor<ElemT>& in, cplx_ten_t<CytnxTensor<ElemT>>& out) {
  to_cplx<CytnxTensor<ElemT>>(ctx, in, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
cplx_ten_t<CytnxTensor<ElemT>> to_cplx(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                         const CytnxTensor<ElemT>& in) {
  return to_cplx<CytnxTensor<ElemT>>(ctx, in);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
real_t<CytnxTensor<ElemT>> norm(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                 const CytnxTensor<ElemT>& a) {
  return norm<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
real_t<CytnxTensor<ElemT>> normalize(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      CytnxTensor<ElemT>& inout) {
  return normalize<CytnxTensor<ElemT>>(ctx, inout);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
real_t<CytnxTensor<ElemT>> normalize(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& in,
                                      CytnxTensor<ElemT>& out) {
  return normalize<CytnxTensor<ElemT>>(ctx, in, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void scale(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
           const elem_t<CytnxTensor<ElemT>> s) {
  scale<CytnxTensor<ElemT>>(ctx, inout, s);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void scale(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const CytnxTensor<ElemT>& in,
           const elem_t<CytnxTensor<ElemT>> s, CytnxTensor<ElemT>& out) {
  scale<CytnxTensor<ElemT>>(ctx, in, s, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void diag(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout) {
  diag<CytnxTensor<ElemT>>(ctx, inout);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void diag(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& in, CytnxTensor<ElemT>& out) {
  diag<CytnxTensor<ElemT>>(ctx, in, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trace(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
           const bond_idx_pairs_t<CytnxTensor<ElemT>>& bdidx_pairs) {
  trace<CytnxTensor<ElemT>>(ctx, inout, bdidx_pairs);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trace(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const CytnxTensor<ElemT>& in,
           const bond_idx_pairs_t<CytnxTensor<ElemT>>& bdidx_pairs,
           CytnxTensor<ElemT>& out) {
  trace<CytnxTensor<ElemT>>(ctx, in, bdidx_pairs, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void expand(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
            const Map<bond_idx_t<CytnxTensor<ElemT>>,
                      bond_dim_t<CytnxTensor<ElemT>>>& bond_idx_increment_map) {
  expand<CytnxTensor<ElemT>>(ctx, inout, bond_idx_increment_map);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void expand(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& in,
            const Map<bond_idx_t<CytnxTensor<ElemT>>,
                      bond_dim_t<CytnxTensor<ElemT>>>& bond_idx_increment_map,
            CytnxTensor<ElemT>& out) {
  expand<CytnxTensor<ElemT>>(ctx, in, bond_idx_increment_map, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
            const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map) {
  shrink<CytnxTensor<ElemT>>(ctx, inout, bd_idx_el_coor_pair_map);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& in,
            const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map,
            CytnxTensor<ElemT>& out) {
  shrink<CytnxTensor<ElemT>>(ctx, in, bd_idx_el_coor_pair_map, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void extract_sub(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
                 const List<Pair<elem_coor_t<CytnxTensor<ElemT>>,
                                 elem_coor_t<CytnxTensor<ElemT>>>>& coor_pairs) {
  extract_sub<CytnxTensor<ElemT>>(ctx, inout, coor_pairs);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void extract_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 const CytnxTensor<ElemT>& in,
                 const List<Pair<elem_coor_t<CytnxTensor<ElemT>>,
                                 elem_coor_t<CytnxTensor<ElemT>>>>& coor_pairs,
                 CytnxTensor<ElemT>& out) {
  extract_sub<CytnxTensor<ElemT>>(ctx, in, coor_pairs, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
                 const CytnxTensor<ElemT>& sub,
                 const elem_coors_t<CytnxTensor<ElemT>>& begin_pt) {
  replace_sub<CytnxTensor<ElemT>>(ctx, inout, sub, begin_pt);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 const CytnxTensor<ElemT>& in,
                 const CytnxTensor<ElemT>& sub,
                 const elem_coors_t<CytnxTensor<ElemT>>& begin_pt,
                 CytnxTensor<ElemT>& out) {
  replace_sub<CytnxTensor<ElemT>>(ctx, in, sub, begin_pt, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void concatenate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 const List<CytnxTensor<ElemT>>& ins,
                 const bond_idx_t<CytnxTensor<ElemT>> axis,
                 CytnxTensor<ElemT>& out) {
  concatenate<CytnxTensor<ElemT>>(ctx, ins, axis, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void stack(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const List<CytnxTensor<ElemT>>& ins,
           const bond_idx_t<CytnxTensor<ElemT>> axis,
           CytnxTensor<ElemT>& out) {
  stack<CytnxTensor<ElemT>>(ctx, ins, axis, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void inverse(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
             const order_t<CytnxTensor<ElemT>> num_of_bds_as_row) {
  inverse<CytnxTensor<ElemT>>(ctx, inout, num_of_bds_as_row);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void inverse(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const CytnxTensor<ElemT>& in,
             const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
             CytnxTensor<ElemT>& out) {
  inverse<CytnxTensor<ElemT>>(ctx, in, num_of_bds_as_row, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void exp(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
         const order_t<CytnxTensor<ElemT>> num_of_bds_as_row) {
  exp<CytnxTensor<ElemT>>(ctx, inout, num_of_bds_as_row);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void exp(context_handle_t<CytnxTensor<ElemT>>& ctx,
         const CytnxTensor<ElemT>& in,
         const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
         CytnxTensor<ElemT>& out) {
  exp<CytnxTensor<ElemT>>(ctx, in, num_of_bds_as_row, out);
}

// ============================================================
// Tensor Linear Algebra
// ============================================================

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void contract(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& a,
              const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_a,
              const CytnxTensor<ElemT>& b,
              const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_b,
              CytnxTensor<ElemT>& c,
              const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_c) {
  contract<CytnxTensor<ElemT>>(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void contract(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& a, const std::string_view bd_labs_str_a,
              const CytnxTensor<ElemT>& b, const std::string_view bd_labs_str_b,
              CytnxTensor<ElemT>& c, const std::string_view bd_labs_str_c) {
  contract<CytnxTensor<ElemT>>(ctx, a, bd_labs_str_a, b, bd_labs_str_b, c, bd_labs_str_c);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void linear_combine(context_handle_t<CytnxTensor<ElemT>>& ctx,
                    const std::vector<CytnxTensor<ElemT>>& ins,
                    CytnxTensor<ElemT>& out) {
  linear_combine<CytnxTensor<ElemT>>(ctx, ins, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void linear_combine(context_handle_t<CytnxTensor<ElemT>>& ctx,
                    const std::vector<CytnxTensor<ElemT>>& ins,
                    const std::vector<elem_t<CytnxTensor<ElemT>>>& coefs,
                    CytnxTensor<ElemT>& out) {
  linear_combine<CytnxTensor<ElemT>>(ctx, ins, coefs, out);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
         const CytnxTensor<ElemT>& a,
         const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
         CytnxTensor<ElemT>& u,
         real_ten_t<CytnxTensor<ElemT>>& s_diag,
         CytnxTensor<ElemT>& v_dag) {
  svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void qr(context_handle_t<CytnxTensor<ElemT>>& ctx,
        const CytnxTensor<ElemT>& a,
        const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
        CytnxTensor<ElemT>& q, CytnxTensor<ElemT>& r) {
  qr<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, q, r);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void lq(context_handle_t<CytnxTensor<ElemT>>& ctx,
        const CytnxTensor<ElemT>& a,
        const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
        CytnxTensor<ElemT>& l, CytnxTensor<ElemT>& q) {
  lq<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, l, q);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& a,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               CytnxTensor<ElemT>& u,
               real_ten_t<CytnxTensor<ElemT>>& s_diag,
               CytnxTensor<ElemT>& v_dag,
               real_t<CytnxTensor<ElemT>>& trunc_err,
               const bond_dim_t<CytnxTensor<ElemT>> chi_max,
               const real_t<CytnxTensor<ElemT>> s_min) {
  trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err,
                                 chi_max, s_min);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& a,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               CytnxTensor<ElemT>& u,
               real_ten_t<CytnxTensor<ElemT>>& s_diag,
               CytnxTensor<ElemT>& v_dag,
               real_t<CytnxTensor<ElemT>>& trunc_err,
               const real_t<CytnxTensor<ElemT>> target_trunc_err,
               const real_t<CytnxTensor<ElemT>> s_min) {
  trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err,
                                 target_trunc_err, s_min);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& a,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               CytnxTensor<ElemT>& u,
               real_ten_t<CytnxTensor<ElemT>>& s_diag,
               CytnxTensor<ElemT>& v_dag,
               real_t<CytnxTensor<ElemT>>& trunc_err,
               const bond_dim_t<CytnxTensor<ElemT>> chi_min,
               const bond_dim_t<CytnxTensor<ElemT>> chi_max,
               const real_t<CytnxTensor<ElemT>> target_trunc_err,
               const real_t<CytnxTensor<ElemT>> s_min) {
  trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err,
                                 chi_min, chi_max, target_trunc_err, s_min);
}

// Deprecated old parameter-order overloads (wrapping the already-deprecated TenT versions)
template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& a,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               CytnxTensor<ElemT>& u,
               real_ten_t<CytnxTensor<ElemT>>& s_diag,
               CytnxTensor<ElemT>& v_dag,
               real_t<CytnxTensor<ElemT>>& trunc_err,
               const real_t<CytnxTensor<ElemT>> s_min) {
  trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err,
                                 s_min);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& a,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               CytnxTensor<ElemT>& u,
               real_ten_t<CytnxTensor<ElemT>>& s_diag,
               CytnxTensor<ElemT>& v_dag,
               real_t<CytnxTensor<ElemT>>& trunc_err,
               const bond_dim_t<CytnxTensor<ElemT>> chi_max,
               const real_t<CytnxTensor<ElemT>> target_trunc_err,
               const real_t<CytnxTensor<ElemT>> s_min) {
  trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err,
                                 chi_max, target_trunc_err, s_min);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void eigvals(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const CytnxTensor<ElemT>& a,
             const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
             cplx_ten_t<CytnxTensor<ElemT>>& w_diag) {
  eigvals<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void eigvalsh(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& a,
              const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
              real_ten_t<CytnxTensor<ElemT>>& w_diag) {
  eigvalsh<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void eig(context_handle_t<CytnxTensor<ElemT>>& ctx,
         const CytnxTensor<ElemT>& a,
         const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
         cplx_ten_t<CytnxTensor<ElemT>>& w_diag,
         cplx_ten_t<CytnxTensor<ElemT>>& v) {
  eig<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag, v);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void eigh(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& a,
          const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
          real_ten_t<CytnxTensor<ElemT>>& w_diag,
          CytnxTensor<ElemT>& v) {
  eigh<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag, v);
}

// ============================================================
// Miscellaneous
// ============================================================

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void show(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a) {
  show<CytnxTensor<ElemT>>(ctx, a);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
bool close(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const CytnxTensor<ElemT>& a, const CytnxTensor<ElemT>& b,
           const real_t<CytnxTensor<ElemT>> epsilon) {
  return close<CytnxTensor<ElemT>>(ctx, a, b, epsilon);
}

template <typename ElemT>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
bool eq(context_handle_t<CytnxTensor<ElemT>>& ctx,
        const CytnxTensor<ElemT>& a, const CytnxTensor<ElemT>& b,
        const real_t<CytnxTensor<ElemT>> epsilon) {
  return eq<CytnxTensor<ElemT>>(ctx, a, b, epsilon);
}

template <typename ElemT, typename RandomIt, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void to_range(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& a, RandomIt first, Func&& coors2idx) {
  to_range<CytnxTensor<ElemT>>(ctx, a, first, std::forward<Func>(coors2idx));
}

template <typename ElemT, typename RandomIt, typename Func>
[[deprecated(TCI_DEPRECATED_ELEMT_API)]]
void to_container(context_handle_t<CytnxTensor<ElemT>>& ctx,
                  const CytnxTensor<ElemT>& a, RandomIt first, Func&& coors2idx) {
  to_container<CytnxTensor<ElemT>>(ctx, a, first, std::forward<Func>(coors2idx));
}

#undef TCI_DEPRECATED_ELEMT_API

}  // namespace tci
