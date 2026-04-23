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
#include "tci/io_operations.h"
#include "tci/miscellaneous.h"

namespace tci {

  namespace detail {
    template <typename T> struct is_cytnx_tensor : std::false_type {};
    template <typename E> struct is_cytnx_tensor<CytnxTensor<E>> : std::true_type {};
    template <typename T> inline constexpr bool is_cytnx_tensor_v = is_cytnx_tensor<T>::value;
  }  // namespace detail

// Shared deprecation message used by every wrapper in this file.
#define TCI_DEPRECATED_ELEMT_API \
  "Use TenT-based API: replace template parameter ElemT with TenT = CytnxTensor<ElemT>"

  // ============================================================
  // Construction and Destruction — TenT in-place / renamed API
  // ============================================================

  template <typename TenT>
  [[deprecated("Reserved for future GPU support. Use: auto result = tci::fill(ctx, shape, v);")]]
  void fill(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, elem_t<TenT> value, TenT& a) {
    a = allocate<TenT>(ctx, shape);
    auto total_size = static_cast<cytnx::cytnx_uint64>(a.backend.storage().size());
    auto* data = a.backend.storage().template data<elem_t<TenT>>();
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = value;
    }
  }

  template <typename TenT>
  [[deprecated("Reserved for future GPU support. Use: auto result = tci::zeros(ctx, shape);")]]
  void zeros(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, TenT& a) {
    fill(ctx, shape, elem_t<TenT>(0), a);
  }

  template <typename TenT>
  [[deprecated("Reserved for future GPU support. Use: auto result = tci::eye(ctx, N);")]]
  void eye(context_handle_t<TenT>& ctx, bond_dim_t<TenT> dim, TenT& a) {
    a.backend = cytnx::eye(dim, detail::elem_to_cytnx_type<elem_t<TenT>>(), ctx);
  }

  template <typename TenT, typename RandNumGen>
  [[deprecated("Use return-value version instead: auto result = random(ctx, shape, gen)")]]
  void random(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, RandNumGen& gen, TenT& a) {
    a = random<TenT>(ctx, shape, gen);
  }

  template <typename TenT, typename RandomIt, typename Func>
  [[deprecated("Use assign_from_range instead. This API will be removed in the next major version")]]
  TenT assign_from_container(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                             RandomIt init_elems_begin, Func&& coors2idx) {
    return assign_from_range<TenT, RandomIt, Func>(ctx, shape, init_elems_begin,
                                                   std::forward<Func>(coors2idx));
  }

  // ============================================================
  // Construction and Destruction — ElemT wrappers
  // ============================================================

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void allocate(context_handle_t<CytnxTensor<ElemT>>& ctx, const shape_t<CytnxTensor<ElemT>>& shape,
                CytnxTensor<ElemT>& a) {
    allocate<CytnxTensor<ElemT>>(ctx, shape, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                              const shape_t<CytnxTensor<ElemT>>& shape) {
    return allocate<CytnxTensor<ElemT>>(ctx, shape);
  }

  // Generic template for allocate (in-place, deprecated)
  template <typename TenT>
  [[deprecated("Use return-value version instead: auto result = allocate(ctx, shape)")]]
  void allocate(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape, TenT& a) {
    a = allocate<TenT>(ctx, shape);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void zeros(context_handle_t<CytnxTensor<ElemT>>& ctx, const shape_t<CytnxTensor<ElemT>>& shape,
             CytnxTensor<ElemT>& a) {
    zeros<CytnxTensor<ElemT>>(ctx, shape, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> zeros(context_handle_t<CytnxTensor<ElemT>>& ctx,
                           const shape_t<CytnxTensor<ElemT>>& shape) {
    return zeros<CytnxTensor<ElemT>>(ctx, shape);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void eye(context_handle_t<CytnxTensor<ElemT>>& ctx, bond_dim_t<CytnxTensor<ElemT>> dim,
           CytnxTensor<ElemT>& a) {
    eye<CytnxTensor<ElemT>>(ctx, dim, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> eye(context_handle_t<CytnxTensor<ElemT>>& ctx,
                         bond_dim_t<CytnxTensor<ElemT>> N) {
    return eye<CytnxTensor<ElemT>>(ctx, N);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void fill(context_handle_t<CytnxTensor<ElemT>>& ctx, const shape_t<CytnxTensor<ElemT>>& shape,
            elem_t<CytnxTensor<ElemT>> value, CytnxTensor<ElemT>& a) {
    fill<CytnxTensor<ElemT>>(ctx, shape, value, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> fill(context_handle_t<CytnxTensor<ElemT>>& ctx,
                          const shape_t<CytnxTensor<ElemT>>& shape,
                          elem_t<CytnxTensor<ElemT>> value) {
    return fill<CytnxTensor<ElemT>>(ctx, shape, value);
  }

  template <typename ElemT, typename RandNumGen,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> random(context_handle_t<CytnxTensor<ElemT>>& ctx,
                            const shape_t<CytnxTensor<ElemT>>& shape, RandNumGen& gen) {
    return random<CytnxTensor<ElemT>>(ctx, shape, gen);
  }

  template <typename ElemT, typename RandNumGen,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void random(context_handle_t<CytnxTensor<ElemT>>& ctx, const shape_t<CytnxTensor<ElemT>>& shape,
              RandNumGen& gen, CytnxTensor<ElemT>& a) {
    random<CytnxTensor<ElemT>>(ctx, shape, gen, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> copy(context_handle_t<CytnxTensor<ElemT>>& ctx,
                          const CytnxTensor<ElemT>& orig) {
    return copy<CytnxTensor<ElemT>>(ctx, orig);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void copy(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& orig,
            CytnxTensor<ElemT>& dist) {
    copy<CytnxTensor<ElemT>>(ctx, orig, dist);
  }

  template <typename TenT>
  [[deprecated("Use return-value version instead: auto result = copy(ctx, orig)")]]
  void copy(context_handle_t<TenT>& ctx, const TenT& orig, TenT& dist) {
    dist = copy(ctx, orig);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  CytnxTensor<ElemT> move(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& from) {
    return move<CytnxTensor<ElemT>>(ctx, from);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void move(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& from,
            CytnxTensor<ElemT>& to) {
    move<CytnxTensor<ElemT>>(ctx, from, to);
  }

  // move - move tensor contents (in-place)
  template <typename TenT>
  [[deprecated("Use return-value version instead: auto result = move(ctx, from)")]]
  void move(context_handle_t<TenT>& ctx, TenT& from, TenT& to) {
    to = move(ctx, from);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void clear(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& a) {
    clear<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename ElemT, typename RandomIt, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void assign_from_container(context_handle_t<CytnxTensor<ElemT>>& ctx,
                             const shape_t<CytnxTensor<ElemT>>& shape, RandomIt init_elems_begin,
                             Func&& coors2idx, CytnxTensor<ElemT>& a) {
    assign_from_container<CytnxTensor<ElemT>>(ctx, shape, init_elems_begin,
                                              std::forward<Func>(coors2idx), a);
  }

  // ============================================================
  // Context Management
  // ============================================================

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void create_context(context_handle_t<CytnxTensor<ElemT>>& ctx) {
    create_context<CytnxTensor<ElemT>>(ctx);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void destroy_context(context_handle_t<CytnxTensor<ElemT>>& ctx) {
    destroy_context<CytnxTensor<ElemT>>(ctx);
  }

  // ============================================================
  // Read-only Getters
  // ============================================================

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                const elem_coors_t<CytnxTensor<ElemT>>& coors, elem_t<CytnxTensor<ElemT>>& elem) {
    get_elem<CytnxTensor<ElemT>>(ctx, a, coors, elem);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  elem_t<CytnxTensor<ElemT>> get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& a,
                                      const elem_coors_t<CytnxTensor<ElemT>>& coors) {
    return get_elem<CytnxTensor<ElemT>>(ctx, a, coors);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  shape_t<CytnxTensor<ElemT>> shape(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                    const CytnxTensor<ElemT>& a) {
    return shape<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  order_t<CytnxTensor<ElemT>> order(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                    const CytnxTensor<ElemT>& a) {
    return order<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  order_t<CytnxTensor<ElemT>> rank(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                   const CytnxTensor<ElemT>& a) {
    return rank<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename TenT>
  [[deprecated("Use tci::order instead. This API will be removed in the next major version")]]
  order_t<TenT> rank(context_handle_t<TenT>& ctx, const TenT& a) {
    return order(ctx, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  ten_size_t<CytnxTensor<ElemT>> size(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& a) {
    return size<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  std::size_t size_bytes(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a) {
    return size_bytes<CytnxTensor<ElemT>>(ctx, a);
  }

  // ============================================================
  // Tensor Manipulation
  // ============================================================

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void set_elem(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& a,
                const elem_coors_t<CytnxTensor<ElemT>>& coors, elem_t<CytnxTensor<ElemT>> elem) {
    set_elem<CytnxTensor<ElemT>>(ctx, a, coors, elem);
  }

  template <typename ElemT, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout, Func&& f) {
    for_each<CytnxTensor<ElemT>>(ctx, inout, std::forward<Func>(f));
  }

  template <typename ElemT, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in, Func&& f) {
    for_each<CytnxTensor<ElemT>>(ctx, in, std::forward<Func>(f));
  }

  template <typename ElemT, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void for_each_with_coors(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
                           Func&& f) {
    for_each_with_coors<CytnxTensor<ElemT>>(ctx, inout, std::forward<Func>(f));
  }

  template <typename ElemT, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void for_each_with_coors(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
                           Func&& f) {
    for_each_with_coors<CytnxTensor<ElemT>>(ctx, in, std::forward<Func>(f));
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void reshape(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
               const shape_t<CytnxTensor<ElemT>>& new_shape) {
    reshape<CytnxTensor<ElemT>>(ctx, inout, new_shape);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void reshape(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
               const shape_t<CytnxTensor<ElemT>>& new_shape, CytnxTensor<ElemT>& out) {
    reshape<CytnxTensor<ElemT>>(ctx, in, new_shape, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void transpose(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
                 const std::vector<bond_idx_t<CytnxTensor<ElemT>>>& new_order) {
    transpose<CytnxTensor<ElemT>>(ctx, inout, new_order);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void transpose(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
                 const std::vector<bond_idx_t<CytnxTensor<ElemT>>>& new_order,
                 CytnxTensor<ElemT>& out) {
    transpose<CytnxTensor<ElemT>>(ctx, in, new_order, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void cplx_conj(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout) {
    cplx_conj<CytnxTensor<ElemT>>(ctx, inout);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void cplx_conj(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
                 CytnxTensor<ElemT>& out) {
    cplx_conj<CytnxTensor<ElemT>>(ctx, in, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void real(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
            real_ten_t<CytnxTensor<ElemT>>& out) {
    real<CytnxTensor<ElemT>>(ctx, in, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  real_ten_t<CytnxTensor<ElemT>> real(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& in) {
    return real<CytnxTensor<ElemT>>(ctx, in);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void imag(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
            real_ten_t<CytnxTensor<ElemT>>& out) {
    imag<CytnxTensor<ElemT>>(ctx, in, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  real_ten_t<CytnxTensor<ElemT>> imag(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& in) {
    return imag<CytnxTensor<ElemT>>(ctx, in);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void to_cplx(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
               cplx_ten_t<CytnxTensor<ElemT>>& out) {
    to_cplx<CytnxTensor<ElemT>>(ctx, in, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  cplx_ten_t<CytnxTensor<ElemT>> to_cplx(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                         const CytnxTensor<ElemT>& in) {
    return to_cplx<CytnxTensor<ElemT>>(ctx, in);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  real_t<CytnxTensor<ElemT>> norm(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                  const CytnxTensor<ElemT>& a) {
    return norm<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  real_t<CytnxTensor<ElemT>> normalize(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       CytnxTensor<ElemT>& inout) {
    return normalize<CytnxTensor<ElemT>>(ctx, inout);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  real_t<CytnxTensor<ElemT>> normalize(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       const CytnxTensor<ElemT>& in, CytnxTensor<ElemT>& out) {
    return normalize<CytnxTensor<ElemT>>(ctx, in, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void scale(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
             const elem_t<CytnxTensor<ElemT>> s) {
    scale<CytnxTensor<ElemT>>(ctx, inout, s);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void scale(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
             const elem_t<CytnxTensor<ElemT>> s, CytnxTensor<ElemT>& out) {
    scale<CytnxTensor<ElemT>>(ctx, in, s, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void diag(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout) {
    diag<CytnxTensor<ElemT>>(ctx, inout);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void diag(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
            CytnxTensor<ElemT>& out) {
    diag<CytnxTensor<ElemT>>(ctx, in, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trace(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
             const bond_idx_pairs_t<CytnxTensor<ElemT>>& bdidx_pairs) {
    trace<CytnxTensor<ElemT>>(ctx, inout, bdidx_pairs);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trace(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
             const bond_idx_pairs_t<CytnxTensor<ElemT>>& bdidx_pairs, CytnxTensor<ElemT>& out) {
    trace<CytnxTensor<ElemT>>(ctx, in, bdidx_pairs, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void expand(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
              const Map<bond_idx_t<CytnxTensor<ElemT>>, bond_dim_t<CytnxTensor<ElemT>>>&
                  bond_idx_increment_map) {
    expand<CytnxTensor<ElemT>>(ctx, inout, bond_idx_increment_map);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void expand(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
              const Map<bond_idx_t<CytnxTensor<ElemT>>, bond_dim_t<CytnxTensor<ElemT>>>&
                  bond_idx_increment_map,
              CytnxTensor<ElemT>& out) {
    expand<CytnxTensor<ElemT>>(ctx, in, bond_idx_increment_map, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
              const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map) {
    shrink<CytnxTensor<ElemT>>(ctx, inout, bd_idx_el_coor_pair_map);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
              const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map,
              CytnxTensor<ElemT>& out) {
    shrink<CytnxTensor<ElemT>>(ctx, in, bd_idx_el_coor_pair_map, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void extract_sub(
      context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
      const List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>>&
          coor_pairs) {
    extract_sub<CytnxTensor<ElemT>>(ctx, inout, coor_pairs);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void extract_sub(
      context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
      const List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>>&
          coor_pairs,
      CytnxTensor<ElemT>& out) {
    extract_sub<CytnxTensor<ElemT>>(ctx, in, coor_pairs, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
                   const CytnxTensor<ElemT>& sub,
                   const elem_coors_t<CytnxTensor<ElemT>>& begin_pt) {
    replace_sub<CytnxTensor<ElemT>>(ctx, inout, sub, begin_pt);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
                   const CytnxTensor<ElemT>& sub, const elem_coors_t<CytnxTensor<ElemT>>& begin_pt,
                   CytnxTensor<ElemT>& out) {
    replace_sub<CytnxTensor<ElemT>>(ctx, in, sub, begin_pt, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void concatenate(context_handle_t<CytnxTensor<ElemT>>& ctx, const List<CytnxTensor<ElemT>>& ins,
                   const bond_idx_t<CytnxTensor<ElemT>> axis, CytnxTensor<ElemT>& out) {
    concatenate<CytnxTensor<ElemT>>(ctx, ins, axis, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void stack(context_handle_t<CytnxTensor<ElemT>>& ctx, const List<CytnxTensor<ElemT>>& ins,
             const bond_idx_t<CytnxTensor<ElemT>> axis, CytnxTensor<ElemT>& out) {
    stack<CytnxTensor<ElemT>>(ctx, ins, axis, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void inverse(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row) {
    inverse<CytnxTensor<ElemT>>(ctx, inout, num_of_bds_as_row);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void inverse(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& out) {
    inverse<CytnxTensor<ElemT>>(ctx, in, num_of_bds_as_row, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void exp(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout,
           const order_t<CytnxTensor<ElemT>> num_of_bds_as_row) {
    exp<CytnxTensor<ElemT>>(ctx, inout, num_of_bds_as_row);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void exp(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in,
           const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& out) {
    exp<CytnxTensor<ElemT>>(ctx, in, num_of_bds_as_row, out);
  }

  // ============================================================
  // Tensor Linear Algebra
  // ============================================================

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void contract(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_a,
                const CytnxTensor<ElemT>& b,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_b,
                CytnxTensor<ElemT>& c,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_c) {
    contract<CytnxTensor<ElemT>>(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void contract(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                const std::string_view bd_labs_str_a, const CytnxTensor<ElemT>& b,
                const std::string_view bd_labs_str_b, CytnxTensor<ElemT>& c,
                const std::string_view bd_labs_str_c) {
    contract<CytnxTensor<ElemT>>(ctx, a, bd_labs_str_a, b, bd_labs_str_b, c, bd_labs_str_c);
  }

  // Deprecated: in-place linear_combine
  template <typename TenT> [[deprecated(
      "Use return-value version instead: auto result = linear_combine(ctx, ins). "
      "This API will be removed in the next major version")]]
  void linear_combine(context_handle_t<TenT>& ctx, const std::vector<TenT>& ins, TenT& out) {
    out = linear_combine(ctx, ins);
  }

  template <typename TenT> [[deprecated(
      "Use return-value version instead: auto result = linear_combine(ctx, ins, coefs). "
      "This API will be removed in the next major version")]]
  void linear_combine(context_handle_t<TenT>& ctx, const std::vector<TenT>& ins,
                      const std::vector<elem_t<TenT>>& coefs, TenT& out) {
    out = linear_combine(ctx, ins, coefs);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void linear_combine(context_handle_t<CytnxTensor<ElemT>>& ctx,
                      const std::vector<CytnxTensor<ElemT>>& ins, CytnxTensor<ElemT>& out) {
    linear_combine<CytnxTensor<ElemT>>(ctx, ins, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void linear_combine(context_handle_t<CytnxTensor<ElemT>>& ctx,
                      const std::vector<CytnxTensor<ElemT>>& ins,
                      const std::vector<elem_t<CytnxTensor<ElemT>>>& coefs,
                      CytnxTensor<ElemT>& out) {
    linear_combine<CytnxTensor<ElemT>>(ctx, ins, coefs, out);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void svd(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
           const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& u,
           real_ten_t<CytnxTensor<ElemT>>& s_diag, CytnxTensor<ElemT>& v_dag) {
    svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void qr(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
          const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& q,
          CytnxTensor<ElemT>& r) {
    qr<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, q, r);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void lq(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
          const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& l,
          CytnxTensor<ElemT>& q) {
    lq<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, l, q);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                 const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& u,
                 real_ten_t<CytnxTensor<ElemT>>& s_diag, CytnxTensor<ElemT>& v_dag,
                 real_t<CytnxTensor<ElemT>>& trunc_err,
                 const bond_dim_t<CytnxTensor<ElemT>> chi_max,
                 const real_t<CytnxTensor<ElemT>> s_min) {
    trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_max,
                                  s_min);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                 const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& u,
                 real_ten_t<CytnxTensor<ElemT>>& s_diag, CytnxTensor<ElemT>& v_dag,
                 real_t<CytnxTensor<ElemT>>& trunc_err,
                 const real_t<CytnxTensor<ElemT>> target_trunc_err,
                 const real_t<CytnxTensor<ElemT>> s_min) {
    trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err,
                                  target_trunc_err, s_min);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                 const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& u,
                 real_ten_t<CytnxTensor<ElemT>>& s_diag, CytnxTensor<ElemT>& v_dag,
                 real_t<CytnxTensor<ElemT>>& trunc_err,
                 const bond_dim_t<CytnxTensor<ElemT>> chi_min,
                 const bond_dim_t<CytnxTensor<ElemT>> chi_max,
                 const real_t<CytnxTensor<ElemT>> target_trunc_err,
                 const real_t<CytnxTensor<ElemT>> s_min) {
    trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_min,
                                  chi_max, target_trunc_err, s_min);
  }

  // Deprecated old parameter-order overloads (wrapping the already-deprecated TenT versions)
  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                 const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& u,
                 real_ten_t<CytnxTensor<ElemT>>& s_diag, CytnxTensor<ElemT>& v_dag,
                 real_t<CytnxTensor<ElemT>>& trunc_err, const real_t<CytnxTensor<ElemT>> s_min) {
    trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, s_min);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                 const order_t<CytnxTensor<ElemT>> num_of_bds_as_row, CytnxTensor<ElemT>& u,
                 real_ten_t<CytnxTensor<ElemT>>& s_diag, CytnxTensor<ElemT>& v_dag,
                 real_t<CytnxTensor<ElemT>>& trunc_err,
                 const bond_dim_t<CytnxTensor<ElemT>> chi_max,
                 const real_t<CytnxTensor<ElemT>> target_trunc_err,
                 const real_t<CytnxTensor<ElemT>> s_min) {
    trunc_svd<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_max,
                                  target_trunc_err, s_min);
  }

  // Deprecated: Old trunc_svd overload (1) - only s_min
  template <typename TenT> [[deprecated(
      "Parameter order changed. Use trunc_svd(..., trunc_err, chi_max, s_min) or trunc_svd(..., "
      "trunc_err, chi_min, chi_max, target_trunc_err, s_min). This API will be removed in the next "
      "major version")]]
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const order_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const real_t<TenT> s_min) {
    // Forward to spec_v1 overload (1) with chi_min=1, target_trunc_err=0
    constexpr bond_dim_t<TenT> chi_min = 1;
    constexpr real_t<TenT> target_trunc_err = 0.0;
    trunc_svd(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_min,
              static_cast<bond_dim_t<TenT>>(std::numeric_limits<std::uint64_t>::max()),
              target_trunc_err, s_min);
  }

  // Deprecated: Old trunc_svd overload (2) - chi_max, target_trunc_err, s_min
  template <typename TenT> [[deprecated(
      "Parameter order changed. Use trunc_svd(..., trunc_err, chi_max, s_min) or trunc_svd(..., "
      "trunc_err, chi_min, chi_max, target_trunc_err, s_min). This API will be removed in the next "
      "major version")]]
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const order_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const bond_dim_t<TenT> chi_max, const real_t<TenT> target_trunc_err,
                 const real_t<TenT> s_min) {
    // Forward to spec_v1 overload (2) with chi_min=1
    constexpr bond_dim_t<TenT> chi_min = 1;
    trunc_svd(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_min, chi_max,
              target_trunc_err, s_min);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void eigvals(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
               const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               cplx_ten_t<CytnxTensor<ElemT>>& w_diag) {
    eigvals<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void eigvalsh(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
                real_ten_t<CytnxTensor<ElemT>>& w_diag) {
    eigvalsh<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void eig(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
           const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
           cplx_ten_t<CytnxTensor<ElemT>>& w_diag, cplx_ten_t<CytnxTensor<ElemT>>& v) {
    eig<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag, v);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void eigh(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
            const order_t<CytnxTensor<ElemT>> num_of_bds_as_row,
            real_ten_t<CytnxTensor<ElemT>>& w_diag, CytnxTensor<ElemT>& v) {
    eigh<CytnxTensor<ElemT>>(ctx, a, num_of_bds_as_row, w_diag, v);
  }

  // ============================================================
  // Miscellaneous
  // ============================================================

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void show(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a) {
    show<CytnxTensor<ElemT>>(ctx, a);
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  bool close(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
             const CytnxTensor<ElemT>& b, const real_t<CytnxTensor<ElemT>> epsilon) {
    return close<CytnxTensor<ElemT>>(ctx, a, b, epsilon);
  }

  // Deprecated: Tensor equality check with epsilon tolerance
  template <typename TenT>
  [[deprecated("Use close instead. This API will be removed in the next major version")]]
  bool eq(context_handle_t<TenT>& ctx, const TenT& a, const TenT& b, const real_t<TenT> epsilon) {
    return close(ctx, a, b, epsilon);
  }

  // assign_from_container - create tensor from container
  template <typename TenT, typename RandomIt, typename Func>
  void assign_from_container(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape,
                             RandomIt init_elems_begin, Func&& coors2idx, TenT& a) {
    // Allocate tensor with the specified shape
    allocate(ctx, shape, a);

    // Generate all coordinate combinations and assign values
    std::function<void(elem_coors_t<TenT>, std::size_t)> assign_recursive;
    assign_recursive = [&](elem_coors_t<TenT> current_coords, std::size_t dim) {
      if (dim == shape.size()) {
        // Base case: all dimensions set, assign the element
        auto index = std::invoke(coors2idx, current_coords);
        auto value = *(init_elems_begin + index);

        // Convert value to elem_t<TenT>
        elem_t<TenT> elem_val;
        if constexpr (std::is_same_v<elem_t<TenT>, decltype(value)>) {
          elem_val = value;
        } else if constexpr (std::is_arithmetic_v<decltype(value)>) {
          elem_val = static_cast<elem_t<TenT>>(value);
        } else if constexpr (std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex128>
                             && std::is_same_v<decltype(value), std::complex<double>>) {
          elem_val = cytnx::cytnx_complex128(value.real(), value.imag());
        } else if constexpr (std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex64>
                             && std::is_same_v<decltype(value), std::complex<float>>) {
          elem_val = cytnx::cytnx_complex64(value.real(), value.imag());
        } else {
          elem_val = static_cast<elem_t<TenT>>(value);
        }

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
  }

  template <typename ElemT, std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  bool eq(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
          const CytnxTensor<ElemT>& b, const real_t<CytnxTensor<ElemT>> epsilon) {
    return eq<CytnxTensor<ElemT>>(ctx, a, b, epsilon);
  }

  template <typename ElemT, typename RandomIt, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void to_range(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                RandomIt first, Func&& coors2idx) {
    to_range<CytnxTensor<ElemT>>(ctx, a, first, std::forward<Func>(coors2idx));
  }

  template <typename ElemT, typename RandomIt, typename Func,
            std::enable_if_t<!detail::is_cytnx_tensor_v<ElemT>, int> = 0>
  [[deprecated(TCI_DEPRECATED_ELEMT_API)]]
  void to_container(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                    RandomIt first, Func&& coors2idx) {
    to_container<CytnxTensor<ElemT>>(ctx, a, first, std::forward<Func>(coors2idx));
  }

#undef TCI_DEPRECATED_ELEMT_API

  // ============================================================
  // Miscellaneous — TenT deprecated
  // ============================================================

  template <typename TenT, typename RandomIt, typename Func>
  [[deprecated("Use to_range instead. This API will be removed in the next major version")]]
  void to_container(context_handle_t<TenT>& ctx, const TenT& a, RandomIt first, Func&& coors2idx) {
    to_range(ctx, a, first, std::forward<Func>(coors2idx));
  }

  // ============================================================
  // I/O — TenT deprecated
  // ============================================================

  template <typename TenT, typename Storage>
  [[deprecated("Use return-value version instead: auto result = tci::load<TenT>(ctx, strg)")]]
  inline void load(context_handle_t<TenT>& ctx, Storage&& strg, TenT& a) {
    a = load<TenT>(ctx, std::forward<Storage>(strg));
  }

  // ============================================================
  // Deprecated type aliases
  // ============================================================

  template <typename TenT> using rank_t
      [[deprecated("Use tci::order_t instead. This API will be removed in the next major version")]]
      = typename detail::rank_type_fallback<TenT>::type;

}  // namespace tci
