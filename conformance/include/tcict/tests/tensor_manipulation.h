#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>

namespace tcict { namespace tests {

// --- shrink (in-place) ---

template <typename TenT>
void test_shrink_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SHRINK
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {3, 3}, tensor);

  // Fill with values 1-9
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      tci::set_elem(ctx, tensor,
                    {static_cast<tci::elem_coor_t<TenT>>(i),
                     static_cast<tci::elem_coor_t<TenT>>(j)},
                    make_elem<TenT>(i * 3 + j + 1));
    }
  }

  // Shrink to top-left 2x2
  tci::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));
  shrink_map[1] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));
  tci::shrink(ctx, tensor, shrink_map);

  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 2);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 4.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 5.0, eps);
}

// --- shrink (out-of-place) ---

template <typename TenT>
void test_shrink_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SHRINK
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT input;
  tci::zeros(ctx, {4, 4}, input);

  // Set values in center 2x2 region [1:3, 1:3]
  tci::set_elem(ctx, input, {1, 1}, make_elem<TenT>(11.0));
  tci::set_elem(ctx, input, {1, 2}, make_elem<TenT>(12.0));
  tci::set_elem(ctx, input, {2, 1}, make_elem<TenT>(21.0));
  tci::set_elem(ctx, input, {2, 2}, make_elem<TenT>(22.0));

  tci::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(1),
                                 static_cast<tci::elem_coor_t<TenT>>(3));
  shrink_map[1] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(1),
                                 static_cast<tci::elem_coor_t<TenT>>(3));

  TenT output;
  tci::shrink(ctx, input, shrink_map, output);

  auto result_shape = tci::shape(ctx, output);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 2);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 0})), 11.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 1})), 12.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 0})), 21.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 1})), 22.0, eps);
}

// --- shrink preserves complex values ---

template <typename TenT>
void test_shrink_complex_values(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_SHRINK
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {3, 3}, tensor);

  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.5, 2.5));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(3.5, 4.5));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.5, 6.5));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(7.5, 8.5));

  tci::bond_idx_elem_coor_pair_map<TenT> shrink_map;
  shrink_map[0] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));
  shrink_map[1] = std::make_pair(static_cast<tci::elem_coor_t<TenT>>(0),
                                 static_cast<tci::elem_coor_t<TenT>>(2));

  TenT output;
  tci::shrink(ctx, tensor, shrink_map, output);

  auto e00 = tci::get_elem(ctx, output, {0, 0});
  auto e01 = tci::get_elem(ctx, output, {0, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(e00), 1.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(e00), 2.5, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(e01), 3.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(e01), 4.5, eps);
}

// --- real extraction (out-of-place) ---

template <typename TenT>
void test_real_extraction(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_REAL
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto real_tensor = tci::real(ctx, tensor);

  using RealTenT = tci::real_ten_t<TenT>;
  auto elem00 = tci::get_elem(ctx, real_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, real_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 3.14, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), -1.59, eps);
}

// --- imag extraction (out-of-place) ---

template <typename TenT>
void test_imag_extraction(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_IMAG
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.59, 0.58));

  auto imag_tensor = tci::imag(ctx, tensor);

  using RealTenT = tci::real_ten_t<TenT>;
  auto elem00 = tci::get_elem(ctx, imag_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, imag_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem00), 2.71, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(elem11), 0.58, eps);
}

// --- real and imag extraction (in-place) ---

template <typename TenT>
void test_real_imag_inplace(tci_test_fixture<TenT>& fix) {
#if defined(TCICT_SKIP_REAL) || defined(TCICT_SKIP_IMAG)
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(5.25, 7.75));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-2.25, -3.75));

  tci::real_ten_t<TenT> real_output, imag_output;
  tci::real(ctx, tensor, real_output);
  tci::imag(ctx, tensor, imag_output);

  using RealTenT = tci::real_ten_t<TenT>;
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, real_output, {0, 0})), 5.25, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, real_output, {1, 1})), -2.25, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, imag_output, {0, 0})), 7.75, eps);
  TCICT_ASSERT_CLOSE(real_part<RealTenT>(tci::get_elem(ctx, imag_output, {1, 1})), -3.75, eps);
}

// --- cplx_conj (in-place) ---

template <typename TenT>
void test_cplx_conj_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CPLX_CONJ
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 2.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(-3.0, 4.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(5.0, -6.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-7.0, -8.0));

  tci::cplx_conj(ctx, tensor);

  // Real parts unchanged, imaginary parts negated
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), -2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), -3.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), -4.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 5.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 6.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), -7.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 8.0, eps);
}

// --- cplx_conj (out-of-place) ---

template <typename TenT>
void test_cplx_conj_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CPLX_CONJ
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT input;
  tci::zeros(ctx, {2, 2}, input);
  tci::set_elem(ctx, input, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, input, {1, 1}, make_elem<TenT>(-1.41, -1.73));

  TenT output;
  tci::cplx_conj(ctx, input, output);

  // Input unchanged
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, input, {0, 0})), 3.14, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, input, {0, 0})), 2.71, eps);

  // Output conjugated
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {0, 0})), 3.14, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, output, {0, 0})), -2.71, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, output, {1, 1})), -1.41, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(tci::get_elem(ctx, output, {1, 1})), 1.73, eps);
}

// --- to_cplx (out-of-place, from real type) ---
// NOTE: TenT here should be a real tensor type (e.g., CytnxTensor<double>)

template <typename RealTenT>
void test_to_cplx_outofplace(tci_test_fixture<RealTenT>& fix) {
#ifdef TCICT_SKIP_TO_CPLX
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  RealTenT real_tensor;
  tci::zeros(ctx, {2, 2}, real_tensor);

  tci::set_elem(ctx, real_tensor, {0, 0}, static_cast<tci::elem_t<RealTenT>>(1.5));
  tci::set_elem(ctx, real_tensor, {0, 1}, static_cast<tci::elem_t<RealTenT>>(2.5));
  tci::set_elem(ctx, real_tensor, {1, 0}, static_cast<tci::elem_t<RealTenT>>(3.5));
  tci::set_elem(ctx, real_tensor, {1, 1}, static_cast<tci::elem_t<RealTenT>>(4.5));

  auto complex_tensor = tci::to_cplx(ctx, real_tensor);

  using CplxTenT = tci::cplx_ten_t<RealTenT>;
  auto elem00 = tci::get_elem(ctx, complex_tensor, {0, 0});
  auto elem11 = tci::get_elem(ctx, complex_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem00), 1.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem00), 0.0, eps);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem11), 4.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem11), 0.0, eps);
}

// --- to_cplx (in-place, from real type) ---

template <typename RealTenT>
void test_to_cplx_inplace(tci_test_fixture<RealTenT>& fix) {
#ifdef TCICT_SKIP_TO_CPLX
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  RealTenT real_tensor;
  tci::zeros(ctx, {2, 2}, real_tensor);

  tci::set_elem(ctx, real_tensor, {0, 0}, static_cast<tci::elem_t<RealTenT>>(7.25));
  tci::set_elem(ctx, real_tensor, {1, 1}, static_cast<tci::elem_t<RealTenT>>(8.75));

  tci::cplx_ten_t<RealTenT> complex_output;
  tci::to_cplx(ctx, real_tensor, complex_output);

  using CplxTenT = tci::cplx_ten_t<RealTenT>;
  auto elem00 = tci::get_elem(ctx, complex_output, {0, 0});
  auto elem11 = tci::get_elem(ctx, complex_output, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem00), 7.25, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem00), 0.0, eps);
  TCICT_ASSERT_CLOSE(real_part<CplxTenT>(elem11), 8.75, eps);
  TCICT_ASSERT_CLOSE(imag_part<CplxTenT>(elem11), 0.0, eps);
}

// --- to_cplx (complex to complex) ---

template <typename TenT>
void test_to_cplx_complex_to_complex(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_TO_CPLX
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.14, 2.71));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(-1.41, 1.73));

  auto result = tci::to_cplx(ctx, tensor);

  auto elem00 = tci::get_elem(ctx, result, {0, 0});
  auto elem11 = tci::get_elem(ctx, result, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(elem00), 3.14, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(elem00), 2.71, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(elem11), -1.41, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(elem11), 1.73, eps);
}

// --- for_each: element doubling ---

template <typename TenT>
void test_for_each_doubling(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT tensor;
  tci::zeros(ctx, {2, 3}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {0, 2}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor, {1, 2}, make_elem<TenT>(6.0));

  tci::for_each(ctx, tensor, [](Elem& elem) { elem = elem * 2.0; });

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 2})), 6.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 2})), 12.0, eps);
}

// --- for_each: iteration and summation ---

template <typename TenT>
void test_for_each_summation(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(4.0));

  int count = 0;
  Elem sum = make_elem<TenT>(0.0);
  tci::for_each(ctx, tensor, [&count, &sum](Elem& elem) {
    count++;
    sum = sum + elem;
  });

  TCICT_ASSERT(count == 4);
  TCICT_ASSERT_CLOSE(real_part<TenT>(sum), 10.0, eps);
}

// --- for_each: scalar multiplication with capture ---

template <typename TenT>
void test_for_each_capture(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT tensor;
  tci::fill(ctx, {2, 2}, make_elem<TenT>(3.0, 1.0), tensor);

  double multiplier = 0.5;
  tci::for_each(ctx, tensor, [multiplier](Elem& elem) { elem = elem * multiplier; });

  auto result = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result), 1.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(result), 0.5, eps);
}

// --- for_each: const version ---

template <typename TenT>
void test_for_each_const(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT tensor;
  tci::fill(ctx, {3}, make_elem<TenT>(2.0, 3.0), tensor);

  Elem sum = make_elem<TenT>(0.0);
  tci::for_each(ctx, static_cast<const TenT&>(tensor),
                [&sum](const Elem& elem) { sum = sum + elem; });

  TCICT_ASSERT_CLOSE(real_part<TenT>(sum), 6.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(sum), 9.0, eps);
}

// --- for_each: element-wise inversion ---

template <typename TenT>
void test_for_each_inversion(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT tensor;
  tci::fill(ctx, {2, 2}, make_elem<TenT>(0.5), tensor);

  tci::for_each(ctx, tensor, [](Elem& elem) {
    if (std::abs(elem) > 1e-12) {
      elem = Elem(1.0, 0.0) / elem;
    }
  });

  auto result = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result), 2.0, eps);
}

// --- for_each_with_coors: mutable ---

template <typename TenT>
void test_for_each_with_coors(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH_WITH_COORS
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT a = tci::template eye<TenT>(ctx, 2);

  tci::for_each_with_coors(
      ctx, a, [](Elem& elem, const tci::elem_coors_t<TenT>& coors) {
        if (coors[0] == coors[1]) {
          elem = static_cast<Elem>(2.0);
        }
      });

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, a, {0, 0})), 2.0, eps);
}

// --- for_each_with_coors: const version ---

template <typename TenT>
void test_for_each_with_coors_const(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_FOR_EACH_WITH_COORS
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  using Elem = tci::elem_t<TenT>;

  TenT a = tci::template eye<TenT>(ctx, 2);
  const TenT& const_a = a;

  double sum_diagonal = 0.0;
  tci::for_each_with_coors(
      ctx, const_a,
      [&sum_diagonal](const Elem& elem, const tci::elem_coors_t<TenT>& coors) {
        if (coors[0] == coors[1]) {
          sum_diagonal += real_part<TenT>(elem);
        }
      });

  TCICT_ASSERT_CLOSE(sum_diagonal, 2.0, eps);
}

}}  // namespace tcict::tests
