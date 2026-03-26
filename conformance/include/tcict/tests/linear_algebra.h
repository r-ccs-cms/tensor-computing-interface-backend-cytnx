#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>

namespace tcict { namespace tests {

// --- norm (basic: identity matrix) ---

template <typename TenT>
void test_norm_identity(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORM
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT identity;
  tci::eye(ctx, 3, identity);

  auto norm_val = tci::norm(ctx, identity);
  // Frobenius norm of 3x3 identity = sqrt(3)
  TCICT_ASSERT_CLOSE(norm_val, std::sqrt(3.0), eps);
}

// --- linear_combine: uniform coefficients ---

template <typename TenT>
void test_linear_combine_uniform(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LINEAR_COMBINE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT tensor_a, tensor_b, tensor_c, result;
  tci::zeros(ctx, {2, 2}, tensor_a);
  tci::zeros(ctx, {2, 2}, tensor_b);
  tci::zeros(ctx, {2, 2}, tensor_c);

  // a = [[1,2],[3,4]], b = [[5,6],[7,8]], c = [[1,1],[1,1]]
  tci::set_elem(ctx, tensor_a, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_a, {0, 1}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor_a, {1, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor_a, {1, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor_b, {0, 0}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor_b, {0, 1}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, tensor_b, {1, 0}, make_elem<TenT>(7.0));
  tci::set_elem(ctx, tensor_b, {1, 1}, make_elem<TenT>(8.0));
  tci::set_elem(ctx, tensor_c, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_c, {0, 1}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_c, {1, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_c, {1, 1}, make_elem<TenT>(1.0));

  tci::List<TenT> tensors = {tensor_a, tensor_b, tensor_c};
  TCICT_ASSERT_NOTHROW(tci::linear_combine(ctx, tensors, result));

  // Expected: [[7,9],[11,13]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 7.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), 9.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), 11.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 13.0, eps);
}

// --- linear_combine: weighted coefficients ---

template <typename TenT>
void test_linear_combine_weighted(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LINEAR_COMBINE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT tensor_a, tensor_b, result;
  tci::zeros(ctx, {2, 2}, tensor_a);
  tci::zeros(ctx, {2, 2}, tensor_b);

  // a = [[2,4],[6,8]], b = [[1,3],[5,7]]
  tci::set_elem(ctx, tensor_a, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, tensor_a, {0, 1}, make_elem<TenT>(4.0));
  tci::set_elem(ctx, tensor_a, {1, 0}, make_elem<TenT>(6.0));
  tci::set_elem(ctx, tensor_a, {1, 1}, make_elem<TenT>(8.0));
  tci::set_elem(ctx, tensor_b, {0, 0}, make_elem<TenT>(1.0));
  tci::set_elem(ctx, tensor_b, {0, 1}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor_b, {1, 0}, make_elem<TenT>(5.0));
  tci::set_elem(ctx, tensor_b, {1, 1}, make_elem<TenT>(7.0));

  tci::List<TenT> tensors = {tensor_a, tensor_b};
  tci::List<tci::elem_t<TenT>> coefficients = {make_elem<TenT>(0.5), make_elem<TenT>(2.0)};
  TCICT_ASSERT_NOTHROW(tci::linear_combine(ctx, tensors, coefficients, result));

  // Expected: 0.5*a + 2*b = [[3,8],[13,18]]
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 1})), 8.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 0})), 13.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {1, 1})), 18.0, eps);
}

// --- linear_combine: single tensor ---

template <typename TenT>
void test_linear_combine_single(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_LINEAR_COMBINE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT single_tensor, result;
  tci::zeros(ctx, {1, 1}, single_tensor);
  tci::set_elem(ctx, single_tensor, {0, 0}, make_elem<TenT>(5.0));

  tci::List<TenT> single_list = {single_tensor};
  TCICT_ASSERT_NOTHROW(tci::linear_combine(ctx, single_list, result));
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 5.0, eps);

  tci::List<tci::elem_t<TenT>> single_coef = {make_elem<TenT>(3.0)};
  TCICT_ASSERT_NOTHROW(tci::linear_combine(ctx, single_list, single_coef, result));
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, result, {0, 0})), 15.0, eps);
}

// --- normalize: in-place ---

template <typename TenT>
void test_normalize_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORMALIZE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT tensor;
  tci::zeros(ctx, {2, 2}, tensor);
  // [[3,4],[0,0]] -> norm = 5
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(3.0));
  tci::set_elem(ctx, tensor, {0, 1}, make_elem<TenT>(4.0));

  auto original_norm = tci::normalize(ctx, tensor);
  TCICT_ASSERT_CLOSE(std::abs(original_norm), 5.0, eps);

  // Verify normalized values
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 0.6, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 0.8, eps);

  // New norm should be 1
  auto new_norm = tci::norm(ctx, tensor);
  TCICT_ASSERT_CLOSE(new_norm, 1.0, eps);
}

// --- normalize: out-of-place ---

template <typename TenT>
void test_normalize_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORMALIZE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  TenT original, normalized;
  tci::zeros(ctx, {3, 1}, original);
  // [[2],[2],[1]] -> norm = 3
  tci::set_elem(ctx, original, {0, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, original, {1, 0}, make_elem<TenT>(2.0));
  tci::set_elem(ctx, original, {2, 0}, make_elem<TenT>(1.0));

  auto original_norm = tci::normalize(ctx, original, normalized);
  TCICT_ASSERT_CLOSE(std::abs(original_norm), 3.0, eps);

  // Original unchanged
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, original, {0, 0})), 2.0, eps);

  // Normalized tensor
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, normalized, {0, 0})), 2.0 / 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, normalized, {2, 0})), 1.0 / 3.0, eps);

  auto new_norm = tci::norm(ctx, normalized);
  TCICT_ASSERT_CLOSE(new_norm, 1.0, eps);
}

// --- normalize: edge cases ---

template <typename TenT>
void test_normalize_edge_cases(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_NORMALIZE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  // Single non-zero element
  TenT single_elem;
  tci::zeros(ctx, {2, 2}, single_elem);
  tci::set_elem(ctx, single_elem, {1, 1}, make_elem<TenT>(7.0));

  auto norm1 = tci::normalize(ctx, single_elem);
  TCICT_ASSERT_CLOSE(std::abs(norm1), 7.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, single_elem, {1, 1})), 1.0, eps);

  // Zero tensor
  TenT zero_tensor;
  tci::zeros(ctx, {2, 2}, zero_tensor);
  auto norm_zero = tci::normalize(ctx, zero_tensor);
  TCICT_ASSERT_CLOSE(std::abs(norm_zero), 0.0, eps);
}

}}  // namespace tcict::tests
