#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

namespace tcict { namespace tests {

// --- close (eq) : identical tensors ---

template <typename TenT>
void test_close_identical(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLOSE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor1, tensor2;
  tci::eye(ctx, 2, tensor1);
  tci::eye(ctx, 2, tensor2);

  bool are_equal = tci::eq(ctx, tensor1, tensor2, eps);
  TCICT_ASSERT(are_equal == true);
}

// --- close (eq) : different tensors ---

template <typename TenT>
void test_close_different(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLOSE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  TenT tensor1, tensor2;
  tci::eye(ctx, 2, tensor1);
  tci::zeros(ctx, {2, 2}, tensor2);

  bool are_equal = tci::eq(ctx, tensor1, tensor2, eps);
  TCICT_ASSERT(are_equal == false);
}

}}  // namespace tcict::tests
