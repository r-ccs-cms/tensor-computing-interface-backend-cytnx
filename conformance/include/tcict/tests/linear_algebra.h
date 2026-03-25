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

}}  // namespace tcict::tests
