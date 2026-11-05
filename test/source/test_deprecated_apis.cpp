/**
 * @file test_deprecated_apis.cpp
 * @brief Test that deprecated APIs produce compilation warnings
 *
 * This test file intentionally uses deprecated APIs to verify that
 * deprecation warnings are properly emitted during compilation.
 * The warnings should appear in the build output but not cause compilation failure.
 */

#include <doctest/doctest.h>
#include <tci/cytnx_typed_tensor_impl_deprecated.h>
#include <tci/tci.h>

#include <functional>
#include <vector>

#include "diagonal_check.h"

TEST_CASE("Deprecated APIs produce warnings") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
  using Real = tci::real_t<Tensor>;
  using RealTensor = tci::real_ten_t<Tensor>;

  tci::CytnxContextHandle ctx;
  tci::create_context(ctx);

  // diag(1, 2), whose singular values 2 and 1 give epsilon(1) = 1 / 5. Every
  // trunc_svd form below is called so that it keeps both: each reaches the
  // spec's general form with chi_max of at least 2, chi_min = 1 and a target
  // below 1 / 5. A forward passing them as chi_min, chi_max, target_trunc_err,
  // s_min caps chi_max at 1 or discards both values, so the checks below also
  // cover the forwards against that order.
  auto diag_1_2 = [&ctx]() {
    std::vector<Real> data = {1.0, 0.0, 0.0, 2.0};
    auto coors2idx = [](const std::vector<tci::bond_dim_t<Tensor>>& coors) -> size_t {
      return coors[0] * 2 + coors[1];
    };
    return tci::assign_from_range<Tensor>(ctx, {2, 2}, data.begin(), coors2idx);
  };
  auto check_both_kept
      = [&ctx](const Tensor& u, const RealTensor& s_diag, const Tensor& v_dag, Real trunc_err) {
          // CHECK rather than REQUIRE, so a failure leaves the later subcases running.
          const auto kept = require_diagonal(ctx, s_diag);
          CHECK(kept.size() == 2);
          if (kept.size() == 2) {
            CHECK(kept[0] + kept[1] == doctest::Approx(3.0));
            CHECK(kept[0] * kept[1] == doctest::Approx(2.0));
          }
          CHECK(tci::shape(ctx, u) == tci::shape_t<Tensor>{2, 2});
          CHECK(tci::shape(ctx, v_dag) == tci::shape_t<Tensor>{2, 2});
          CHECK(trunc_err == doctest::Approx(0.0));
        };

  SUBCASE("deprecated: assign_from_container") {
    // This should produce a deprecation warning at compile time
    std::vector<Real> data = {1.0, 2.0, 3.0, 4.0};
    auto coors2idx = [](const std::vector<tci::bond_dim_t<Tensor>>& coors) -> size_t {
      return coors[0] * 2 + coors[1];
    };

    // Using deprecated API - should show warning
    auto tensor = tci::assign_from_container<Tensor>(ctx, {2, 2}, data.begin(), coors2idx);

    CHECK(tci::size(ctx, tensor) == 4);
  }

  SUBCASE("deprecated: to_container") {
    // Create a simple tensor
    auto tensor = tci::zeros<Tensor>(ctx, {2, 2});

    std::vector<Real> output(4);
    auto coors2idx = [](const std::vector<tci::bond_dim_t<Tensor>>& coors) -> size_t {
      return coors[0] * 2 + coors[1];
    };

    // Using deprecated API - should show warning
    tci::to_container(ctx, tensor, output.begin(), coors2idx);

    CHECK(output.size() == 4);
  }

  SUBCASE("deprecated: eq") {
    auto a = tci::zeros<Tensor>(ctx, {2, 2});
    auto b = tci::zeros<Tensor>(ctx, {2, 2});

    // Using deprecated API - should show warning
    bool same = tci::eq(ctx, a, b, 1e-6);

    CHECK(same == true);
  }

  SUBCASE("deprecated: trunc_svd old overload (1) - only s_min") {
    auto matrix = diag_1_2();

    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    // Using deprecated API - should show warning (8 parameters)
    tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, 1e-10);

    check_both_kept(u, s_diag, v_dag, trunc_err);
  }

  SUBCASE("deprecated: trunc_svd old overload (2) - chi_max, target_trunc_err, s_min") {
    auto matrix = diag_1_2();

    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    // Using deprecated API - should show warning (10 parameters with chi_max, target_trunc_err,
    // s_min)
    tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, 2ULL, 1e-2, 1e-10);

    check_both_kept(u, s_diag, v_dag, trunc_err);
  }

  SUBCASE("deprecated: trunc_svd(target_trunc_err, s_min) - not a spec_v1 form") {
    auto matrix = diag_1_2();

    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    // Using deprecated API - should show warning (9 parameters, both trailing arguments real,
    // which is what picks this overload over (chi_max, s_min)). TenT is spelled out: left to
    // deduction the ElemT wrapper is the more specialized candidate and wins, so the warning
    // would come from that wrapper's own deprecation rather than this overload's.
    tci::trunc_svd<Tensor>(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, 1e-2, 1e-10);

    check_both_kept(u, s_diag, v_dag, trunc_err);
  }

  SUBCASE("deprecated: ElemT trunc_svd - chi_max, s_min, chi_min, target_trunc_err") {
    auto matrix = diag_1_2();

    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    // Using deprecated API - should show warning. Left to deduction, the call picks the ElemT
    // wrapper over the TenT form, as the subcase above notes.
    tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err,
                   static_cast<tci::bond_dim_t<Tensor>>(2), static_cast<Real>(1e-10),
                   static_cast<tci::bond_dim_t<Tensor>>(1), static_cast<Real>(1e-2));

    check_both_kept(u, s_diag, v_dag, trunc_err);
  }

  // Note: get_elem void overload is not deprecated, it's reserved for future GPU support
  // and produces a runtime warning instead of a compile-time warning
}

TEST_CASE("get_elem void overload produces runtime warning") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
  using Real = tci::real_t<Tensor>;

  tci::CytnxContextHandle ctx;
  tci::create_context(ctx);

  SUBCASE("void overload triggers runtime warning") {
    auto tensor = tci::zeros<Tensor>(ctx, {2, 2});
    Real elem;

    // This should produce a runtime warning (not compile-time deprecation)
    // Warning can be suppressed with: export TCI_SUPPRESS_FUTURE_API_WARNING=1
    tci::get_elem(ctx, tensor, {0, 0}, elem);

    CHECK(elem == 0.0);
  }
}
