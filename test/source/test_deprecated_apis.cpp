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

TEST_CASE("Deprecated APIs produce warnings") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
  using Real = tci::real_t<Tensor>;
  using RealTensor = tci::real_ten_t<Tensor>;

  tci::CytnxContextHandle ctx;
  tci::create_context(ctx);

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
    // Create a simple 2x2 matrix
    std::vector<Real> data = {1.0, 0.0, 0.0, 2.0};
    auto coors2idx = [](const std::vector<tci::bond_dim_t<Tensor>>& coors) -> size_t {
      return coors[0] * 2 + coors[1];
    };
    auto matrix = tci::assign_from_range<Tensor>(ctx, {2, 2}, data.begin(), coors2idx);

    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    // Using deprecated API - should show warning (8 parameters)
    tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, 1e-10);

    CHECK(tci::size(ctx, u) > 0);
  }

  SUBCASE("deprecated: trunc_svd old overload (2) - chi_max, target_trunc_err, s_min") {
    // Create a simple 2x2 matrix
    std::vector<Real> data = {1.0, 0.0, 0.0, 2.0};
    auto coors2idx = [](const std::vector<tci::bond_dim_t<Tensor>>& coors) -> size_t {
      return coors[0] * 2 + coors[1];
    };
    auto matrix = tci::assign_from_range<Tensor>(ctx, {2, 2}, data.begin(), coors2idx);

    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    // Using deprecated API - should show warning (10 parameters with chi_max, target_trunc_err,
    // s_min)
    tci::trunc_svd(ctx, matrix, 1, u, s_diag, v_dag, trunc_err, 2ULL, 1e-2, 1e-10);

    CHECK(tci::size(ctx, u) > 0);
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
