#include <doctest/doctest.h>
#include <tcapi/tcapi.h>

#include <cmath>
#include <cytnx.hpp>
#include <functional>

// FIXME: Trace operations cause AddressSanitizer container-overflow in Cytnx internal code
// This appears to be a bug in the Cytnx library itself (UniTensor::UniTensor constructor)
// Skip this entire test case until the Cytnx library issue is resolved
TEST_CASE("TCAPI Trace Operations" * doctest::skip()) {
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tcapi::create_context(ctx);

  SUBCASE("Matrix trace calculation") {
    tcapi::shape_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 3};
    tcapi::CytnxTensor<cytnx::cytnx_complex128> matrix;
    tcapi::zeros(ctx, shape, matrix);

    tcapi::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tcapi::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));
    tcapi::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(4.0, 0.0));
    tcapi::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tcapi::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    tcapi::bond_idx_pairs_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> pairs = {{0, 1}};
    tcapi::trace(ctx, matrix, pairs);

    CHECK(tcapi::order(ctx, matrix) == 0);
    CHECK(std::abs(tcapi::real(tcapi::get_elem(ctx, matrix, {})) - 9.0) < 1e-10);
  }

  tcapi::destroy_context(ctx);
}

TEST_CASE("TCAPI Contract - duplicate label validation") {
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tcapi::create_context(ctx);

  SUBCASE("Duplicate labels within single tensor should throw") {
    tcapi::shape_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {3, 3, 4};
    tcapi::shape_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {5};
    tcapi::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tcapi::zeros(ctx, shape_a, a);
    tcapi::zeros(ctx, shape_b, b);

    CHECK_THROWS_AS(tcapi::contract(ctx, a, "iij", b, "k", c, "jk"), std::invalid_argument);

    tcapi::shape_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> shape_b2 = {5, 5};
    tcapi::CytnxTensor<cytnx::cytnx_complex128> b2;
    tcapi::zeros(ctx, shape_b2, b2);
    CHECK_THROWS_AS(tcapi::contract(ctx, a, "ijk", b2, "kk", c, "ij"), std::invalid_argument);

    tcapi::List<tcapi::bond_label_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_a_dup
        = {1, 1, 2};
    tcapi::List<tcapi::bond_label_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_b_single
        = {3};
    tcapi::List<tcapi::bond_label_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_c_out
        = {2, 3};
    CHECK_THROWS_AS(tcapi::contract(ctx, a, bd_labs_a_dup, b, bd_labs_b_single, c, bd_labs_c_out),
                    std::invalid_argument);
  }

  tcapi::destroy_context(ctx);
}

TEST_CASE("TCAPI linear_combine - zeroth-order inputs") {
  using Ten = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = cytnx::cytnx_complex128;
  tcapi::context_handle_t<Ten> ctx;
  tcapi::create_context(ctx);

  auto s = tcapi::fill<Ten>(ctx, {}, Elem(2.0, -1.0));
  auto t = tcapi::fill<Ten>(ctx, {}, Elem(0.5, 3.0));

  SUBCASE("Zeroth-order inputs combine into a zeroth-order result") {
    auto sum = tcapi::linear_combine<Ten>(ctx, {std::cref(s), std::cref(t)});
    CHECK(tcapi::order(ctx, sum) == 0);
    CHECK(tcapi::get_elem(ctx, sum, {}) == Elem(2.5, 2.0));

    auto weighted = tcapi::linear_combine<Ten>(ctx, {std::cref(s), std::cref(t)},
                                               {Elem(0.0, 1.0), Elem(2.0)});
    CHECK(tcapi::order(ctx, weighted) == 0);
    CHECK(tcapi::get_elem(ctx, weighted, {}) == Elem(0.0, 1.0) * Elem(2.0, -1.0) + Elem(1.0, 6.0));
  }

  SUBCASE("A zeroth-order input beside another shape is rejected") {
    auto one = tcapi::fill<Ten>(ctx, {1}, Elem(2.0, -1.0));
    auto three = tcapi::fill<Ten>(ctx, {3}, Elem(1.0));
    Ten empty;
    CHECK_THROWS_AS(tcapi::linear_combine<Ten>(ctx, {std::cref(s), std::cref(one)}),
                    std::invalid_argument);
    CHECK_THROWS_AS(tcapi::linear_combine<Ten>(ctx, {std::cref(three), std::cref(s)}),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        tcapi::linear_combine<Ten>(ctx, {std::cref(s), std::cref(empty)}, {Elem(1.0), Elem(1.0)}),
        std::invalid_argument);
  }

  tcapi::destroy_context(ctx);
}

TEST_CASE("TCAPI Stack Operations") {
  using Ten = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  tcapi::context_handle_t<Ten> ctx;
  tcapi::create_context(ctx);

  SUBCASE("Stack three 2x4 tensors along axis 1") {
    tcapi::shape_t<Ten> shape = {2, 4};
    Ten a, b, c;
    tcapi::zeros(ctx, shape, a);
    tcapi::zeros(ctx, shape, b);
    tcapi::zeros(ctx, shape, c);

    tcapi::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tcapi::set_elem(ctx, b, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tcapi::set_elem(ctx, c, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));

    auto d = tcapi::stack<Ten>(ctx, {std::cref(a), std::cref(b), std::cref(c)}, 1);

    auto result_shape = tcapi::shape(ctx, d);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 4);

    auto el_a = tcapi::get_elem(ctx, a, {0, 0});
    auto el_d_a = tcapi::get_elem(ctx, d, {0, 0, 0});
    CHECK(std::abs(tcapi::real(el_a) - tcapi::real(el_d_a)) < 1e-10);

    auto el_b = tcapi::get_elem(ctx, b, {0, 0});
    auto el_d_b = tcapi::get_elem(ctx, d, {0, 1, 0});
    CHECK(std::abs(tcapi::real(el_b) - tcapi::real(el_d_b)) < 1e-10);
  }

  SUBCASE("Stack two 3x3 tensors along axis 0") {
    tcapi::shape_t<Ten> shape = {3, 3};
    Ten a, b;
    tcapi::zeros(ctx, shape, a);
    tcapi::zeros(ctx, shape, b);

    tcapi::set_elem(ctx, a, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tcapi::set_elem(ctx, b, {2, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    auto result = tcapi::stack<Ten>(ctx, {std::cref(a), std::cref(b)}, 0);

    auto result_shape = tcapi::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 3);

    auto el_a11 = tcapi::get_elem(ctx, result, {0, 1, 1});
    auto el_b22 = tcapi::get_elem(ctx, result, {1, 2, 2});
    CHECK(std::abs(tcapi::real(el_a11) - 5.0) < 1e-10);
    CHECK(std::abs(tcapi::real(el_b22) - 6.0) < 1e-10);
  }

  SUBCASE("Stack along last dimension (axis 2)") {
    tcapi::shape_t<Ten> shape = {2, 3};
    Ten a, b;
    tcapi::zeros(ctx, shape, a);
    tcapi::zeros(ctx, shape, b);

    tcapi::set_elem(ctx, a, {1, 2}, cytnx::cytnx_complex128(7.0, 0.0));
    tcapi::set_elem(ctx, b, {1, 2}, cytnx::cytnx_complex128(8.0, 0.0));

    auto result = tcapi::stack<Ten>(ctx, {std::cref(a), std::cref(b)}, 2);

    auto result_shape = tcapi::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 2);

    auto el_a = tcapi::get_elem(ctx, result, {1, 2, 0});
    auto el_b = tcapi::get_elem(ctx, result, {1, 2, 1});
    CHECK(std::abs(tcapi::real(el_a) - 7.0) < 1e-10);
    CHECK(std::abs(tcapi::real(el_b) - 8.0) < 1e-10);
  }

  SUBCASE("Error handling: empty tensor list") {
    tcapi::List<tcapi::CRef<Ten>> empty_tensors = {};
    CHECK_THROWS_AS(tcapi::stack(ctx, empty_tensors, 0), std::invalid_argument);
  }

  SUBCASE("Error handling: mismatched tensor shapes") {
    Ten a, b;
    tcapi::zeros(ctx, {2, 3}, a);
    tcapi::zeros(ctx, {2, 4}, b);

    CHECK_THROWS_AS(tcapi::stack<Ten>(ctx, {std::cref(a), std::cref(b)}, 0), std::invalid_argument);
  }

  tcapi::destroy_context(ctx);
}
