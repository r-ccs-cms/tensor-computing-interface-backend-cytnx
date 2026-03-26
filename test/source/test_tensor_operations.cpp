#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cmath>
#include <cytnx.hpp>

// FIXME: Trace operations cause AddressSanitizer container-overflow in Cytnx internal code
// This appears to be a bug in the Cytnx library itself (UniTensor::UniTensor constructor)
// Skip this entire test case until the Cytnx library issue is resolved
TEST_CASE("TCI Trace Operations" * doctest::skip()) {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Matrix trace calculation") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> matrix;
    tci::zeros(ctx, shape, matrix);

    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    tci::bond_idx_pairs_t<tci::CytnxTensor<cytnx::cytnx_complex128>> pairs = {{0, 1}};
    tci::trace(ctx, matrix, pairs);

    CHECK(tci::order(ctx, matrix) == 0);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, matrix, {})) - 9.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Contract - duplicate label validation") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Duplicate labels within single tensor should throw") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {3, 3, 4};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {5};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::zeros(ctx, shape_b, b);

    CHECK_THROWS_AS(tci::contract(ctx, a, "iij", b, "k", c, "jk"), std::invalid_argument);

    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b2 = {5, 5};
    tci::CytnxTensor<cytnx::cytnx_complex128> b2;
    tci::zeros(ctx, shape_b2, b2);
    CHECK_THROWS_AS(tci::contract(ctx, a, "ijk", b2, "kk", c, "ij"), std::invalid_argument);

    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_a_dup
        = {1, 1, 2};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_b_single = {3};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_c_out = {2, 3};
    CHECK_THROWS_AS(tci::contract(ctx, a, bd_labs_a_dup, b, bd_labs_b_single, c, bd_labs_c_out),
                    std::invalid_argument);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Stack Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Stack three 2x4 tensors along axis 1") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 4};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c, d;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);
    tci::zeros(ctx, shape, c);

    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, b, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, c, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));

    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b, c};
    tci::stack(ctx, tensors, 1, d);

    auto result_shape = tci::shape(ctx, d);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 4);

    auto el_a = tci::get_elem(ctx, a, {0, 0});
    auto el_d_a = tci::get_elem(ctx, d, {0, 0, 0});
    CHECK(std::abs(tci::real(el_a) - tci::real(el_d_a)) < 1e-10);

    auto el_b = tci::get_elem(ctx, b, {0, 0});
    auto el_d_b = tci::get_elem(ctx, d, {0, 1, 0});
    CHECK(std::abs(tci::real(el_b) - tci::real(el_d_b)) < 1e-10);
  }

  SUBCASE("Stack two 3x3 tensors along axis 0") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, result;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    tci::set_elem(ctx, a, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, b, {2, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b};
    tci::stack(ctx, tensors, 0, result);

    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 3);

    auto el_a11 = tci::get_elem(ctx, result, {0, 1, 1});
    auto el_b22 = tci::get_elem(ctx, result, {1, 2, 2});
    CHECK(std::abs(tci::real(el_a11) - 5.0) < 1e-10);
    CHECK(std::abs(tci::real(el_b22) - 6.0) < 1e-10);
  }

  SUBCASE("Stack along last dimension (axis 2)") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, result;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    tci::set_elem(ctx, a, {1, 2}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, b, {1, 2}, cytnx::cytnx_complex128(8.0, 0.0));

    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b};
    tci::stack(ctx, tensors, 2, result);

    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 2);

    auto el_a = tci::get_elem(ctx, result, {1, 2, 0});
    auto el_b = tci::get_elem(ctx, result, {1, 2, 1});
    CHECK(std::abs(tci::real(el_a) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(el_b) - 8.0) < 1e-10);
  }

  SUBCASE("Error handling: empty tensor list") {
    tci::CytnxTensor<cytnx::cytnx_complex128> result;
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> empty_tensors = {};
    CHECK_THROWS_AS(tci::stack(ctx, empty_tensors, 0, result), std::invalid_argument);
  }

  SUBCASE("Error handling: mismatched tensor shapes") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, result;
    tci::zeros(ctx, {2, 3}, a);
    tci::zeros(ctx, {2, 4}, b);

    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b};
    CHECK_THROWS_AS(tci::stack(ctx, tensors, 0, result), std::invalid_argument);
  }

  tci::destroy_context(ctx);
}
