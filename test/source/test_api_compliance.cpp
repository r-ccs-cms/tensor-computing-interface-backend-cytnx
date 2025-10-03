#include <doctest/doctest.h>
#include <tci/tci.h>
#include <cmath>
#include <cytnx.hpp>

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

TEST_CASE("tci::for_each API compliance test - CytnxTensor") {
  tci::context_handle_t<Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("for_each modifying version - element doubling") {
    // Create a test tensor with known values
    tci::shape_t<Tensor> shape = {2, 3};
    Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set initial values: [1, 2, 3, 4, 5, 6]
    tci::set_elem(ctx, tensor, {0, 0}, Elem(1.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 1}, Elem(2.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 2}, Elem(3.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 0}, Elem(4.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 1}, Elem(5.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 2}, Elem(6.0, 0.0));

    // Double each element
    tci::for_each(ctx, tensor, [](Elem& elem) {
        elem = elem * 2.0;
    });

    // Verify all elements were doubled: [2, 4, 6, 8, 10, 12]
    CHECK(std::abs(tci::get_elem(ctx, tensor, {0, 0}) - Elem(2.0, 0.0)) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {0, 1}) - Elem(4.0, 0.0)) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {0, 2}) - Elem(6.0, 0.0)) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {1, 0}) - Elem(8.0, 0.0)) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {1, 1}) - Elem(10.0, 0.0)) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {1, 2}) - Elem(12.0, 0.0)) < 1e-10);
  }

  SUBCASE("for_each iteration and summation") {
    // Test that for_each properly iterates through all elements
    tci::shape_t<Tensor> shape = {2, 2};
    Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set unique values
    tci::set_elem(ctx, tensor, {0, 0}, Elem(1.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 1}, Elem(2.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 0}, Elem(3.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 1}, Elem(4.0, 0.0));

    // Count and sum elements
    int count = 0;
    Elem sum(0.0, 0.0);
    tci::for_each(ctx, tensor, [&count, &sum](Elem& elem) {
      count++;
      sum = sum + elem;
    });

    // Verify all elements were processed
    CHECK(count == 4);
    CHECK(std::abs(sum - Elem(10.0, 0.0)) < 1e-10);  // 1 + 2 + 3 + 4 = 10
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::linear_combine API compliance test") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("linear_combine uniform coefficients - basic tensor addition") {
    // Create test tensors with known values
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};

    tci::CytnxTensor<cytnx::cytnx_complex128> tensor_a, tensor_b, tensor_c, result;
    tci::zeros(ctx, shape, tensor_a);
    tci::zeros(ctx, shape, tensor_b);
    tci::zeros(ctx, shape, tensor_c);

    // Set tensor_a = [[1, 2], [3, 4]]
    tci::set_elem(ctx, tensor_a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_a, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));

    // Set tensor_b = [[5, 6], [7, 8]]
    tci::set_elem(ctx, tensor_b, {0, 0}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor_b, {0, 1}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 1}, cytnx::cytnx_complex128(8.0, 0.0));

    // Set tensor_c = [[1, 1], [1, 1]]
    tci::set_elem(ctx, tensor_c, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_c, {0, 1}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_c, {1, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_c, {1, 1}, cytnx::cytnx_complex128(1.0, 0.0));

    // Test uniform linear combination (simple addition)
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {tensor_a, tensor_b, tensor_c};
    CHECK_NOTHROW(tci::linear_combine(ctx, tensors, result));

    // Expected result: [[7, 9], [11, 13]] = [[1+5+1, 2+6+1], [3+7+1, 4+8+1]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 1})) - 9.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 0})) - 11.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 1})) - 13.0) < 1e-10);
  }

  SUBCASE("linear_combine with specified coefficients - weighted combination") {
    // Create test tensors
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};

    tci::CytnxTensor<cytnx::cytnx_complex128> tensor_a, tensor_b, result;
    tci::zeros(ctx, shape, tensor_a);
    tci::zeros(ctx, shape, tensor_b);

    // Set tensor_a = [[2, 4], [6, 8]]
    tci::set_elem(ctx, tensor_a, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor_a, {0, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 0}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 1}, cytnx::cytnx_complex128(8.0, 0.0));

    // Set tensor_b = [[1, 3], [5, 7]]
    tci::set_elem(ctx, tensor_b, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_b, {0, 1}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 0}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 1}, cytnx::cytnx_complex128(7.0, 0.0));

    // Test weighted linear combination: 0.5 * tensor_a + 2.0 * tensor_b
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {tensor_a, tensor_b};
    tci::List<tci::elem_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> coefficients = {
        cytnx::cytnx_complex128(0.5, 0.0),
        cytnx::cytnx_complex128(2.0, 0.0)
    };

    CHECK_NOTHROW(tci::linear_combine(ctx, tensors, coefficients, result));

    // Expected result: [[3, 8], [13, 18]] = [[0.5*2+2*1, 0.5*4+2*3], [0.5*6+2*5, 0.5*8+2*7]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 3.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 1})) - 8.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 0})) - 13.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 1})) - 18.0) < 1e-10);
  }

  SUBCASE("linear_combine edge cases") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {1, 1};
    tci::CytnxTensor<cytnx::cytnx_complex128> single_tensor, result;
    tci::zeros(ctx, shape, single_tensor);
    tci::set_elem(ctx, single_tensor, {0, 0}, cytnx::cytnx_complex128(5.0, 0.0));

    // Test single tensor uniform combination
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> single_list = {single_tensor};
    CHECK_NOTHROW(tci::linear_combine(ctx, single_list, result));
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 5.0) < 1e-10);

    // Test single tensor with coefficient
    tci::List<tci::elem_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> single_coef = {cytnx::cytnx_complex128(3.0, 0.0)};
    CHECK_NOTHROW(tci::linear_combine(ctx, single_list, single_coef, result));
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 15.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::normalize API compliance test") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("normalize in-place version - basic normalization") {
    // Create test tensor with known values
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tci::zeros(ctx, shape, tensor);

    // Set tensor = [[3, 4], [0, 0]] with norm = 5
    tci::set_elem(ctx, tensor, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 0}, cytnx::cytnx_complex128(0.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 1}, cytnx::cytnx_complex128(0.0, 0.0));

    // Normalize and check returned original norm
    auto original_norm = tci::normalize(ctx, tensor);

    // Verify original norm was 5 (3² + 4² = 9 + 16 = 25, √25 = 5)
    CHECK(std::abs(tci::real(original_norm) - 5.0) < 1e-10);
    CHECK(std::abs(tci::imag(original_norm)) < 1e-10);

    // Verify normalized tensor: [[3/5, 4/5], [0, 0]] = [[0.6, 0.8], [0, 0]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 0})) - 0.6) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 1})) - 0.8) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 0})) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 1})) - 0.0) < 1e-10);

    // Verify new norm is 1
    auto new_norm = tci::norm(ctx, tensor);
    CHECK(std::abs(new_norm - 1.0) < 1e-10);
  }

  SUBCASE("normalize out-of-place version - preserve original") {
    // Create test tensor with known values
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 1};
    tci::CytnxTensor<cytnx::cytnx_complex128> original, normalized;
    tci::zeros(ctx, shape, original);

    // Set original = [[2], [2], [1]] with norm = 3 (2² + 2² + 1² = 9, √9 = 3)
    tci::set_elem(ctx, original, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, original, {1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, original, {2, 0}, cytnx::cytnx_complex128(1.0, 0.0));

    // Normalize out-of-place
    auto original_norm = tci::normalize(ctx, original, normalized);

    // Verify original norm was 3
    CHECK(std::abs(tci::real(original_norm) - 3.0) < 1e-10);

    // Verify original tensor is unchanged
    CHECK(std::abs(tci::real(tci::get_elem(ctx, original, {0, 0})) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, original, {1, 0})) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, original, {2, 0})) - 1.0) < 1e-10);

    // Verify normalized tensor: [[2/3], [2/3], [1/3]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, normalized, {0, 0})) - (2.0/3.0)) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, normalized, {1, 0})) - (2.0/3.0)) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, normalized, {2, 0})) - (1.0/3.0)) < 1e-10);

    // Verify normalized tensor has norm 1
    auto new_norm = tci::norm(ctx, normalized);
    CHECK(std::abs(new_norm - 1.0) < 1e-10);
  }

  SUBCASE("normalize edge cases") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};

    // Test with single non-zero element
    tci::CytnxTensor<cytnx::cytnx_complex128> single_elem;
    tci::zeros(ctx, shape, single_elem);
    tci::set_elem(ctx, single_elem, {1, 1}, cytnx::cytnx_complex128(7.0, 0.0));

    auto norm1 = tci::normalize(ctx, single_elem);
    CHECK(std::abs(tci::real(norm1) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, single_elem, {1, 1})) - 1.0) < 1e-10);

    // Test with zero tensor (should not crash, original implementation handles this)
    tci::CytnxTensor<cytnx::cytnx_complex128> zero_tensor;
    tci::zeros(ctx, shape, zero_tensor);

    auto norm_zero = tci::normalize(ctx, zero_tensor);
    CHECK(std::abs(tci::real(norm_zero) - 0.0) < 1e-10);
    // Zero tensor should remain zero after normalization
    CHECK(std::abs(tci::real(tci::get_elem(ctx, zero_tensor, {0, 0})) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, zero_tensor, {1, 1})) - 0.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}