#include <doctest/doctest.h>
#include <tci/tci.h>

#include <complex>
#include <vector>

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

TEST_CASE("TCI for_each Function Test - CytnxTensor") {
  SUBCASE("for_each with lambda - element modification") {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Create a simple tensor
    Tensor test_tensor;
    tci::fill(ctx, {3}, Elem(1.0, 0.0), test_tensor);

    // Apply operation to each element
    tci::for_each(ctx, test_tensor, [](Elem& elem) { elem = elem * 2.0; });

    // Verify result
    auto result_elem = tci::get_elem(ctx, test_tensor, {0});
    auto expected = Elem(2.0, 0.0);
    CHECK(std::abs(result_elem - expected) < 1e-10);

    tci::destroy_context(ctx);
  }

  SUBCASE("for_each with capture - scalar multiplication") {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    Tensor test_tensor;
    tci::fill(ctx, {2, 2}, Elem(3.0, 1.0), test_tensor);

    // Lambda with capture
    double multiplier = 0.5;
    tci::for_each(ctx, test_tensor, [multiplier](Elem& elem) { elem = elem * multiplier; });

    auto result = tci::get_elem(ctx, test_tensor, {0, 0});
    auto expected = Elem(1.5, 0.5);
    CHECK(std::abs(result - expected) < 1e-10);

    tci::destroy_context(ctx);
  }

  SUBCASE("for_each const version - summation") {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    Tensor test_tensor;
    tci::fill(ctx, {3}, Elem(2.0, 3.0), test_tensor);

    // const version - read only
    Elem sum = Elem(0.0, 0.0);
    tci::for_each(ctx, static_cast<const Tensor&>(test_tensor),
                  [&sum](const Elem& elem) { sum = sum + elem; });

    Elem expected(6.0, 9.0);  // 3 * (2.0 + 3.0i)
    CHECK(std::abs(sum - expected) < 1e-10);

    tci::destroy_context(ctx);
  }

  SUBCASE("Complex arithmetic - std::sqrt, std::abs, etc.") {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    Tensor test_tensor;
    tci::fill(ctx, {2, 3}, Elem(4.0, 3.0), test_tensor);

    // Complex arithmetic with std library functions
    tci::for_each(ctx, test_tensor, [](Elem& elem) {
      // Natural usage of std:: functions!
      elem = std::sqrt(elem) * Elem(2.0, 0.0) + Elem(1.0, 0.5);
    });

    // Expected: sqrt(4+3i) = 2.12... + 0.70...i
    // * 2 = 4.24... + 1.41...i
    // + (1+0.5i) = 5.24... + 1.91...i
    auto result = tci::get_elem(ctx, test_tensor, {0, 0});
    CHECK(std::abs(result.real() - 5.24) < 0.01);
    CHECK(std::abs(result.imag() - 1.91) < 0.01);

    tci::destroy_context(ctx);
  }

  SUBCASE("Element-wise inversion") {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    Tensor test_tensor;
    tci::fill(ctx, {2, 2}, Elem(0.5, 0.0), test_tensor);

    // Inversion operation
    tci::for_each(ctx, test_tensor, [](Elem& elem) {
      if (std::abs(elem) > 1e-12) {
        elem = Elem(1.0, 0.0) / elem;
      } else {
        elem = Elem(1.0, 0.0);
      }
    });

    auto result = tci::get_elem(ctx, test_tensor, {0, 0});
    auto expected = Elem(2.0, 0.0);  // 1/0.5 = 2
    CHECK(std::abs(result - expected) < 1e-10);

    tci::destroy_context(ctx);
  }
}