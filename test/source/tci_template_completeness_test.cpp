#include <doctest/doctest.h>
#include <tci/tci.h>
#include <vector>
#include <complex>

using Ten = cytnx::Tensor;

TEST_CASE("Template Function Completeness Test") {
  using namespace tci;

  // Setup context
  auto ctx = create_context<context_handle_t<Ten>>();

  SUBCASE("for_each_with_coors with lambda function") {
    // Create a 2x2 tensor
    Ten a = eye<Ten>(ctx, 2);

    // Test for_each_with_coors with lambda (this should cause link error if not implemented)
    bool test_passed = false;
    try {
      tci::for_each_with_coors(ctx, a, [&test_passed](tci::elem_t<Ten>& elem, const tci::elem_coors_t<Ten>& coors) {
        // Simple operation: set diagonal elements to 2.0
        if (coors[0] == coors[1]) {
          elem = tci::elem_t<Ten>(2.0, 0.0);
        }
        test_passed = true;
      });

      // If we reach here, the function worked
      auto elem_00 = tci::get_elem(ctx, a, {0, 0});
      CHECK(std::abs(elem_00.real() - 2.0) < 1e-10);
      CHECK(test_passed);

    } catch (const std::exception& e) {
      // If for_each_with_coors isn't properly implemented, we'll get an exception
      INFO("for_each_with_coors threw exception: " << e.what());
      CHECK(false); // Mark as failure for now
    }
  }

  SUBCASE("for_each_with_coors const version with lambda") {
    // Create a 2x2 tensor
    Ten a = eye<Ten>(ctx, 2);
    const Ten& const_a = a;

    // Test const version
    double sum_diagonal = 0.0;
    try {
      tci::for_each_with_coors(ctx, const_a, [&sum_diagonal](const tci::elem_t<Ten>& elem, const tci::elem_coors_t<Ten>& coors) {
        // Sum diagonal elements
        if (coors[0] == coors[1]) {
          sum_diagonal += elem.real();
        }
      });

      // Should sum to 2.0 (two 1.0s on diagonal)
      CHECK(std::abs(sum_diagonal - 2.0) < 1e-10);

    } catch (const std::exception& e) {
      INFO("const for_each_with_coors threw exception: " << e.what());
      CHECK(false); // Mark as failure for now
    }
  }

  destroy_context(ctx);
}