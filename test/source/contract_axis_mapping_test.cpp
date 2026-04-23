#include <doctest/doctest.h>

#include <cytnx.hpp>

#include "tci/tci.h"

using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
using ContextHandle = tci::context_handle_t<Tensor>;

TEST_CASE("Contract Axis Mapping Debug - NCON notation") {
  ContextHandle context;
  tci::create_context(context);

  // Named (lvalue) so it binds to tci::random's RandNumGen& parameter (TCI v1 spec).
  auto ones_gen = []() { return 1.0; };

  SUBCASE("Simple 2D contract test") {
    // Test basic contract behavior
    auto A = tci::random<Tensor>(context, tci::shape_t<Tensor>{3, 4}, ones_gen);
    auto B = tci::random<Tensor>(context, tci::shape_t<Tensor>{4, 5}, ones_gen);

    Tensor C;
    tci::contract(context, A, {0, -1}, B, {-1, 1}, C, {0, 1});

    auto C_shape = tci::shape(context, C);
    CHECK(C_shape.size() == 2);
    CHECK(C_shape[0] == 3);
    CHECK(C_shape[1] == 5);

    std::cout << "2D contract test passed: (3,4) × (4,5) = (3,5)" << std::endl;
  }

  SUBCASE("4D time evolution contract test - exact replica") {
    std::cout << "\n=== 4D Time Evolution Contract Test ===" << std::endl;

    // Create tensors with exact same shapes as in iTEBD test
    auto theta
        = tci::random<Tensor>(context, tci::shape_t<Tensor>{5, 2, 2, 5}, ones_gen);
    auto u = tci::random<Tensor>(context, tci::shape_t<Tensor>{2, 2, 2, 2}, ones_gen);

    auto theta_shape = tci::shape(context, theta);
    auto u_shape = tci::shape(context, u);

    std::cout << "Input shapes:" << std::endl;
    std::cout << "theta: ";
    for (auto dim : theta_shape) std::cout << dim << " ";
    std::cout << std::endl;
    std::cout << "u: ";
    for (auto dim : u_shape) std::cout << dim << " ";
    std::cout << std::endl;

    std::cout << "\nContract: theta{0, -1, -2, 3} × u{-1, -2, 1, 2} = result{0, 1, 2, 3}"
              << std::endl;
    std::cout << "Expected: contract theta[1,2] with u[0,1], result should be (5,2,2,5)"
              << std::endl;

    Tensor result;
    tci::contract(context, theta, {0, -1, -2, 3}, u, {-1, -2, 1, 2}, result, {0, 1, 2, 3});

    auto result_shape = tci::shape(context, result);
    std::cout << "Actual result: ";
    for (auto dim : result_shape) std::cout << dim << " ";
    std::cout << std::endl;

    // Check if result shape is correct
    bool correct_shape = (result_shape.size() == 4 && result_shape[0] == 5 && result_shape[1] == 2
                          && result_shape[2] == 2 && result_shape[3] == 5);

    if (correct_shape) {
      std::cout << "✅ Contract result shape is CORRECT (5,2,2,5)!" << std::endl;
    } else {
      std::cout << "❌ Contract result shape is WRONG!" << std::endl;
      std::cout << "Expected: (5, 2, 2, 5)" << std::endl;
      std::cout << "Got: (";
      for (size_t i = 0; i < result_shape.size(); ++i) {
        std::cout << result_shape[i];
        if (i < result_shape.size() - 1) std::cout << ", ";
      }
      std::cout << ")" << std::endl;
    }

    // Test assertion
    CHECK(correct_shape);
  }

  SUBCASE("Step-by-step axis analysis") {
    std::cout << "\n=== Step-by-step Axis Analysis ===" << std::endl;

    // Simpler test to understand axis mapping
    auto A = tci::random<Tensor>(context, tci::shape_t<Tensor>{2, 3}, ones_gen);
    auto B = tci::random<Tensor>(context, tci::shape_t<Tensor>{3, 4}, ones_gen);

    std::cout << "Test 1: A{0, -1} × B{-1, 1} = C{0, 1}" << std::endl;
    std::cout << "A(2,3) × B(3,4) should give C(2,4)" << std::endl;

    Tensor C1;
    tci::contract(context, A, {0, -1}, B, {-1, 1}, C1, {0, 1});
    auto C1_shape = tci::shape(context, C1);

    std::cout << "Result: ";
    for (auto dim : C1_shape) std::cout << dim << " ";
    std::cout << std::endl;

    CHECK(C1_shape.size() == 2);
    CHECK(C1_shape[0] == 2);
    CHECK(C1_shape[1] == 4);

    std::cout << "\nTest 2: Different output order A{0, -1} × B{-1, 1} = C{1, 0}" << std::endl;
    std::cout << "A(2,3) × B(3,4) should give C(4,2)" << std::endl;

    Tensor C2;
    tci::contract(context, A, {0, -1}, B, {-1, 1}, C2, {1, 0});
    auto C2_shape = tci::shape(context, C2);

    std::cout << "Result: ";
    for (auto dim : C2_shape) std::cout << dim << " ";
    std::cout << std::endl;

    CHECK(C2_shape.size() == 2);
    CHECK(C2_shape[0] == 4);
    CHECK(C2_shape[1] == 2);
  }

  tci::destroy_context(context);
}