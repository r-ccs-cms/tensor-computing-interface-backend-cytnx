#include <cytnx.hpp>
#include <iostream>

#include "tci/tci.h"

int main() {
  std::cout << "TCI Basic Tensor Example\n";
  std::cout << "========================\n\n";

  // Create context
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  try {
    // Create a 3x4 tensor filled with zeros
    std::cout << "Creating a 3x4 zero tensor...\n";
    auto a = tci::zeros<cytnx::Tensor>(ctx, {3, 4});

    std::cout << "Tensor shape: ";
    auto shape = tci::shape(ctx, a);
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) std::cout << " x ";
    }
    std::cout << "\n";

    std::cout << "Tensor rank: " << tci::rank(ctx, a) << "\n";
    std::cout << "Tensor size: " << tci::size(ctx, a) << " elements\n";

    // Set some elements
    std::cout << "\nSetting element at (1,2) to 1.5+2.0i...\n";
    tci::set_elem(ctx, a, {1, 2}, std::complex<double>(1.5, 2.0));

    // Get element back
    auto elem = tci::get_elem(ctx, a, {1, 2});
    // Note: elem_t for cytnx::Tensor is std::variant, so we use std::visit to print it
    std::visit([](auto&& val) {
      std::cout << "Element at (1,2): " << val << "\n";
    }, elem);

    // Create identity matrix
    std::cout << "\nCreating 3x3 identity matrix...\n";
    auto eye_tensor = tci::eye<cytnx::Tensor>(ctx, 3);
    std::cout << "Identity matrix created.\n";

    // Print tensors
    std::cout << "\nZero tensor:\n";
    tci::show(ctx, a);

    std::cout << "\nIdentity tensor:\n";
    tci::show(ctx, eye_tensor);

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  tci::destroy_context(ctx);

  std::cout << "\nExample completed successfully!\n";
  return 0;
}