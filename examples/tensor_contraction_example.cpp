#include <cytnx.hpp>
#include <iostream>
#include <random>

#include "tci/tci.h"

int main() {
  std::cout << "TCI Tensor Contraction Example\n";
  std::cout << "===============================\n\n";

  // Create context
  auto ctx = tci::create_context<tci::context_handle_t<cytnx::Tensor>>();

  try {
    // Create random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto rand_gen = [&dis, &gen]() { return std::complex<double>(dis(gen), dis(gen)); };

    // Create tensors for contraction: A_{ijk} * B_{kjl} -> C_{il}
    std::cout << "Creating tensor A with shape [3, 4, 2]...\n";
    auto A = tci::random<cytnx::Tensor>(ctx, {3, 4, 2}, rand_gen);

    std::cout << "Creating tensor B with shape [2, 4, 5]...\n";
    auto B = tci::random<cytnx::Tensor>(ctx, {2, 4, 5}, rand_gen);

    // Perform contraction using string notation
    // A_{ijk} * B_{kjl} -> C_{il}
    std::cout << "\nPerforming contraction: A_{ijk} * B_{kjl} -> C_{il}...\n";
    cytnx::Tensor C;
    tci::contract(ctx, A, "ijk", B, "kjl", C, "il");

    auto C_shape = tci::shape(ctx, C);
    std::cout << "Result tensor C shape: " << C_shape[0] << " x " << C_shape[1] << "\n";

    // Verify the contraction by computing norm
    auto norm_C = tci::norm(ctx, C);
    std::cout << "Frobenius norm of result tensor: " << norm_C << "\n";

    // Show a small part of the result
    std::cout << "\nElement C(0,0): ";
    std::visit([](auto&& val) { std::cout << val << "\n"; }, tci::get_elem(ctx, C, {0, 0}));
    std::cout << "Element C(1,1): ";
    std::visit([](auto&& val) { std::cout << val << "\n"; }, tci::get_elem(ctx, C, {1, 1}));

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  tci::destroy_context(ctx);

  std::cout << "\nContraction example completed successfully!\n";
  return 0;
}