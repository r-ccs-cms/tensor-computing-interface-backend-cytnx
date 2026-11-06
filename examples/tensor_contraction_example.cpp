#include <cytnx.hpp>
#include <iostream>
#include <random>

#include "tcapi/tcapi.h"

int main() {
  std::cout << "TCAPI Tensor Contraction Example\n";
  std::cout << "===============================\n\n";

  // Create context
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tcapi::create_context(ctx);

  try {
    // Create random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto rand_gen = [&dis, &gen]() { return std::complex<double>(dis(gen), dis(gen)); };

    // Create tensors for contraction: A_{ijk} * B_{kjl} -> C_{il}
    std::cout << "Creating tensor A with shape [3, 4, 2]...\n";
    auto A = tcapi::random<tcapi::CytnxTensor<cytnx::cytnx_complex128>>(ctx, {3, 4, 2}, rand_gen);

    std::cout << "Creating tensor B with shape [2, 4, 5]...\n";
    auto B = tcapi::random<tcapi::CytnxTensor<cytnx::cytnx_complex128>>(ctx, {2, 4, 5}, rand_gen);

    // Perform contraction using string notation
    // A_{ijk} * B_{kjl} -> C_{il}
    std::cout << "\nPerforming contraction: A_{ijk} * B_{kjl} -> C_{il}...\n";
    tcapi::CytnxTensor<cytnx::cytnx_complex128> C;
    tcapi::contract(ctx, A, "ijk", B, "kjl", C, "il");

    auto C_shape = tcapi::shape(ctx, C);
    std::cout << "Result tensor C shape: " << C_shape[0] << " x " << C_shape[1] << "\n";

    // Verify the contraction by computing norm
    auto norm_C = tcapi::norm(ctx, C);
    std::cout << "Frobenius norm of result tensor: " << norm_C << "\n";

    // Show a small part of the result
    std::cout << "\nElement C(0,0): " << tcapi::get_elem(ctx, C, {0, 0}) << "\n";
    std::cout << "Element C(1,1): " << tcapi::get_elem(ctx, C, {1, 1}) << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  tcapi::destroy_context(ctx);

  std::cout << "\nContraction example completed successfully!\n";
  return 0;
}