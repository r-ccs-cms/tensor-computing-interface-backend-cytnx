#include <cytnx.hpp>
#include <iostream>
#include <random>

#include "tcapi/tcapi.h"

int main() {
  std::cout << "TCAPI SVD Example\n";
  std::cout << "===============\n\n";

  // Create context
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tcapi::create_context(ctx);

  try {
    // Create random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto rand_gen = [&dis, &gen]() { return std::complex<double>(dis(gen), dis(gen)); };

    // Create a random tensor for SVD: shape [3, 4, 12]
    std::cout << "Creating random tensor with shape [3, 4, 12]...\n";
    auto A = tcapi::random<tcapi::CytnxTensor<cytnx::cytnx_complex128>>(ctx, {3, 4, 12}, rand_gen);

    std::cout << "Original tensor shape: ";
    auto shape = tcapi::shape(ctx, A);
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) std::cout << " x ";
    }
    std::cout << "\n";

    // Perform SVD: treat first 2 bonds as rows, last bond as columns
    std::cout << "\nPerforming SVD (first 2 bonds as rows)...\n";
    tcapi::CytnxTensor<cytnx::cytnx_complex128> U, V_dag;
    // Singular values are real; S_diag is the square diagonal tensor holding them
    tcapi::CytnxTensor<cytnx::cytnx_double> S_diag;

    tcapi::svd(ctx, A, 2, U, S_diag, V_dag);

    // Check shapes
    auto U_shape = tcapi::shape(ctx, U);
    auto S_shape = tcapi::shape(ctx, S_diag);
    auto V_shape = tcapi::shape(ctx, V_dag);

    std::cout << "U shape: ";
    for (size_t i = 0; i < U_shape.size(); ++i) {
      std::cout << U_shape[i];
      if (i < U_shape.size() - 1) std::cout << " x ";
    }
    std::cout << "\n";

    std::cout << "S shape: ";
    for (size_t i = 0; i < S_shape.size(); ++i) {
      std::cout << S_shape[i];
      if (i < S_shape.size() - 1) std::cout << " x ";
    }
    std::cout << "\n";

    std::cout << "V† shape: ";
    for (size_t i = 0; i < V_shape.size(); ++i) {
      std::cout << V_shape[i];
      if (i < V_shape.size() - 1) std::cout << " x ";
    }
    std::cout << "\n";

    // Show some singular values
    std::cout << "\nFirst few singular values:\n";
    for (int i = 0; i < std::min(5, static_cast<int>(S_shape[0])); ++i) {
      const auto c = static_cast<tcapi::elem_coor_t<tcapi::CytnxTensor<cytnx::cytnx_double>>>(i);
      auto sv = tcapi::get_elem(ctx, S_diag, {c, c});
      std::cout << "S[" << i << "] = " << sv << "\n";
    }

    // Calculate norms for verification
    auto norm_A = tcapi::norm(ctx, A);
    auto norm_S = tcapi::norm(ctx, S_diag);

    std::cout << "\nOriginal tensor norm: " << norm_A << "\n";
    std::cout << "Singular values norm: " << norm_S << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  tcapi::destroy_context(ctx);

  std::cout << "\nSVD example completed successfully!\n";
  return 0;
}