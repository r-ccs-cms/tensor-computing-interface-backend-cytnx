#include <cytnx.hpp>
#include <iostream>
#include <random>

#include "tci/tci.h"

int main() {
  std::cout << "TCI SVD Example\n";
  std::cout << "===============\n\n";

  // Create context
  cytnx::Device ctx;
  tci::create_context(ctx);

  try {
    // Create random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto rand_gen = [&dis, &gen]() { return std::complex<double>(dis(gen), dis(gen)); };

    // Create a random tensor for SVD: shape [3, 4, 12]
    std::cout << "Creating random tensor with shape [3, 4, 12]...\n";
    auto A = tci::random<cytnx::Tensor>(ctx, {3, 4, 12}, rand_gen);

    std::cout << "Original tensor shape: ";
    auto shape = tci::shape(ctx, A);
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) std::cout << " x ";
    }
    std::cout << "\n";

    // Perform SVD: treat first 2 bonds as rows, last bond as columns
    std::cout << "\nPerforming SVD (first 2 bonds as rows)...\n";
    cytnx::Tensor U, V_dag;
    cytnx::Tensor S_diag;

    tci::svd(ctx, A, 2, U, S_diag, V_dag);

    // Check shapes
    auto U_shape = tci::shape(ctx, U);
    auto S_shape = tci::shape(ctx, S_diag);
    auto V_shape = tci::shape(ctx, V_dag);

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
      auto sv = tci::get_elem(ctx, S_diag, {static_cast<tci::elem_coor_t<cytnx::Tensor>>(i)});
      std::cout << "S[" << i << "] = " << std::real(sv) << "\n";
    }

    // Calculate norms for verification
    auto norm_A = tci::norm(ctx, A);
    auto norm_S = tci::norm(ctx, S_diag);

    std::cout << "\nOriginal tensor norm: " << norm_A << "\n";
    std::cout << "Singular values norm: " << norm_S << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  tci::destroy_context(ctx);

  std::cout << "\nSVD example completed successfully!\n";
  return 0;
}