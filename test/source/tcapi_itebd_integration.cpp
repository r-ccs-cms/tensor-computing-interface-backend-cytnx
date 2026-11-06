#include <doctest/doctest.h>

#include <cytnx.hpp>
#include <random>

#include "tcapi/tcapi.h"

// iTEBD uses real tensors for imaginary-time evolution
using Tensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
using ContextHandle = tcapi::context_handle_t<Tensor>;
using Elem = tcapi::elem_t<Tensor>;  // double
using Real = tcapi::real_t<Tensor>;  // double

TEST_CASE("iTEBD Integration Test - Comprehensive TCAPI API Usage") {
  // Create context
  ContextHandle context;
  tcapi::create_context(context);

  // Define the parameters of the model and the simulation
  Real J = 1.0, g = 0.5;
  Real dt = 0.005;
  size_t d = 2, chi = 5;
  size_t N = 3000;

  // Prepare Hamiltonian and imaginary-time evolution operator
  auto g2 = -0.5 * g;
  // Initialize Hamiltonian matrix elements
  std::vector<Elem> h_elements = {J, g2, g2, 0.0, g2, -J, 0.0, g2, g2, 0.0, -J, g2, 0.0, g2, g2, J};
  Tensor h = tcapi::assign_from_range<Tensor>(
      context, {4, 4}, h_elements.begin(),
      [](const tcapi::elem_coors_t<Tensor>& coors) { return coors[0] * 4 + coors[1]; });

  tcapi::transpose(context, h, {1, 0});
  Tensor v, w;
  tcapi::eigh(context, h, 1, w, v);
  // eigh returns the {4, 4} diagonal lambda_mat; take its diagonal, so the
  // exponential below maps the eigenvalues and not the off-diagonal zeros.
  tcapi::diag(context, w);

  // Apply exponential to eigenvalues
  tcapi::for_each(context, w, [dt](Elem& elem) { elem = std::exp(-dt * elem); });

  tcapi::diag(context, w);
  Tensor u;
  tcapi::contract(context, v, {0, -1}, w, {-1, 1}, u, {0, 1});
  tcapi::transpose(context, v, {1, 0});
  tcapi::contract(context, u, {0, -1}, v, {-1, 1}, u, {0, 1});
  tcapi::reshape(context, u, {d, d, d, d});

  // Initialize infinite MPS
  std::mt19937 gen;
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  auto random_gen = [&gen, &dist]() { return dist(gen); };

  std::vector<Tensor> Gamma
      = {tcapi::random<Tensor>(context, tcapi::shape_t<Tensor>{chi, d, chi}, random_gen),
         tcapi::random<Tensor>(context, tcapi::shape_t<Tensor>{chi, d, chi}, random_gen)};
  std::vector<Tensor> Lambda
      = {tcapi::random<Tensor>(context, tcapi::shape_t<Tensor>{chi}, random_gen),
         tcapi::random<Tensor>(context, tcapi::shape_t<Tensor>{chi}, random_gen)};

  Tensor Theta;  // Declare outside loop for energy calculation

  // iTEBD iterations
  for (size_t i = 0; i < static_cast<size_t>(N); i++) {
    auto A = i % 2;
    auto B = (i + 1) % 2;
    auto LambdaA_full = tcapi::copy(context, Lambda[A]);
    tcapi::diag(context, LambdaA_full);
    auto LambdaB_full = tcapi::copy(context, Lambda[B]);
    tcapi::diag(context, LambdaB_full);

    // Build two-site wavefunction with correct NCON notation
    tcapi::contract(context, LambdaB_full, {0, -1}, Gamma[A], {-1, 1, 2}, Theta, {0, 1, 2});
    tcapi::contract(context, Theta, {0, 1, -1}, LambdaA_full, {-1, 2}, Theta, {0, 1, 2});
    tcapi::contract(context, Theta, {0, 1, -1}, Gamma[B], {-1, 2, 3}, Theta, {0, 1, 2, 3});
    tcapi::contract(context, Theta, {0, 1, 2, -1}, LambdaB_full, {-1, 3}, Theta, {0, 1, 2, 3});

    // Apply time evolution operator
    tcapi::contract(context, Theta, {0, -1, -2, 3}, u, {-1, -2, 1, 2}, Theta, {0, 1, 2, 3});

    // SVD decomposition
    Real trunc_err = 0;
    Tensor GA_new, GB_new, Lambda_new;
    tcapi::trunc_svd(context, Theta, 2, GA_new, Lambda_new, GB_new, trunc_err,
                     static_cast<tcapi::bond_dim_t<Tensor>>(chi), 0.0);

    // Update tensors. Lambda keeps the singular values as a list, so take the
    // diagonal of the {chi, chi} sigma trunc_svd returns.
    tcapi::diag(context, Lambda_new);
    Gamma[A] = GA_new;
    Gamma[B] = GB_new;
    Lambda[A] = Lambda_new;

    // Normalize singular values
    tcapi::normalize(context, Lambda[A]);

    // Inverse lambda for canonicalization
    auto LambdaB_inv = tcapi::copy(context, Lambda[B]);
    tcapi::for_each(context, LambdaB_inv, [](Elem& elem) { elem = 1.0 / elem; });
    tcapi::diag(context, LambdaB_inv);

    // Apply inverse lambda
    tcapi::contract(context, LambdaB_inv, {0, -1}, Gamma[A], {-1, 1, 2}, Gamma[A], {0, 1, 2});
    tcapi::contract(context, Gamma[B], {0, 1, -1}, LambdaB_inv, {-1, 2}, Gamma[B], {0, 1, 2});

    // Basic checks for integration test
    CHECK(trunc_err >= 0.0);
    CHECK(std::isfinite(trunc_err));
  }

  // Calculate energy
  Elem E_iTEBD = 0.0;
  tcapi::for_each(context, Theta, [&E_iTEBD](const Elem& elem) { E_iTEBD += (elem * elem); });
  E_iTEBD = -std::log(E_iTEBD) / dt / 2.0;
  std::printf("E_iTEBD = %.15f\n", E_iTEBD);

  // Integration test checks
  CHECK(std::isfinite(E_iTEBD));

  // Verify tensor shapes and properties
  auto gamma_shape = tcapi::shape(context, Gamma[0]);
  CHECK(gamma_shape.size() == 3);

  auto lambda_shape = tcapi::shape(context, Lambda[0]);
  CHECK(lambda_shape.size() == 1);

  // Destroy context
  tcapi::destroy_context(context);
}
