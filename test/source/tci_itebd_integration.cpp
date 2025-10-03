#include <doctest/doctest.h>

#include <cytnx.hpp>
#include <random>

#include "tci/tci.h"

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using ContextHandle = tci::context_handle_t<Tensor>;
using Elem = tci::elem_t<Tensor>;
using Real = double;
using Int = tci::bond_label_t<Tensor>;

TEST_CASE("iTEBD Integration Test - Comprehensive TCI API Usage") {
  // Create context
  ContextHandle context;
  tci::create_context(context);

  // Define the parameters of the model and the simulation
  Real J = 1.0, g = 0.5;
  Real dt = 0.005;
  size_t d = 2, chi = 5;
  size_t N = 3000;

  // Prepare Hamiltonian and imaginary-time evolution operator
  auto g2 = -0.5 * g;
  // Initialize Hamiltonian matrix elements
  std::vector<Elem> h_elements = {J, g2, g2, 0.0, g2, -J, 0.0, g2, g2, 0.0, -J, g2, 0.0, g2, g2, J};
  Tensor h;
  tci::assign_from_container(
      context, {4, 4}, h_elements.begin(),
      [](const tci::elem_coors_t<Tensor>& coors) { return coors[0] * 4 + coors[1]; }, h);

  tci::transpose(context, h, {1, 0});
  Tensor v, w;
  tci::eigh(context, h, 1, w, v);

  // Apply exponential to eigenvalues
  tci::for_each(context, w, [dt](Elem& elem) {
    std::visit([dt, &elem](auto&& v) {
      using T = std::decay_t<decltype(v)>;
      if constexpr (std::is_arithmetic_v<T>) {
        elem = cytnx::cytnx_complex128(std::exp(-dt * v), 0.0);
      } else {
        elem = cytnx::cytnx_complex128(std::exp(-dt * v.real()), std::exp(-dt * v.imag()));
      }
    }, elem);
  });

  tci::diag(context, w);
  Tensor u;
  tci::contract(context, v, {0, -1}, w, {-1, 1}, u, {0, 1});
  tci::transpose(context, v, {1, 0});
  tci::contract(context, u, {0, -1}, v, {-1, 1}, u, {0, 1});
  tci::reshape(context, u, {d, d, d, d});

  // Initialize infinite MPS
  std::mt19937 gen;
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  auto random_gen = [&gen, &dist]() { return dist(gen); };

  std::vector<Tensor> Gamma
      = {tci::random<Tensor>(context, tci::shape_t<Tensor>{chi, d, chi}, random_gen),
         tci::random<Tensor>(context, tci::shape_t<Tensor>{chi, d, chi}, random_gen)};
  std::vector<Tensor> Lambda
      = {tci::random<Tensor>(context, tci::shape_t<Tensor>{chi}, random_gen),
         tci::random<Tensor>(context, tci::shape_t<Tensor>{chi}, random_gen)};

  Tensor Theta;  // Declare outside loop for energy calculation

  // iTEBD iterations
  for (size_t i = 0; i < static_cast<size_t>(N); i++) {
    auto A = i % 2;
    auto B = (i + 1) % 2;
    auto LambdaA_full = tci::copy(context, Lambda[A]);
    tci::diag(context, LambdaA_full);
    auto LambdaB_full = tci::copy(context, Lambda[B]);
    tci::diag(context, LambdaB_full);

    // Build two-site wavefunction with correct NCON notation
    tci::contract(context, LambdaB_full, {0, -1}, Gamma[A], {-1, 1, 2}, Theta, {0, 1, 2});
    tci::contract(context, Theta, {0, 1, -1}, LambdaA_full, {-1, 2}, Theta, {0, 1, 2});
    tci::contract(context, Theta, {0, 1, -1}, Gamma[B], {-1, 2, 3}, Theta, {0, 1, 2, 3});
    tci::contract(context, Theta, {0, 1, 2, -1}, LambdaB_full, {-1, 3}, Theta, {0, 1, 2, 3});

    // Apply time evolution operator
    tci::contract(context, Theta, {0, -1, -2, 3}, u, {-1, -2, 1, 2}, Theta, {0, 1, 2, 3});

    // SVD decomposition
    Real trunc_err = 0;
    Tensor GA_new, GB_new;
    tci::real_ten_t<Tensor> Lambda_new;
    tci::trunc_svd(context, Theta, 2, GA_new, Lambda_new, GB_new, trunc_err, chi, 0.0);

    // Update tensors
    Gamma[A] = GA_new;
    Gamma[B] = GB_new;
    Lambda[A] = Lambda_new;

    // Normalize singular values
    tci::normalize(context, Lambda[A]);

    // Inverse lambda for canonicalization
    auto LambdaB_inv = tci::copy(context, Lambda[B]);
    tci::for_each(context, LambdaB_inv, [](Elem& elem) {
      std::visit([&elem](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<T>) {
          elem = cytnx::cytnx_complex128(1.0 / v, 0.0);
        } else {
          // Calculate 1/elem for complex: 1/(a+bi) = (a-bi)/(a²+b²)
          auto denom = v.real() * v.real() + v.imag() * v.imag();
          elem = cytnx::cytnx_complex128(v.real() / denom, -v.imag() / denom);
        }
      }, elem);
    });
    tci::diag(context, LambdaB_inv);

    // Apply inverse lambda
    tci::contract(context, LambdaB_inv, {0, -1}, Gamma[A], {-1, 1, 2}, Gamma[A], {0, 1, 2});
    tci::contract(context, Gamma[B], {0, 1, -1}, LambdaB_inv, {-1, 2}, Gamma[B], {0, 1, 2});

    // Basic checks for integration test
    CHECK(trunc_err >= 0.0);
    CHECK(std::isfinite(trunc_err));
  }

  // Calculate energy
  double E_iTEBD_real = 0.0;
  double E_iTEBD_imag = 0.0;
  tci::for_each(context, Theta, [&E_iTEBD_real, &E_iTEBD_imag](const Elem& elem) {
    std::visit([&E_iTEBD_real, &E_iTEBD_imag](auto&& v) {
      using T = std::decay_t<decltype(v)>;
      if constexpr (std::is_arithmetic_v<T>) {
        // Real type: elem * elem = v²
        E_iTEBD_real += v * v;
      } else {
        // Complex: elem * elem = (a + bi) * (a + bi) = a² - b² + 2abi
        auto square_real = v.real() * v.real() - v.imag() * v.imag();
        auto square_imag = 2.0 * v.real() * v.imag();
        E_iTEBD_real += square_real;
        E_iTEBD_imag += square_imag;
      }
    }, elem);
  });

  // Convert to energy value
  auto E_complex = cytnx::cytnx_complex128(E_iTEBD_real, E_iTEBD_imag);
  Elem E_iTEBD = cytnx::cytnx_complex128(-std::log(std::abs(E_complex)) / dt / 2.0, 0.0);

  std::visit([](auto&& v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_arithmetic_v<T>) {
      std::printf("E_iTEBD = %.15f\n", v);
    } else {
      std::printf("E_iTEBD = %.15f\n", v.real());
    }
  }, E_iTEBD);

  // Integration test checks
  std::visit([](auto&& v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_arithmetic_v<T>) {
      CHECK(std::isfinite(v));
    } else {
      CHECK(std::isfinite(v.real()));
      CHECK(std::isfinite(v.imag()));
    }
  }, E_iTEBD);

  // Verify tensor shapes and properties
  auto gamma_shape = tci::shape(context, Gamma[0]);
  CHECK(gamma_shape.size() == 3);

  auto lambda_shape = tci::shape(context, Lambda[0]);
  CHECK(lambda_shape.size() == 1);

  // Destroy context
  tci::destroy_context(context);
}
