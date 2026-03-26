#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cmath>
#include <cytnx.hpp>

TEST_CASE("TCI Eigenvalue - invalid num_of_bds_as_row") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("eigvals with num_of_bds_as_row=2 on 2D tensor should throw") {
    tci::CytnxTensor<cytnx::cytnx_complex128> diagonal;
    tci::eye(ctx, 3, diagonal);
    tci::set_elem(ctx, diagonal, {1, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, diagonal, {2, 2}, cytnx::cytnx_complex128(3.0, 0.0));

    tci::CytnxTensor<cytnx::cytnx_complex128> eigenvals;
    CHECK_THROWS_AS(tci::eigvals(ctx, diagonal, 2, eigenvals), std::invalid_argument);
  }

  SUBCASE("eigvalsh with num_of_bds_as_row=2 on 2D tensor should throw") {
    tci::CytnxTensor<cytnx::cytnx_complex128> symmetric;
    tci::zeros(ctx, {2, 2}, symmetric);
    tci::set_elem(ctx, symmetric, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, symmetric, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, symmetric, {1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, symmetric, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));

    tci::real_ten_t<tci::CytnxTensor<cytnx::cytnx_complex128>> eigenvals;
    CHECK_THROWS_AS(tci::eigvalsh(ctx, symmetric, 2, eigenvals), std::invalid_argument);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Matrix exponential - complex64 dtype preservation") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  // Test that complex64 tensors preserve their dtype through anti-Hermitian path
  tci::CytnxTensor<cytnx::cytnx_complex64> anti_herm_f;
  tci::zeros(ctx, {2, 2}, anti_herm_f);
  tci::set_elem(ctx, anti_herm_f, {0, 1}, cytnx::cytnx_complex64(1.0f, 0.0f));
  tci::set_elem(ctx, anti_herm_f, {1, 0}, cytnx::cytnx_complex64(-1.0f, 0.0f));

  tci::CytnxTensor<cytnx::cytnx_complex64> exp_h_f;
  tci::exp(ctx, anti_herm_f, 1, exp_h_f);

  // Verify dtype is preserved (Cytnx-specific check)
  CHECK(exp_h_f.backend.dtype() == cytnx::Type.ComplexFloat);

  // Verify unitarity with lower tolerance for float32
  auto e00 = tci::get_elem(ctx, exp_h_f, {0, 0});
  auto e10 = tci::get_elem(ctx, exp_h_f, {1, 0});
  float col0_norm_sq = std::norm(e00) + std::norm(e10);
  CHECK(std::abs(col0_norm_sq - 1.0f) < 1e-2f);

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Matrix inverse - singular matrix error") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  tci::CytnxTensor<cytnx::cytnx_complex128> singular;
  tci::zeros(ctx, {2, 2}, singular);
  tci::set_elem(ctx, singular, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
  tci::set_elem(ctx, singular, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
  tci::set_elem(ctx, singular, {1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
  tci::set_elem(ctx, singular, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));

  std::cout << "\n[Expected Cytnx Error] Testing singular matrix - Cytnx will output LAPACK "
               "error (zgetrf INFO=2). This is expected behavior.\n"
            << std::endl;

  tci::CytnxTensor<cytnx::cytnx_complex128> tmp;
  CHECK_THROWS_AS(tci::inverse(ctx, singular, 1, tmp), std::runtime_error);

  tci::destroy_context(ctx);
}
