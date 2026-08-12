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

TEST_CASE("TCI Matrix exponential - in-place real dtype preservation (double)") {
  // Regression for: in-place tci::exp on a real-typed tensor must return a real
  // backend so tci::get_elem reads correct values via at<double>. The
  // out-of-place tcict conformance tests do not exercise the in-place overload,
  // and the eigendecomposition path returns a complex tensor that previously
  // leaked into the user's real backend.
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_double>> ctx;
  tci::create_context(ctx);

  tci::CytnxTensor<cytnx::cytnx_double> diagonal;
  tci::zeros(ctx, {2, 2}, diagonal);
  tci::set_elem(ctx, diagonal, {0, 0}, 1.0);
  tci::set_elem(ctx, diagonal, {1, 1}, 2.0);

  tci::exp(ctx, diagonal, 1);

  CHECK(diagonal.backend.dtype() == cytnx::Type.Double);
  CHECK(std::abs(tci::get_elem(ctx, diagonal, {0, 0}) - std::exp(1.0)) < 1e-10);
  CHECK(std::abs(tci::get_elem(ctx, diagonal, {1, 1}) - std::exp(2.0)) < 1e-10);
  CHECK(std::abs(tci::get_elem(ctx, diagonal, {0, 1})) < 1e-10);
  CHECK(std::abs(tci::get_elem(ctx, diagonal, {1, 0})) < 1e-10);

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Matrix exponential - in-place real dtype preservation (float)") {
  // Regression for the float-specific failure mode: the previous implementation
  // hit `astype(Float)` on a complex result inside the anti-Hermitian path,
  // which Cytnx rejects ("not support type with dtype=4"). The unified
  // post-block now takes the real part first so the precision-aligning cast is
  // real-to-real and never throws. Numerical correctness for cytnx_float is
  // upstream-blocked on Cytnx single-precision Eigh and is covered by the
  // double-precision cases above.
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_float>> ctx;
  tci::create_context(ctx);

  tci::CytnxTensor<cytnx::cytnx_float> anti_herm;
  tci::zeros(ctx, {2, 2}, anti_herm);
  tci::set_elem(ctx, anti_herm, {0, 1}, 1.0f);
  tci::set_elem(ctx, anti_herm, {1, 0}, -1.0f);

  CHECK_NOTHROW(tci::exp(ctx, anti_herm, 1));
  CHECK(anti_herm.backend.dtype() == cytnx::Type.Float);

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

// Backend-specific coverage for the precision of epsilon in single precision.
//
// trunc_svd assembles epsilon from the singular values alone, in double. What
// it must not do is measure the discarded weight against a separately computed
// ||A||_F: Norm follows the input dtype, so on a float instantiation that norm
// carries ~1e-7 relative error, which enters epsilon as an absolute error and
// leaves any target below it decided by the rounding rather than by the data.
//
// Each fixture is diagonal with its trailing singular values sized to put the
// discarded weight at that scale, and each pair of targets brackets the true
// epsilon(chi=1) from either side -- which pins where the selection boundary
// lies rather than only bounding it. The first two leave chi_max at the full
// rank, so every discarded value is one the selection dropped; the third puts
// chi_max below it so the SVD cuts one first, which is the only path that reads
// the discarded values Cytnx returns.
//
// The precision of ||A||_F is a cytnx-backend detail, not a TCI spec form, so
// this is covered here rather than in the conformance suite.
TEST_CASE("TCI trunc_svd - float epsilon survives the norm's own rounding") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_float>;
  tci::context_handle_t<Ten> ctx;
  tci::create_context(ctx);

  // diag(svs) truncated to chi_max, then to whatever `target` allows. Returns
  // how many singular values survived both.
  auto retained_at = [&](const std::vector<float>& svs, int chi_max, double target) {
    const auto n = static_cast<cytnx::cytnx_uint64>(svs.size());
    Ten a;
    tci::zeros(ctx, {n, n}, a);
    for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
      tci::set_elem(ctx, a, {i, i}, svs[i]);
    }

    Ten u, v_dag;
    tci::real_ten_t<Ten> s_diag;
    tci::real_t<Ten> trunc_err;
    tci::trunc_svd(ctx, a, 1, u, s_diag, v_dag, trunc_err, static_cast<tci::bond_dim_t<Ten>>(1),
                   static_cast<tci::bond_dim_t<Ten>>(chi_max),
                   static_cast<tci::real_t<Ten>>(target), static_cast<tci::real_t<Ten>>(0.0));
    auto s_shape = tci::shape(ctx, s_diag);
    REQUIRE(s_shape.size() == 1);
    return s_shape[0];
  };

  SUBCASE("a tail larger than the rounding is still measured, not swallowed") {
    // epsilon(chi=1) is 9.99999e-7, an order above the rounding, so the bound
    // must be read off the tail itself rather than off anything the norm's
    // error can move by 4.6%.
    CHECK(retained_at({1.0f, 1e-3f}, 2, 9.8e-7) == 2);
    CHECK(retained_at({1.0f, 1e-3f}, 2, 1.1e-6) == 1);
  }

  SUBCASE("a tail the size of the rounding is not confused with it") {
    // epsilon(chi=1) is 1.19212e-7, placed deliberately at the float rounding
    // of ||A||_F^2 itself (2^-23 = 1.192e-7): a formulation deriving the
    // discarded weight from that norm answers with the rounding here, in
    // whichever direction the norm happens to round.
    CHECK(retained_at({1.0f, 0.000345271f}, 2, 1.5e-7) == 1);
    CHECK(retained_at({1.0f, 0.000345271f}, 2, 1.0e-7) == 2);
  }

  SUBCASE("weight the SVD itself cut still counts toward epsilon") {
    // chi_max = 2 makes the SVD drop 1e-4 before trunc_svd sees it, so this is
    // the one path that reads the discarded values Cytnx returns. That weight
    // is 1e-8 of the total, below the rounding and therefore unrecoverable from
    // the norm, and it is what separates epsilon(chi=1) = 1.00999e-6 from the
    // 9.99999e-7 the returned values alone account for.
    CHECK(retained_at({1.0f, 1e-3f, 1e-4f}, 2, 1.005e-6) == 2);
    CHECK(retained_at({1.0f, 1e-3f, 1e-4f}, 2, 1.02e-6) == 1);
  }

  tci::destroy_context(ctx);
}

// The same precision concern one level down, where only double reaches.
//
// A discarded weight derived as a difference of two sums over the whole array
// vanishes once it falls below the accumulation's ~1e-16 of the total, because
// both sums round to the same double. diag(1, 1e-9) is past that edge: the
// total rounds to exactly the retained sum, so the difference is 0 where
// epsilon(chi=1) is 1e-18, and every positive target is then met at chi = 1.
// Summing the dropped values directly keeps the tail representable, since 1e-18
// is a fine double on its own -- it is only unrepresentable as an increment to
// 1. The targets bracket 1e-18.
TEST_CASE("TCI trunc_svd - a tail below the accumulation's resolution still counts") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_double>;
  tci::context_handle_t<Ten> ctx;
  tci::create_context(ctx);

  Ten a;
  tci::zeros(ctx, {2, 2}, a);
  tci::set_elem(ctx, a, {0, 0}, 1.0);
  tci::set_elem(ctx, a, {1, 1}, 1e-9);

  auto retained_at = [&](double target) {
    Ten u, v_dag;
    tci::real_ten_t<Ten> s_diag;
    tci::real_t<Ten> trunc_err;
    tci::trunc_svd(ctx, a, 1, u, s_diag, v_dag, trunc_err, static_cast<tci::bond_dim_t<Ten>>(1),
                   static_cast<tci::bond_dim_t<Ten>>(2), static_cast<tci::real_t<Ten>>(target),
                   static_cast<tci::real_t<Ten>>(0.0));
    auto s_shape = tci::shape(ctx, s_diag);
    REQUIRE(s_shape.size() == 1);
    return s_shape[0];
  };

  CHECK(retained_at(1e-20) == 2);
  CHECK(retained_at(1e-17) == 1);

  tci::destroy_context(ctx);
}

// Scale invariance is a property the criterion is defined by, not a range it
// happens to cover: epsilon is a ratio, so scaling `a` by any constant must
// leave the retained chi alone. Squaring singular values as they come breaks
// that at the ends of the double range -- diag(1e-200, 5e-201) squares to
// 1e-400 and 2.5e-401, both of which underflow to zero, so the total does too
// and the selection is skipped entirely. epsilon(chi=1) is 0.2 whatever the
// scale, so the targets that bracket 0.2 must bracket it here as well.
TEST_CASE("TCI trunc_svd - the relative criterion holds at extreme scales") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_double>;
  tci::context_handle_t<Ten> ctx;
  tci::create_context(ctx);

  auto retained_at = [&](double scale, double target) {
    Ten a;
    tci::zeros(ctx, {2, 2}, a);
    tci::set_elem(ctx, a, {0, 0}, 2.0 * scale);
    tci::set_elem(ctx, a, {1, 1}, 1.0 * scale);

    Ten u, v_dag;
    tci::real_ten_t<Ten> s_diag;
    tci::real_t<Ten> trunc_err;
    tci::trunc_svd(ctx, a, 1, u, s_diag, v_dag, trunc_err, static_cast<tci::bond_dim_t<Ten>>(1),
                   static_cast<tci::bond_dim_t<Ten>>(2), static_cast<tci::real_t<Ten>>(target),
                   static_cast<tci::real_t<Ten>>(0.0));
    auto s_shape = tci::shape(ctx, s_diag);
    REQUIRE(s_shape.size() == 1);
    return s_shape[0];
  };

  // epsilon(chi=1) = 1 / (4 + 1) = 0.2 at every scale.
  for (double scale : {1.0, 1e-200, 1e200}) {
    CHECK(retained_at(scale, 0.3) == 1);
    CHECK(retained_at(scale, 0.1) == 2);
  }

  tci::destroy_context(ctx);
}

// Backend-specific coverage for the gesdd fast path.
//
// tci::svd / tci::trunc_svd route through the gesdd (divide-and-conquer)
// backend routine only when the smaller matrix dimension is >= kGesddMinDim
// (64), and only when the build-time TCICYTNX_USE_GESDD capability is enabled.
// The conformance fixtures are all small (<= 4x6) and therefore only ever
// exercise the gesvd path.  This is a cytnx-backend implementation detail, not
// part of the TCI spec, so the >= 64 case is covered here rather than in the
// conformance suite.
//
// A 64x64 identity has all singular values exactly 1, giving a deterministic
// check.  Under TCICYTNX_USE_GESDD=ON (the OpenBLAS/MKL default) this drives
// the gesdd branch; under OFF it drives gesvd.  The assertions hold either way,
// so the test is valid in both build configurations while adding the missing
// gesdd-branch coverage on the default development build.
TEST_CASE("TCI SVD - large matrix exercises gesdd path (64x64 identity)") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_double>;
  tci::context_handle_t<Ten> ctx;
  tci::create_context(ctx);

  // n == kGesddMinDim: the smallest dimension that takes the gesdd branch.
  constexpr int n = 64;
  Ten a;
  tci::eye(ctx, n, a);

  SUBCASE("full svd: identity has 64 singular values all equal to 1") {
    Ten u, v_dag;
    tci::real_ten_t<Ten> s_diag;
    tci::svd(ctx, a, 1, u, s_diag, v_dag);

    auto s_shape = tci::shape(ctx, s_diag);
    REQUIRE(s_shape.size() == 1);
    CHECK(s_shape[0] == n);
    for (cytnx::cytnx_uint64 i = 0; i < static_cast<cytnx::cytnx_uint64>(n); ++i) {
      CHECK(tci::get_elem(ctx, s_diag, {i}) == doctest::Approx(1.0));
    }
  }

  SUBCASE("trunc svd: keep all 64, singular values all 1, zero truncation error") {
    Ten u, v_dag;
    tci::real_ten_t<Ten> s_diag;
    tci::real_t<Ten> trunc_err;
    tci::trunc_svd(ctx, a, 1, u, s_diag, v_dag, trunc_err, static_cast<tci::bond_dim_t<Ten>>(n),
                   static_cast<tci::real_t<Ten>>(0.0));

    auto s_shape = tci::shape(ctx, s_diag);
    REQUIRE(s_shape.size() == 1);
    CHECK(s_shape[0] == n);
    for (cytnx::cytnx_uint64 i = 0; i < static_cast<cytnx::cytnx_uint64>(n); ++i) {
      CHECK(tci::get_elem(ctx, s_diag, {i}) == doctest::Approx(1.0));
    }
    CHECK(trunc_err == doctest::Approx(0.0));
  }

  tci::destroy_context(ctx);
}
