#include <doctest/doctest.h>

#include <complex>
#include <cytnx.hpp>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>

#include "diagonal_check.h"
#include "tcapi/miscellaneous.h"
#include "tcapi/tcapi.h"
#include "tcapi/tensor_linear_algebra.h"

TEST_CASE("CytnxTensor - Type Traits") {
  SUBCASE("Double precision real tensor") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    using Real = tcapi::tensor_traits<Tensor>::real_t;
    using Cplx = tcapi::tensor_traits<Tensor>::cplx_t;

    // elem_t should be cytnx::cytnx_double (= double)
    CHECK(std::is_same_v<Elem, cytnx::cytnx_double>);
    CHECK(std::is_same_v<Elem, double>);

    // real_t should equal elem_t for real tensors
    CHECK(std::is_same_v<Real, Elem>);

    // cplx_t should be complex128
    CHECK(std::is_same_v<Cplx, cytnx::cytnx_complex128>);
  }

  SUBCASE("Double precision complex tensor") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    using Real = tcapi::tensor_traits<Tensor>::real_t;
    using Cplx = tcapi::tensor_traits<Tensor>::cplx_t;

    // elem_t should be cytnx::cytnx_complex128 (= std::complex<double>)
    CHECK(std::is_same_v<Elem, cytnx::cytnx_complex128>);
    CHECK(std::is_same_v<Elem, std::complex<double>>);

    // real_t should be double
    CHECK(std::is_same_v<Real, cytnx::cytnx_double>);
    CHECK(std::is_same_v<Real, double>);

    // cplx_t should equal elem_t for complex tensors
    CHECK(std::is_same_v<Cplx, Elem>);
  }

  SUBCASE("Single precision real tensor") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_float>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    using Real = tcapi::tensor_traits<Tensor>::real_t;
    using Cplx = tcapi::tensor_traits<Tensor>::cplx_t;

    // elem_t should be cytnx::cytnx_float (= float)
    CHECK(std::is_same_v<Elem, cytnx::cytnx_float>);
    CHECK(std::is_same_v<Elem, float>);

    // real_t should equal elem_t for real tensors
    CHECK(std::is_same_v<Real, Elem>);

    // cplx_t should be complex64
    CHECK(std::is_same_v<Cplx, cytnx::cytnx_complex64>);
  }

  SUBCASE("Single precision complex tensor") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex64>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    using Real = tcapi::tensor_traits<Tensor>::real_t;
    using Cplx = tcapi::tensor_traits<Tensor>::cplx_t;

    // elem_t should be cytnx::cytnx_complex64 (= std::complex<float>)
    CHECK(std::is_same_v<Elem, cytnx::cytnx_complex64>);
    CHECK(std::is_same_v<Elem, std::complex<float>>);

    // real_t should be float
    CHECK(std::is_same_v<Real, cytnx::cytnx_float>);
    CHECK(std::is_same_v<Real, float>);

    // cplx_t should equal elem_t for complex tensors
    CHECK(std::is_same_v<Cplx, Elem>);
  }
}

TEST_CASE("CytnxTensor - Arithmetic Operations on elem_t") {
  using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tcapi::tensor_traits<Tensor>::elem_t;

  SUBCASE("Element type supports standard arithmetic") {
    Elem a = {3.0, 4.0};
    Elem b = {1.0, 2.0};

    // Basic arithmetic
    Elem sum = a + b;
    CHECK(sum.real() == doctest::Approx(4.0));
    CHECK(sum.imag() == doctest::Approx(6.0));

    Elem diff = a - b;
    CHECK(diff.real() == doctest::Approx(2.0));
    CHECK(diff.imag() == doctest::Approx(2.0));

    Elem prod = a * b;
    CHECK(prod.real() == doctest::Approx(-5.0));
    CHECK(prod.imag() == doctest::Approx(10.0));

    Elem quot = a / b;
    CHECK(quot.real() == doctest::Approx(2.2));
    CHECK(quot.imag() == doctest::Approx(-0.4));
  }

  SUBCASE("Element type supports standard math functions") {
    Elem a = {3.0, 4.0};

    // Standard library functions
    auto magnitude = std::abs(a);
    CHECK(magnitude == doctest::Approx(5.0));

    auto root = std::sqrt(Elem{9.0, 0.0});
    CHECK(root.real() == doctest::Approx(3.0));
    CHECK(root.imag() == doctest::Approx(0.0));

    auto exponential = std::exp(Elem{0.0, 0.0});
    CHECK(exponential.real() == doctest::Approx(1.0));
    CHECK(exponential.imag() == doctest::Approx(0.0));
  }
}

TEST_CASE("CytnxTensor - Construction") {
  SUBCASE("Default construction") {
    tcapi::CytnxTensor<cytnx::cytnx_double> t;
    // Should compile and not crash
    CHECK(true);
  }

  SUBCASE("Construction from cytnx::Tensor") {
    cytnx::Tensor backend = cytnx::zeros({2, 2}, cytnx::Type.Double);
    tcapi::CytnxTensor<cytnx::cytnx_double> t(backend);

    // Backend should be accessible
    CHECK(t.backend.shape().size() == 2);
    CHECK(t.backend.shape()[0] == 2);
    CHECK(t.backend.shape()[1] == 2);
  }

  SUBCASE("Copy construction") {
    cytnx::Tensor backend = cytnx::zeros({3, 3}, cytnx::Type.ComplexDouble);
    tcapi::CytnxTensor<cytnx::cytnx_complex128> t1(backend);
    tcapi::CytnxTensor<cytnx::cytnx_complex128> t2 = t1;

    // Both should have valid backends
    CHECK(t1.backend.shape().size() == 2);
    CHECK(t2.backend.shape().size() == 2);
  }

  SUBCASE("Move construction") {
    cytnx::Tensor backend = cytnx::zeros({4, 4}, cytnx::Type.Float);
    tcapi::CytnxTensor<cytnx::cytnx_float> t1(backend);
    tcapi::CytnxTensor<cytnx::cytnx_float> t2 = std::move(t1);

    // t2 should have a valid backend
    CHECK(t2.backend.shape().size() == 2);
  }

  SUBCASE("clear function") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
    tcapi::context_handle_t<Tensor> ctx;
    tcapi::create_context(ctx);

    // Create and fill a tensor
    Tensor tensor;
    tcapi::fill(ctx, {3, 4}, 5.0, tensor);

    // Verify tensor is initialized
    CHECK(tensor.backend.shape().size() == 2);
    CHECK(tensor.backend.shape()[0] == 3);
    CHECK(tensor.backend.shape()[1] == 4);

    // Clear the tensor
    CHECK_NOTHROW(tcapi::clear(ctx, tensor));

    // After clear, tensor should be empty (shape size 0)
    CHECK(tensor.backend.shape().size() == 0);
  }

  SUBCASE("assign_from_container with row-major indexing") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    tcapi::context_handle_t<Tensor> ctx;
    tcapi::create_context(ctx);

    // Create a 2x3 tensor from std::vector
    std::vector<std::complex<double>> container
        = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}};

    auto coors2idx = [](const tcapi::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[0] * 3 + coors[1];  // row-major for 2x3 matrix
    };

    tcapi::shape_t<Tensor> shape = {2, 3};
    Tensor tensor;

    CHECK_NOTHROW(tensor
                  = tcapi::assign_from_range<Tensor>(ctx, shape, container.begin(), coors2idx));

    // Verify tensor properties
    CHECK(tensor.backend.shape().size() == 2);
    CHECK(tensor.backend.shape()[0] == 2);
    CHECK(tensor.backend.shape()[1] == 3);

    // Verify element values
    Elem elem_00 = tcapi::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(elem_00.real() - 1.0) < 1e-10);

    Elem elem_01 = tcapi::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(elem_01.real() - 2.0) < 1e-10);

    Elem elem_10 = tcapi::get_elem(ctx, tensor, {1, 0});
    CHECK(std::abs(elem_10.real() - 4.0) < 1e-10);

    Elem elem_12 = tcapi::get_elem(ctx, tensor, {1, 2});
    CHECK(std::abs(elem_12.real() - 6.0) < 1e-10);
  }

  SUBCASE("assign_from_container with column-major indexing") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    tcapi::context_handle_t<Tensor> ctx;
    tcapi::create_context(ctx);

    // Create a 2x2 tensor with column-major layout
    std::vector<double> container = {1.0, 3.0, 2.0, 4.0};

    auto coors2idx = [](const tcapi::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[1] * 2 + coors[0];  // column-major for 2x2 matrix
    };

    tcapi::shape_t<Tensor> shape = {2, 2};
    Tensor tensor;

    CHECK_NOTHROW(tensor
                  = tcapi::assign_from_range<Tensor>(ctx, shape, container.begin(), coors2idx));

    // Verify element values with column-major layout
    Elem elem_00 = tcapi::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(elem_00 - 1.0) < 1e-10);

    Elem elem_01 = tcapi::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(elem_01 - 2.0) < 1e-10);

    Elem elem_10 = tcapi::get_elem(ctx, tensor, {1, 0});
    CHECK(std::abs(elem_10 - 3.0) < 1e-10);

    Elem elem_11 = tcapi::get_elem(ctx, tensor, {1, 1});
    CHECK(std::abs(elem_11 - 4.0) < 1e-10);
  }
}

TEST_CASE("CytnxTensor - TCAPI API Integration") {
  using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tcapi::tensor_traits<Tensor>::elem_t;
  using ContextHandle = tcapi::tensor_traits<Tensor>::context_handle_t;

  ContextHandle ctx;
  tcapi::create_context(ctx);

  SUBCASE("allocate and get_elem") {
    Tensor tensor;
    tcapi::allocate(ctx, {2, 3}, tensor);

    // Check shape
    CHECK(tensor.backend.shape().size() == 2);
    CHECK(tensor.backend.shape()[0] == 2);
    CHECK(tensor.backend.shape()[1] == 3);

    // Check dtype
    CHECK(tensor.backend.dtype() == cytnx::Type.ComplexDouble);

    // Set and get element
    tensor.backend.at<Elem>({0, 0}) = Elem{3.0, 4.0};
    Elem retrieved = tcapi::get_elem(ctx, tensor, {0, 0});

    CHECK(retrieved.real() == doctest::Approx(3.0));
    CHECK(retrieved.imag() == doctest::Approx(4.0));
  }

  SUBCASE("Element arithmetic operations") {
    Tensor tensor;
    tcapi::allocate(ctx, {2, 2}, tensor);

    tensor.backend.at<Elem>({0, 0}) = Elem{1.0, 2.0};
    tensor.backend.at<Elem>({0, 1}) = Elem{3.0, 4.0};

    Elem a = tcapi::get_elem(ctx, tensor, {0, 0});
    Elem b = tcapi::get_elem(ctx, tensor, {0, 1});

    // Standard C++ operators work!
    Elem sum = a + b;
    CHECK(sum.real() == doctest::Approx(4.0));
    CHECK(sum.imag() == doctest::Approx(6.0));

    Elem product = a * b;
    CHECK(product.real() == doctest::Approx(-5.0));
    CHECK(product.imag() == doctest::Approx(10.0));

    // Standard math functions work!
    double magnitude_a = std::abs(a);  // std::abs returns double for complex
    CHECK(magnitude_a == doctest::Approx(std::sqrt(5.0)));

    Elem sqrt_b = std::sqrt(Elem{9.0, 0.0});
    CHECK(sqrt_b.real() == doctest::Approx(3.0));
  }
}

TEST_CASE("CytnxTensor - for_each with arithmetic operations") {
  using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tcapi::tensor_traits<Tensor>::elem_t;
  using ContextHandle = tcapi::tensor_traits<Tensor>::context_handle_t;

  ContextHandle ctx;
  tcapi::create_context(ctx);

  SUBCASE("for_each with std::sqrt") {
    Tensor tensor;
    tcapi::allocate(ctx, {3, 3}, tensor);

    // Fill with constant values
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{4.0, 0.0};
      }
    }

    // Apply sqrt to each element
    tcapi::for_each(ctx, tensor, [](Elem& elem) { elem = std::sqrt(elem); });

    // Verify all elements are now 2.0
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tcapi::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(2.0));
        CHECK(result.imag() == doctest::Approx(0.0));
      }
    }
  }

  SUBCASE("for_each with addition") {
    Tensor tensor;
    tcapi::allocate(ctx, {2, 2}, tensor);

    // Fill with constant values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{1.0, 2.0};
      }
    }

    // Add constant to each element
    Elem addend{3.0, 4.0};
    tcapi::for_each(ctx, tensor, [addend](Elem& elem) { elem = elem + addend; });

    // Verify all elements are now (4.0, 6.0)
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tcapi::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(4.0));
        CHECK(result.imag() == doctest::Approx(6.0));
      }
    }
  }

  SUBCASE("for_each with multiplication and division") {
    Tensor tensor;
    tcapi::allocate(ctx, {2, 3}, tensor);

    // Fill with constant values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{8.0, 0.0};
      }
    }

    // Multiply by 2, then divide by 4
    tcapi::for_each(ctx, tensor, [](Elem& elem) {
      elem = elem * Elem{2.0, 0.0};
      elem = elem / Elem{4.0, 0.0};
    });

    // Verify all elements are now 4.0
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tcapi::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(4.0));
        CHECK(result.imag() == doctest::Approx(0.0));
      }
    }
  }

  SUBCASE("for_each with std::abs and conditional") {
    Tensor tensor;
    tcapi::allocate(ctx, {3, 3}, tensor);

    // Fill with varying values
    int counter = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double val = (counter % 2 == 0) ? 1.0 : -1.0;
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{val, val};
        counter++;
      }
    }

    // Normalize each element
    tcapi::for_each(ctx, tensor, [](Elem& elem) {
      const double magnitude = std::abs(elem);
      if (magnitude > 1e-10) {
        elem = elem / magnitude;
      }
    });

    // Verify all elements have magnitude 1
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tcapi::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        double magnitude = std::abs(result);
        CHECK(magnitude == doctest::Approx(1.0));
      }
    }
  }

  SUBCASE("for_each with complex arithmetic - inversion") {
    Tensor tensor;
    tcapi::allocate(ctx, {2, 2}, tensor);

    // Fill with non-zero values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{2.0, 0.0};
      }
    }

    // Invert each element: elem = 1/elem
    tcapi::for_each(ctx, tensor, [](Elem& elem) { elem = Elem{1.0, 0.0} / elem; });

    // Verify all elements are now 0.5
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tcapi::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(0.5));
        CHECK(result.imag() == doctest::Approx(0.0));
      }
    }
  }

  SUBCASE("const for_each with arithmetic - sum accumulation") {
    Tensor tensor;
    tcapi::allocate(ctx, {3, 3}, tensor);

    // Fill with constant values
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{1.5, 2.5};
      }
    }

    // Accumulate sum using const for_each
    Elem sum{0.0, 0.0};
    tcapi::for_each(ctx, static_cast<const Tensor&>(tensor),
                    [&sum](const Elem& elem) { sum = sum + elem; });

    // Verify sum (9 elements * (1.5 + 2.5i) = (13.5 + 22.5i))
    CHECK(sum.real() == doctest::Approx(13.5));
    CHECK(sum.imag() == doctest::Approx(22.5));
  }

  SUBCASE("for_each with std::exp and std::log") {
    Tensor tensor;
    tcapi::allocate(ctx, {2, 2}, tensor);

    // Fill with constant values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{1.0, 0.0};
      }
    }

    // Apply exp then log (should return to original)
    tcapi::for_each(ctx, tensor, [](Elem& elem) {
      elem = std::exp(elem);
      elem = std::log(elem);
    });

    // Verify all elements are back to 1.0
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tcapi::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(result.imag() == doctest::Approx(0.0).epsilon(1e-10));
      }
    }
  }
}

TEST_CASE("CytnxTensor - trunc_svd operation") {
  using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  using RealTensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
  using Elem = tcapi::elem_t<Tensor>;
  using Real = tcapi::real_t<Tensor>;

  tcapi::context_handle_t<Tensor> ctx;
  tcapi::create_context(ctx);

  SUBCASE("Basic trunc_svd functionality") {
    // Create a 4x6 random matrix
    Tensor a;
    tcapi::allocate(ctx, {4, 6}, a);

    // Fill with simple values for reproducible test
    std::mt19937 rng(42);
    tcapi::random(ctx, {4, 6}, rng, a);

    // Perform truncated SVD
    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    tcapi::order_t<Tensor> num_rows = 1;
    tcapi::bond_dim_t<Tensor> chi_max = 4;
    Real s_min = 1e-10;

    CHECK_NOTHROW(tcapi::trunc_svd(ctx, a, num_rows, u, s_diag, v_dag, trunc_err, chi_max, s_min));

    // Verify dimensions
    auto u_shape = tcapi::shape(ctx, u);
    auto v_shape = tcapi::shape(ctx, v_dag);
    const auto svs = require_diagonal(ctx, s_diag);

    CHECK(u_shape[0] == 4);
    CHECK(v_shape[1] == 6);
    CHECK(svs.size() == u_shape[1]);
    CHECK(svs.size() == v_shape[0]);
  }

  SUBCASE("for_each on singular values with sqrt") {
    // Create a simple matrix for SVD
    Tensor a;
    tcapi::allocate(ctx, {3, 3}, a);

    std::mt19937 rng(123);
    tcapi::random(ctx, {3, 3}, rng, a);

    // Perform SVD
    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    tcapi::trunc_svd(ctx, a, 1, u, s_diag, v_dag, trunc_err,
                     tcapi::bond_dim_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>>(3), 1e-10);

    // Apply sqrt to singular values
    CHECK_NOTHROW(tcapi::for_each(ctx, s_diag, [](Real& elem) { elem = std::sqrt(elem); }));

    // Verify the diagonal is non-negative after sqrt; sqrt keeps the
    // off-diagonal zeros zero, which require_diagonal checks.
    for (const double val : require_diagonal(ctx, s_diag)) {
      CHECK(val >= 0.0);
    }
  }
}

TEST_CASE("CytnxTensor - Miscellaneous functions") {
  SUBCASE("create_context and destroy_context") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
    tcapi::context_handle_t<Tensor> ctx;

    // Create context
    CHECK_NOTHROW(tcapi::create_context(ctx));

    // Context should be initialized to GPU if available, otherwise CPU
    if (cytnx::Device.Ngpus > 0) {
      CHECK(ctx >= cytnx::Device.cuda);  // GPU context
    } else {
      CHECK(ctx == cytnx::Device.cpu);  // CPU context
    }

    // Destroy context (should not throw)
    CHECK_NOTHROW(tcapi::destroy_context(ctx));
  }

  SUBCASE("to_container with row-major indexing") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    tcapi::context_handle_t<Tensor> ctx;
    tcapi::create_context(ctx);

    // Create a 2x3 tensor with known values
    Tensor tensor;
    tcapi::allocate(ctx, {2, 3}, tensor);

    // Fill with values 1-6
    for (cytnx::cytnx_uint64 i = 0; i < 2; ++i) {
      for (cytnx::cytnx_uint64 j = 0; j < 3; ++j) {
        Elem value{static_cast<double>(i * 3 + j + 1), 0.0};
        tcapi::set_elem(ctx, tensor, {i, j}, value);
      }
    }

    // Extract to container with row-major indexing
    std::vector<std::complex<double>> container(6);
    auto coors2idx = [](const tcapi::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[0] * 3 + coors[1];  // row-major for 2x3 matrix
    };

    CHECK_NOTHROW(tcapi::to_container(ctx, tensor, container.begin(), coors2idx));

    // Verify container contents
    CHECK(std::abs(container[0].real() - 1.0) < 1e-10);
    CHECK(std::abs(container[1].real() - 2.0) < 1e-10);
    CHECK(std::abs(container[2].real() - 3.0) < 1e-10);
    CHECK(std::abs(container[3].real() - 4.0) < 1e-10);
    CHECK(std::abs(container[4].real() - 5.0) < 1e-10);
    CHECK(std::abs(container[5].real() - 6.0) < 1e-10);
  }

  SUBCASE("to_container with column-major indexing") {
    using Tensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
    using Elem = tcapi::tensor_traits<Tensor>::elem_t;
    tcapi::context_handle_t<Tensor> ctx;
    tcapi::create_context(ctx);

    // Create a 2x2 tensor
    Tensor tensor;
    tcapi::allocate(ctx, {2, 2}, tensor);

    // Fill: [[1, 2], [3, 4]]
    tcapi::set_elem(ctx, tensor, {0, 0}, 1.0);
    tcapi::set_elem(ctx, tensor, {0, 1}, 2.0);
    tcapi::set_elem(ctx, tensor, {1, 0}, 3.0);
    tcapi::set_elem(ctx, tensor, {1, 1}, 4.0);

    // Extract to container with column-major indexing
    std::vector<double> container(4);
    auto coors2idx = [](const tcapi::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[1] * 2 + coors[0];  // column-major for 2x2 matrix
    };

    CHECK_NOTHROW(tcapi::to_container(ctx, tensor, container.begin(), coors2idx));

    // Verify: column-major layout should be [1, 3, 2, 4]
    CHECK(std::abs(container[0] - 1.0) < 1e-10);
    CHECK(std::abs(container[1] - 3.0) < 1e-10);
    CHECK(std::abs(container[2] - 2.0) < 1e-10);
    CHECK(std::abs(container[3] - 4.0) < 1e-10);
  }
}

// eig, eigvals and to_cplx on single-precision tensors return complex64
// storage, the element type of cplx_ten_t; complex128 storage read through a
// complex64 tensor turns the eigenvalue 1 into an imaginary part of 1.875.
TEST_CASE_TEMPLATE("CytnxTensor - complex results keep single precision", Tensor,
                   tcapi::CytnxTensor<cytnx::cytnx_float>,
                   tcapi::CytnxTensor<cytnx::cytnx_complex64>) {
  using CplxTensor = typename tcapi::tensor_traits<Tensor>::cplx_ten_t;

  tcapi::context_handle_t<Tensor> ctx;
  tcapi::create_context(ctx);

  Tensor matrix;
  tcapi::eye(ctx, 2, matrix);

  SUBCASE("eig") {
    CplxTensor eigenvals, eigenvecs;
    tcapi::eig(ctx, matrix, 1, eigenvals, eigenvecs);
    CHECK(eigenvals.backend.dtype() == cytnx::Type.ComplexFloat);
    CHECK(eigenvecs.backend.dtype() == cytnx::Type.ComplexFloat);
    for (const double ev : require_diagonal(ctx, eigenvals)) {
      CHECK(ev == doctest::Approx(1.0));
    }
  }

  SUBCASE("eigvals") {
    CplxTensor eigenvals;
    tcapi::eigvals(ctx, matrix, 1, eigenvals);
    CHECK(eigenvals.backend.dtype() == cytnx::Type.ComplexFloat);
    for (cytnx::cytnx_uint64 i = 0; i < 2; ++i) {
      const auto ev = tcapi::get_elem(ctx, eigenvals, {i});
      CHECK(ev.real() == doctest::Approx(1.0));
      CHECK(ev.imag() == doctest::Approx(0.0));
    }
  }

  SUBCASE("to_cplx") {
    CplxTensor cplx;
    tcapi::to_cplx(ctx, matrix, cplx);
    CHECK(cplx.backend.dtype() == cytnx::Type.ComplexFloat);
    const auto one = tcapi::get_elem(ctx, cplx, {0, 0});
    CHECK(one.real() == doctest::Approx(1.0));
    CHECK(one.imag() == doctest::Approx(0.0));
  }

  tcapi::destroy_context(ctx);
}

TEST_CASE("CytnxTensor - Eigenvalue decomposition") {
  using Tensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tcapi::tensor_traits<Tensor>::elem_t;
  using CplxTensor = tcapi::tensor_traits<Tensor>::cplx_ten_t;
  using RealTensor = tcapi::tensor_traits<Tensor>::real_ten_t;

  tcapi::context_handle_t<Tensor> ctx;
  tcapi::create_context(ctx);

  SUBCASE("eigvals - General matrix eigenvalues") {
    // Create a 2x2 non-symmetric matrix
    Tensor matrix;
    tcapi::allocate(ctx, {2, 2}, matrix);

    // Fill with test values: [[1, 2], [3, 4]] (non-symmetric)
    tcapi::set_elem(ctx, matrix, {0, 0}, Elem{1.0, 0.0});
    tcapi::set_elem(ctx, matrix, {0, 1}, Elem{2.0, 0.0});
    tcapi::set_elem(ctx, matrix, {1, 0}, Elem{3.0, 0.0});
    tcapi::set_elem(ctx, matrix, {1, 1}, Elem{4.0, 0.0});

    CplxTensor eigenvalues;

    // GPU version of Cytnx does not support Eig for non-symmetric matrices
    // CPU version supports it
    if (ctx >= cytnx::Device.cuda) {
      // On GPU: expect error for non-symmetric matrix
      CHECK_THROWS(tcapi::eigvals(ctx, matrix, 1, eigenvalues));
    } else {
      // On CPU: should work
      CHECK_NOTHROW(tcapi::eigvals(ctx, matrix, 1, eigenvalues));
      // Should have 2 eigenvalues, as a first-order list unlike eig's
      // diagonal lambda_mat
      CHECK(tcapi::order(ctx, eigenvalues) == 1);
      CHECK(tcapi::shape(ctx, eigenvalues)[0] == 2);
    }
  }

  SUBCASE("eigvalsh - Symmetric matrix eigenvalues") {
    // Create a symmetric 2x2 matrix
    Tensor matrix;
    tcapi::allocate(ctx, {2, 2}, matrix);

    // Fill with symmetric values: [[2, 1], [1, 3]]
    tcapi::set_elem(ctx, matrix, {0, 0}, Elem{2.0, 0.0});
    tcapi::set_elem(ctx, matrix, {0, 1}, Elem{1.0, 0.0});
    tcapi::set_elem(ctx, matrix, {1, 0}, Elem{1.0, 0.0});
    tcapi::set_elem(ctx, matrix, {1, 1}, Elem{3.0, 0.0});

    RealTensor eigenvalues;

    // Perform symmetric eigenvalue calculation
    tcapi::eigvalsh(ctx, matrix, 1, eigenvalues);

    // Should have 2 real eigenvalues, as a first-order list unlike eigh's
    // diagonal lambda_mat
    CHECK(tcapi::order(ctx, eigenvalues) == 1);
    CHECK(tcapi::shape(ctx, eigenvalues)[0] == 2);
  }

  SUBCASE("eig - General matrix eigendecomposition") {
    // Create identity matrix
    Tensor matrix;
    tcapi::eye(ctx, 2, matrix);

    CplxTensor eigenvals, eigenvecs;

    // Perform eigendecomposition
    tcapi::eig(ctx, matrix, 1, eigenvals, eigenvecs);

    // lambda_mat is the {2, 2} diagonal tensor of the eigenvalues
    const auto evals = require_diagonal(ctx, eigenvals);
    REQUIRE(evals.size() == 2);
    CHECK(std::abs(evals[0] - 1.0) < 1e-10);
    CHECK(std::abs(evals[1] - 1.0) < 1e-10);

    // Check eigenvectors
    CHECK(tcapi::order(ctx, eigenvecs) == 2);
    CHECK(tcapi::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tcapi::shape(ctx, eigenvecs)[1] == 2);
  }

  SUBCASE("eigh - Hermitian matrix eigendecomposition") {
    // Create identity matrix
    Tensor matrix;
    tcapi::eye(ctx, 2, matrix);

    RealTensor eigenvals;
    Tensor eigenvecs;

    // Perform hermitian eigendecomposition
    tcapi::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

    // lambda_mat is the {2, 2} diagonal tensor of the eigenvalues
    const auto evals = require_diagonal(ctx, eigenvals);
    REQUIRE(evals.size() == 2);
    CHECK(std::abs(evals[0] - 1.0) < 1e-10);
    CHECK(std::abs(evals[1] - 1.0) < 1e-10);

    // Check eigenvectors
    CHECK(tcapi::order(ctx, eigenvecs) == 2);
    CHECK(tcapi::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tcapi::shape(ctx, eigenvecs)[1] == 2);
  }
}

TEST_CASE("CytnxTensor - I/O Operations") {
  using Tensor = tcapi::CytnxTensor<cytnx::cytnx_double>;
  using Elem = tcapi::tensor_traits<Tensor>::elem_t;
  tcapi::context_handle_t<Tensor> ctx;
  tcapi::create_context(ctx);

  SUBCASE("save and load - double precision") {
    // Create a tensor with known values
    Tensor original;
    tcapi::allocate(ctx, {2, 3}, original);
    tcapi::set_elem(ctx, original, {0, 0}, 1.0);
    tcapi::set_elem(ctx, original, {0, 1}, 2.0);
    tcapi::set_elem(ctx, original, {0, 2}, 3.0);
    tcapi::set_elem(ctx, original, {1, 0}, 4.0);
    tcapi::set_elem(ctx, original, {1, 1}, 5.0);
    tcapi::set_elem(ctx, original, {1, 2}, 6.0);

    // Save to file
    std::string test_file = "test_cytnx_tensor_save_load.cytn";
    tcapi::save(ctx, original, test_file);

    // Load from file
    Tensor loaded;
    tcapi::load(ctx, test_file, loaded);

    // Verify shape
    CHECK(tcapi::order(ctx, loaded) == 2);
    auto loaded_shape = tcapi::shape(ctx, loaded);
    CHECK(loaded_shape[0] == 2);
    CHECK(loaded_shape[1] == 3);

    // Verify values
    CHECK(tcapi::get_elem(ctx, loaded, {0, 0}) == doctest::Approx(1.0));
    CHECK(tcapi::get_elem(ctx, loaded, {0, 1}) == doctest::Approx(2.0));
    CHECK(tcapi::get_elem(ctx, loaded, {0, 2}) == doctest::Approx(3.0));
    CHECK(tcapi::get_elem(ctx, loaded, {1, 0}) == doctest::Approx(4.0));
    CHECK(tcapi::get_elem(ctx, loaded, {1, 1}) == doctest::Approx(5.0));
    CHECK(tcapi::get_elem(ctx, loaded, {1, 2}) == doctest::Approx(6.0));

    // Clean up
    std::filesystem::remove(test_file);
  }

  SUBCASE("save and load - complex tensor") {
    using CTensor = tcapi::CytnxTensor<cytnx::cytnx_complex128>;
    tcapi::context_handle_t<CTensor> cctx;
    tcapi::create_context(cctx);

    // Create a complex tensor
    CTensor original;
    tcapi::allocate(cctx, {2, 2}, original);
    tcapi::set_elem(cctx, original, {0, 0}, cytnx::cytnx_complex128(1.0, 2.0));
    tcapi::set_elem(cctx, original, {0, 1}, cytnx::cytnx_complex128(3.0, 4.0));
    tcapi::set_elem(cctx, original, {1, 0}, cytnx::cytnx_complex128(5.0, 6.0));
    tcapi::set_elem(cctx, original, {1, 1}, cytnx::cytnx_complex128(7.0, 8.0));

    // Save to file
    std::string test_file = "test_cytnx_tensor_save_load_complex.cytn";
    tcapi::save(cctx, original, test_file);

    // Load from file
    CTensor loaded;
    tcapi::load(cctx, test_file, loaded);

    // Verify shape
    CHECK(tcapi::order(cctx, loaded) == 2);
    auto loaded_shape = tcapi::shape(cctx, loaded);
    CHECK(loaded_shape[0] == 2);
    CHECK(loaded_shape[1] == 2);

    // Verify values
    auto val00 = tcapi::get_elem(cctx, loaded, {0, 0});
    CHECK(val00.real() == doctest::Approx(1.0));
    CHECK(val00.imag() == doctest::Approx(2.0));

    auto val01 = tcapi::get_elem(cctx, loaded, {0, 1});
    CHECK(val01.real() == doctest::Approx(3.0));
    CHECK(val01.imag() == doctest::Approx(4.0));

    auto val10 = tcapi::get_elem(cctx, loaded, {1, 0});
    CHECK(val10.real() == doctest::Approx(5.0));
    CHECK(val10.imag() == doctest::Approx(6.0));

    auto val11 = tcapi::get_elem(cctx, loaded, {1, 1});
    CHECK(val11.real() == doctest::Approx(7.0));
    CHECK(val11.imag() == doctest::Approx(8.0));

    // Clean up
    std::filesystem::remove(test_file);
  }

  SUBCASE("save and load - std::string_view and temporaries") {
    // TCAPI V1 declares save's storage a forwarding reference and requires
    // string-like inputs, std::string_view among them, to be accepted. The
    // view below ends before its buffer does.
    const std::string buffer = "test_cytnx_tensor_save_load_view.cytn.not_this_one";
    const std::string_view path = std::string_view(buffer).substr(0, buffer.find(".not"));
    Tensor original;
    tcapi::allocate(ctx, {2}, original);
    tcapi::set_elem(ctx, original, {0}, 1.5);
    tcapi::set_elem(ctx, original, {1}, -2.5);

    // Each save starts without the file, so a save that writes nothing fails.
    auto check_saved = [&](auto&& save_storage) {
      std::filesystem::remove(std::string(path));
      tcapi::save(ctx, original, std::forward<decltype(save_storage)>(save_storage));
      CHECK(std::filesystem::exists(std::string(path)));
      Tensor loaded = tcapi::load<Tensor>(ctx, path);
      CHECK(tcapi::get_elem(ctx, loaded, {0}) == 1.5);
      CHECK(tcapi::get_elem(ctx, loaded, {1}) == -2.5);
    };
    check_saved(path);
    check_saved(std::string_view(path));
    check_saved(std::string(path));

    std::filesystem::remove(std::string(path));
  }
}

// TCAPI V1: close succeeds iff the shapes match and max_i |a_i - b_i| <=
// epsilon, |.| the modulus for complex elements. Two differences of 0.375
// against epsilon 0.5 pass, though their Frobenius norm, about 0.53, does not.
TEST_CASE_TEMPLATE("CytnxTensor - close compares the largest difference", Tensor,
                   tcapi::CytnxTensor<cytnx::cytnx_double>,
                   tcapi::CytnxTensor<cytnx::cytnx_complex64>) {
  using Elem = tcapi::elem_t<Tensor>;
  tcapi::context_handle_t<Tensor> ctx;
  tcapi::create_context(ctx);
  auto zero = tcapi::zeros<Tensor>(ctx, {2});
  auto other = tcapi::zeros<Tensor>(ctx, {2});
  tcapi::set_elem(ctx, other, {0}, Elem(0.375));
  tcapi::set_elem(ctx, other, {1}, Elem(0.375));
  CHECK(tcapi::close(ctx, zero, other, 0.5));
  CHECK_FALSE(tcapi::close(ctx, zero, other, 0.25));

  // A difference equal to epsilon passes.
  tcapi::set_elem(ctx, other, {1}, Elem(0.5));
  CHECK(tcapi::close(ctx, zero, other, 0.5));
  CHECK_FALSE(tcapi::close(ctx, zero, other, 0.4375));

  tcapi::set_elem(ctx, other, {1}, Elem(std::numeric_limits<double>::quiet_NaN()));
  CHECK_FALSE(tcapi::close(ctx, zero, other, 0.5));

  // A complex difference is measured by its modulus: 0.375 + 0.5i has 0.625,
  // above 0.6 though each part is below it, and below 0.7 though their sum
  // is above it.
  if constexpr (!std::is_floating_point_v<Elem>) {
    tcapi::set_elem(ctx, other, {1}, Elem(0.375, 0.5));
    tcapi::set_elem(ctx, other, {0}, Elem(0.0));
    CHECK_FALSE(tcapi::close(ctx, zero, other, 0.6));
    CHECK(tcapi::close(ctx, zero, other, 0.7));
  }
  tcapi::destroy_context(ctx);
}
