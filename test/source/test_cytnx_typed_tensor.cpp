#include <doctest/doctest.h>

#include <complex>
#include <cytnx.hpp>
#include <filesystem>

#include "tci/cytnx_tensor_traits.h"
#include "tci/cytnx_typed_tensor.h"
#include "tci/cytnx_typed_tensor_impl.h"
#include "tci/io_operations.h"
#include "tci/miscellaneous.h"
#include "tci/tensor_linear_algebra.h"

TEST_CASE("CytnxTensor - Type Traits") {
  SUBCASE("Double precision real tensor") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    using Real = tci::tensor_traits<Tensor>::real_t;
    using Cplx = tci::tensor_traits<Tensor>::cplx_t;

    // elem_t should be cytnx::cytnx_double (= double)
    CHECK(std::is_same_v<Elem, cytnx::cytnx_double>);
    CHECK(std::is_same_v<Elem, double>);

    // real_t should equal elem_t for real tensors
    CHECK(std::is_same_v<Real, Elem>);

    // cplx_t should be complex128
    CHECK(std::is_same_v<Cplx, cytnx::cytnx_complex128>);
  }

  SUBCASE("Double precision complex tensor") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    using Real = tci::tensor_traits<Tensor>::real_t;
    using Cplx = tci::tensor_traits<Tensor>::cplx_t;

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
    using Tensor = tci::CytnxTensor<cytnx::cytnx_float>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    using Real = tci::tensor_traits<Tensor>::real_t;
    using Cplx = tci::tensor_traits<Tensor>::cplx_t;

    // elem_t should be cytnx::cytnx_float (= float)
    CHECK(std::is_same_v<Elem, cytnx::cytnx_float>);
    CHECK(std::is_same_v<Elem, float>);

    // real_t should equal elem_t for real tensors
    CHECK(std::is_same_v<Real, Elem>);

    // cplx_t should be complex64
    CHECK(std::is_same_v<Cplx, cytnx::cytnx_complex64>);
  }

  SUBCASE("Single precision complex tensor") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex64>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    using Real = tci::tensor_traits<Tensor>::real_t;
    using Cplx = tci::tensor_traits<Tensor>::cplx_t;

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
  using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tci::tensor_traits<Tensor>::elem_t;

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
    tci::CytnxTensor<cytnx::cytnx_double> t;
    // Should compile and not crash
    CHECK(true);
  }

  SUBCASE("Construction from cytnx::Tensor") {
    cytnx::Tensor backend = cytnx::zeros({2, 2}, cytnx::Type.Double);
    tci::CytnxTensor<cytnx::cytnx_double> t(backend);

    // Backend should be accessible
    CHECK(t.backend.shape().size() == 2);
    CHECK(t.backend.shape()[0] == 2);
    CHECK(t.backend.shape()[1] == 2);
  }

  SUBCASE("Copy construction") {
    cytnx::Tensor backend = cytnx::zeros({3, 3}, cytnx::Type.ComplexDouble);
    tci::CytnxTensor<cytnx::cytnx_complex128> t1(backend);
    tci::CytnxTensor<cytnx::cytnx_complex128> t2 = t1;

    // Both should have valid backends
    CHECK(t1.backend.shape().size() == 2);
    CHECK(t2.backend.shape().size() == 2);
  }

  SUBCASE("Move construction") {
    cytnx::Tensor backend = cytnx::zeros({4, 4}, cytnx::Type.Float);
    tci::CytnxTensor<cytnx::cytnx_float> t1(backend);
    tci::CytnxTensor<cytnx::cytnx_float> t2 = std::move(t1);

    // t2 should have a valid backend
    CHECK(t2.backend.shape().size() == 2);
  }

  SUBCASE("clear function") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Create and fill a tensor
    Tensor tensor;
    tci::fill(ctx, {3, 4}, 5.0, tensor);

    // Verify tensor is initialized
    CHECK(tensor.backend.shape().size() == 2);
    CHECK(tensor.backend.shape()[0] == 3);
    CHECK(tensor.backend.shape()[1] == 4);

    // Clear the tensor
    CHECK_NOTHROW(tci::clear(ctx, tensor));

    // After clear, tensor should be empty (shape size 0)
    CHECK(tensor.backend.shape().size() == 0);
  }

  SUBCASE("assign_from_container with row-major indexing") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Create a 2x3 tensor from std::vector
    std::vector<std::complex<double>> container
        = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}};

    auto coors2idx = [](const tci::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[0] * 3 + coors[1];  // row-major for 2x3 matrix
    };

    tci::shape_t<Tensor> shape = {2, 3};
    Tensor tensor;

    CHECK_NOTHROW(tci::assign_from_container(ctx, shape, container.begin(), coors2idx, tensor));

    // Verify tensor properties
    CHECK(tensor.backend.shape().size() == 2);
    CHECK(tensor.backend.shape()[0] == 2);
    CHECK(tensor.backend.shape()[1] == 3);

    // Verify element values
    Elem elem_00 = tci::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(elem_00.real() - 1.0) < 1e-10);

    Elem elem_01 = tci::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(elem_01.real() - 2.0) < 1e-10);

    Elem elem_10 = tci::get_elem(ctx, tensor, {1, 0});
    CHECK(std::abs(elem_10.real() - 4.0) < 1e-10);

    Elem elem_12 = tci::get_elem(ctx, tensor, {1, 2});
    CHECK(std::abs(elem_12.real() - 6.0) < 1e-10);
  }

  SUBCASE("assign_from_container with column-major indexing") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Create a 2x2 tensor with column-major layout
    std::vector<double> container = {1.0, 3.0, 2.0, 4.0};

    auto coors2idx = [](const tci::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[1] * 2 + coors[0];  // column-major for 2x2 matrix
    };

    tci::shape_t<Tensor> shape = {2, 2};
    Tensor tensor;

    CHECK_NOTHROW(tci::assign_from_container(ctx, shape, container.begin(), coors2idx, tensor));

    // Verify element values with column-major layout
    Elem elem_00 = tci::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(elem_00 - 1.0) < 1e-10);

    Elem elem_01 = tci::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(elem_01 - 2.0) < 1e-10);

    Elem elem_10 = tci::get_elem(ctx, tensor, {1, 0});
    CHECK(std::abs(elem_10 - 3.0) < 1e-10);

    Elem elem_11 = tci::get_elem(ctx, tensor, {1, 1});
    CHECK(std::abs(elem_11 - 4.0) < 1e-10);
  }
}

TEST_CASE("CytnxTensor - TCI API Integration") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tci::tensor_traits<Tensor>::elem_t;
  using ContextHandle = tci::tensor_traits<Tensor>::context_handle_t;

  ContextHandle ctx;
  tci::create_context(ctx);

  SUBCASE("allocate and get_elem") {
    Tensor tensor;
    tci::allocate(ctx, {2, 3}, tensor);

    // Check shape
    CHECK(tensor.backend.shape().size() == 2);
    CHECK(tensor.backend.shape()[0] == 2);
    CHECK(tensor.backend.shape()[1] == 3);

    // Check dtype
    CHECK(tensor.backend.dtype() == cytnx::Type.ComplexDouble);

    // Set and get element
    tensor.backend.at<Elem>({0, 0}) = Elem{3.0, 4.0};
    Elem retrieved = tci::get_elem(ctx, tensor, {0, 0});

    CHECK(retrieved.real() == doctest::Approx(3.0));
    CHECK(retrieved.imag() == doctest::Approx(4.0));
  }

  SUBCASE("Element arithmetic operations") {
    Tensor tensor;
    tci::allocate(ctx, {2, 2}, tensor);

    tensor.backend.at<Elem>({0, 0}) = Elem{1.0, 2.0};
    tensor.backend.at<Elem>({0, 1}) = Elem{3.0, 4.0};

    Elem a = tci::get_elem(ctx, tensor, {0, 0});
    Elem b = tci::get_elem(ctx, tensor, {0, 1});

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
  using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tci::tensor_traits<Tensor>::elem_t;
  using ContextHandle = tci::tensor_traits<Tensor>::context_handle_t;

  ContextHandle ctx;
  tci::create_context(ctx);

  SUBCASE("for_each with std::sqrt") {
    Tensor tensor;
    tci::allocate(ctx, {3, 3}, tensor);

    // Fill with constant values
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{4.0, 0.0};
      }
    }

    // Apply sqrt to each element
    tci::for_each(ctx, tensor, [](Elem& elem) { elem = std::sqrt(elem); });

    // Verify all elements are now 2.0
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tci::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(2.0));
        CHECK(result.imag() == doctest::Approx(0.0));
      }
    }
  }

  SUBCASE("for_each with addition") {
    Tensor tensor;
    tci::allocate(ctx, {2, 2}, tensor);

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
    tci::for_each(ctx, tensor, [addend](Elem& elem) { elem = elem + addend; });

    // Verify all elements are now (4.0, 6.0)
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tci::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(4.0));
        CHECK(result.imag() == doctest::Approx(6.0));
      }
    }
  }

  SUBCASE("for_each with multiplication and division") {
    Tensor tensor;
    tci::allocate(ctx, {2, 3}, tensor);

    // Fill with constant values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{8.0, 0.0};
      }
    }

    // Multiply by 2, then divide by 4
    tci::for_each(ctx, tensor, [](Elem& elem) {
      elem = elem * Elem{2.0, 0.0};
      elem = elem / Elem{4.0, 0.0};
    });

    // Verify all elements are now 4.0
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tci::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(4.0));
        CHECK(result.imag() == doctest::Approx(0.0));
      }
    }
  }

  SUBCASE("for_each with std::abs and conditional") {
    Tensor tensor;
    tci::allocate(ctx, {3, 3}, tensor);

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
    tci::for_each(ctx, tensor, [](Elem& elem) {
      const double magnitude = std::abs(elem);
      if (magnitude > 1e-10) {
        elem = elem / magnitude;
      }
    });

    // Verify all elements have magnitude 1
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tci::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        double magnitude = std::abs(result);
        CHECK(magnitude == doctest::Approx(1.0));
      }
    }
  }

  SUBCASE("for_each with complex arithmetic - inversion") {
    Tensor tensor;
    tci::allocate(ctx, {2, 2}, tensor);

    // Fill with non-zero values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{2.0, 0.0};
      }
    }

    // Invert each element: elem = 1/elem
    tci::for_each(ctx, tensor, [](Elem& elem) { elem = Elem{1.0, 0.0} / elem; });

    // Verify all elements are now 0.5
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tci::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(0.5));
        CHECK(result.imag() == doctest::Approx(0.0));
      }
    }
  }

  SUBCASE("const for_each with arithmetic - sum accumulation") {
    Tensor tensor;
    tci::allocate(ctx, {3, 3}, tensor);

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
    tci::for_each(ctx, static_cast<const Tensor&>(tensor),
                  [&sum](const Elem& elem) { sum = sum + elem; });

    // Verify sum (9 elements * (1.5 + 2.5i) = (13.5 + 22.5i))
    CHECK(sum.real() == doctest::Approx(13.5));
    CHECK(sum.imag() == doctest::Approx(22.5));
  }

  SUBCASE("for_each with std::exp and std::log") {
    Tensor tensor;
    tci::allocate(ctx, {2, 2}, tensor);

    // Fill with constant values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.backend.at<Elem>(
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)})
            = Elem{1.0, 0.0};
      }
    }

    // Apply exp then log (should return to original)
    tci::for_each(ctx, tensor, [](Elem& elem) {
      elem = std::exp(elem);
      elem = std::log(elem);
    });

    // Verify all elements are back to 1.0
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tci::get_elem(
            ctx, tensor,
            {static_cast<cytnx::cytnx_uint64>(i), static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(result.imag() == doctest::Approx(0.0).epsilon(1e-10));
      }
    }
  }
}

TEST_CASE("CytnxTensor - trunc_svd operation") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using RealTensor = tci::CytnxTensor<cytnx::cytnx_double>;
  using Elem = tci::elem_t<Tensor>;
  using Real = tci::real_t<Tensor>;

  tci::context_handle_t<Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Basic trunc_svd functionality") {
    // Create a 4x6 random matrix
    Tensor a;
    tci::allocate(ctx, {4, 6}, a);

    // Fill with simple values for reproducible test
    std::mt19937 rng(42);
    tci::random(ctx, {4, 6}, rng, a);

    // Perform truncated SVD
    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    tci::rank_t<Tensor> num_rows = 1;
    tci::bond_dim_t<Tensor> chi_max = 4;
    Real s_min = 1e-10;

    CHECK_NOTHROW(tci::trunc_svd(ctx, a, num_rows, u, s_diag, v_dag, trunc_err, chi_max, s_min));

    // Verify dimensions
    auto u_shape = tci::shape(ctx, u);
    auto s_shape = tci::shape(ctx, s_diag);
    auto v_shape = tci::shape(ctx, v_dag);

    CHECK(u_shape[0] == 4);
    CHECK(v_shape[1] == 6);
    CHECK(s_shape[0] == u_shape[1]);
    CHECK(s_shape[0] == v_shape[0]);
  }

  SUBCASE("for_each on singular values with sqrt") {
    // Create a simple matrix for SVD
    Tensor a;
    tci::allocate(ctx, {3, 3}, a);

    std::mt19937 rng(123);
    tci::random(ctx, {3, 3}, rng, a);

    // Perform SVD
    Tensor u, v_dag;
    RealTensor s_diag;
    Real trunc_err;

    tci::trunc_svd(ctx, a, 1, u, s_diag, v_dag, trunc_err, tci::bond_dim_t<tci::CytnxTensor<cytnx::cytnx_complex128>>(3), 1e-10);

    // Apply sqrt to singular values
    CHECK_NOTHROW(tci::for_each(ctx, s_diag, [](Real& elem) { elem = std::sqrt(elem); }));

    // Verify all elements are non-negative after sqrt
    auto size = tci::size(ctx, s_diag);
    for (size_t i = 0; i < size; ++i) {
      auto val = tci::get_elem(ctx, s_diag, {static_cast<cytnx::cytnx_uint64>(i)});
      CHECK(val >= 0.0);
    }
  }
}

TEST_CASE("CytnxTensor - Miscellaneous functions") {
  SUBCASE("create_context and destroy_context") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
    tci::context_handle_t<Tensor> ctx;

    // Create context
    CHECK_NOTHROW(tci::create_context(ctx));

    // Context should be initialized to CPU device
    CHECK(ctx == cytnx::Device.cpu);

    // Destroy context (should not throw)
    CHECK_NOTHROW(tci::destroy_context(ctx));
  }

  SUBCASE("to_container with row-major indexing") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Create a 2x3 tensor with known values
    Tensor tensor;
    tci::allocate(ctx, {2, 3}, tensor);

    // Fill with values 1-6
    for (cytnx::cytnx_uint64 i = 0; i < 2; ++i) {
      for (cytnx::cytnx_uint64 j = 0; j < 3; ++j) {
        Elem value{static_cast<double>(i * 3 + j + 1), 0.0};
        tci::set_elem(ctx, tensor, {i, j}, value);
      }
    }

    // Extract to container with row-major indexing
    std::vector<std::complex<double>> container(6);
    auto coors2idx = [](const tci::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[0] * 3 + coors[1];  // row-major for 2x3 matrix
    };

    CHECK_NOTHROW(tci::to_container(ctx, tensor, container.begin(), coors2idx));

    // Verify container contents
    CHECK(std::abs(container[0].real() - 1.0) < 1e-10);
    CHECK(std::abs(container[1].real() - 2.0) < 1e-10);
    CHECK(std::abs(container[2].real() - 3.0) < 1e-10);
    CHECK(std::abs(container[3].real() - 4.0) < 1e-10);
    CHECK(std::abs(container[4].real() - 5.0) < 1e-10);
    CHECK(std::abs(container[5].real() - 6.0) < 1e-10);
  }

  SUBCASE("to_container with column-major indexing") {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
    using Elem = tci::tensor_traits<Tensor>::elem_t;
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Create a 2x2 tensor
    Tensor tensor;
    tci::allocate(ctx, {2, 2}, tensor);

    // Fill: [[1, 2], [3, 4]]
    tci::set_elem(ctx, tensor, {0, 0}, 1.0);
    tci::set_elem(ctx, tensor, {0, 1}, 2.0);
    tci::set_elem(ctx, tensor, {1, 0}, 3.0);
    tci::set_elem(ctx, tensor, {1, 1}, 4.0);

    // Extract to container with column-major indexing
    std::vector<double> container(4);
    auto coors2idx = [](const tci::elem_coors_t<Tensor>& coors) -> std::size_t {
      return coors[1] * 2 + coors[0];  // column-major for 2x2 matrix
    };

    CHECK_NOTHROW(tci::to_container(ctx, tensor, container.begin(), coors2idx));

    // Verify: column-major layout should be [1, 3, 2, 4]
    CHECK(std::abs(container[0] - 1.0) < 1e-10);
    CHECK(std::abs(container[1] - 3.0) < 1e-10);
    CHECK(std::abs(container[2] - 2.0) < 1e-10);
    CHECK(std::abs(container[3] - 4.0) < 1e-10);
  }
}

TEST_CASE("CytnxTensor - Eigenvalue decomposition") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tci::tensor_traits<Tensor>::elem_t;
  using CplxTensor = tci::tensor_traits<Tensor>::cplx_ten_t;
  using RealTensor = tci::tensor_traits<Tensor>::real_ten_t;

  tci::context_handle_t<Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("eigvals - General matrix eigenvalues") {
    // Create a 2x2 matrix
    Tensor matrix;
    tci::allocate(ctx, {2, 2}, matrix);

    // Fill with test values: [[1, 2], [3, 4]]
    tci::set_elem(ctx, matrix, {0, 0}, Elem{1.0, 0.0});
    tci::set_elem(ctx, matrix, {0, 1}, Elem{2.0, 0.0});
    tci::set_elem(ctx, matrix, {1, 0}, Elem{3.0, 0.0});
    tci::set_elem(ctx, matrix, {1, 1}, Elem{4.0, 0.0});

    CplxTensor eigenvalues;

    // Perform eigenvalue calculation
    tci::eigvals(ctx, matrix, 1, eigenvalues);

    // Should have 2 eigenvalues
    CHECK(tci::shape(ctx, eigenvalues)[0] == 2);
  }

  SUBCASE("eigvalsh - Symmetric matrix eigenvalues") {
    // Create a symmetric 2x2 matrix
    Tensor matrix;
    tci::allocate(ctx, {2, 2}, matrix);

    // Fill with symmetric values: [[2, 1], [1, 3]]
    tci::set_elem(ctx, matrix, {0, 0}, Elem{2.0, 0.0});
    tci::set_elem(ctx, matrix, {0, 1}, Elem{1.0, 0.0});
    tci::set_elem(ctx, matrix, {1, 0}, Elem{1.0, 0.0});
    tci::set_elem(ctx, matrix, {1, 1}, Elem{3.0, 0.0});

    RealTensor eigenvalues;

    // Perform symmetric eigenvalue calculation
    tci::eigvalsh(ctx, matrix, 1, eigenvalues);

    // Should have 2 real eigenvalues
    CHECK(tci::shape(ctx, eigenvalues)[0] == 2);
  }

  SUBCASE("eig - General matrix eigendecomposition") {
    // Create identity matrix
    Tensor matrix;
    tci::eye(ctx, 2, matrix);

    CplxTensor eigenvals, eigenvecs;

    // Perform eigendecomposition
    tci::eig(ctx, matrix, 1, eigenvals, eigenvecs);

    // Check eigenvalues
    CHECK(tci::rank(ctx, eigenvals) == 1);
    CHECK(tci::size(ctx, eigenvals) == 2);

    auto eval0 = tci::get_elem(ctx, eigenvals, {0});
    auto eval1 = tci::get_elem(ctx, eigenvals, {1});
    CHECK(std::abs(eval0.real() - 1.0) < 1e-10);
    CHECK(std::abs(eval1.real() - 1.0) < 1e-10);

    // Check eigenvectors
    CHECK(tci::rank(ctx, eigenvecs) == 2);
    CHECK(tci::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tci::shape(ctx, eigenvecs)[1] == 2);
  }

  SUBCASE("eigh - Hermitian matrix eigendecomposition") {
    // Create identity matrix
    Tensor matrix;
    tci::eye(ctx, 2, matrix);

    RealTensor eigenvals;
    Tensor eigenvecs;

    // Perform hermitian eigendecomposition
    tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

    // Check eigenvalues
    CHECK(tci::rank(ctx, eigenvals) == 1);
    CHECK(tci::size(ctx, eigenvals) == 2);

    auto eval0 = tci::get_elem(ctx, eigenvals, {0});
    auto eval1 = tci::get_elem(ctx, eigenvals, {1});
    CHECK(std::abs(eval0 - 1.0) < 1e-10);
    CHECK(std::abs(eval1 - 1.0) < 1e-10);

    // Check eigenvectors
    CHECK(tci::rank(ctx, eigenvecs) == 2);
    CHECK(tci::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tci::shape(ctx, eigenvecs)[1] == 2);
  }
}

TEST_CASE("CytnxTensor - I/O Operations") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
  using Elem = tci::tensor_traits<Tensor>::elem_t;
  tci::context_handle_t<Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("save and load - double precision") {
    // Create a tensor with known values
    Tensor original;
    tci::allocate(ctx, {2, 3}, original);
    tci::set_elem(ctx, original, {0, 0}, 1.0);
    tci::set_elem(ctx, original, {0, 1}, 2.0);
    tci::set_elem(ctx, original, {0, 2}, 3.0);
    tci::set_elem(ctx, original, {1, 0}, 4.0);
    tci::set_elem(ctx, original, {1, 1}, 5.0);
    tci::set_elem(ctx, original, {1, 2}, 6.0);

    // Save to file
    std::string test_file = "test_cytnx_tensor_save_load.cytn";
    tci::save(ctx, original, test_file);

    // Load from file
    Tensor loaded;
    tci::load(ctx, test_file, loaded);

    // Verify shape
    CHECK(tci::rank(ctx, loaded) == 2);
    auto loaded_shape = tci::shape(ctx, loaded);
    CHECK(loaded_shape[0] == 2);
    CHECK(loaded_shape[1] == 3);

    // Verify values
    CHECK(tci::get_elem(ctx, loaded, {0, 0}) == doctest::Approx(1.0));
    CHECK(tci::get_elem(ctx, loaded, {0, 1}) == doctest::Approx(2.0));
    CHECK(tci::get_elem(ctx, loaded, {0, 2}) == doctest::Approx(3.0));
    CHECK(tci::get_elem(ctx, loaded, {1, 0}) == doctest::Approx(4.0));
    CHECK(tci::get_elem(ctx, loaded, {1, 1}) == doctest::Approx(5.0));
    CHECK(tci::get_elem(ctx, loaded, {1, 2}) == doctest::Approx(6.0));

    // Clean up
    std::filesystem::remove(test_file);
  }

  SUBCASE("save and load - complex tensor") {
    using CTensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
    tci::context_handle_t<CTensor> cctx;
    tci::create_context(cctx);

    // Create a complex tensor
    CTensor original;
    tci::allocate(cctx, {2, 2}, original);
    tci::set_elem(cctx, original, {0, 0}, cytnx::cytnx_complex128(1.0, 2.0));
    tci::set_elem(cctx, original, {0, 1}, cytnx::cytnx_complex128(3.0, 4.0));
    tci::set_elem(cctx, original, {1, 0}, cytnx::cytnx_complex128(5.0, 6.0));
    tci::set_elem(cctx, original, {1, 1}, cytnx::cytnx_complex128(7.0, 8.0));

    // Save to file
    std::string test_file = "test_cytnx_tensor_save_load_complex.cytn";
    tci::save(cctx, original, test_file);

    // Load from file
    CTensor loaded;
    tci::load(cctx, test_file, loaded);

    // Verify shape
    CHECK(tci::rank(cctx, loaded) == 2);
    auto loaded_shape = tci::shape(cctx, loaded);
    CHECK(loaded_shape[0] == 2);
    CHECK(loaded_shape[1] == 2);

    // Verify values
    auto val00 = tci::get_elem(cctx, loaded, {0, 0});
    CHECK(val00.real() == doctest::Approx(1.0));
    CHECK(val00.imag() == doctest::Approx(2.0));

    auto val01 = tci::get_elem(cctx, loaded, {0, 1});
    CHECK(val01.real() == doctest::Approx(3.0));
    CHECK(val01.imag() == doctest::Approx(4.0));

    auto val10 = tci::get_elem(cctx, loaded, {1, 0});
    CHECK(val10.real() == doctest::Approx(5.0));
    CHECK(val10.imag() == doctest::Approx(6.0));

    auto val11 = tci::get_elem(cctx, loaded, {1, 1});
    CHECK(val11.real() == doctest::Approx(7.0));
    CHECK(val11.imag() == doctest::Approx(8.0));

    // Clean up
    std::filesystem::remove(test_file);
  }
}