#include <doctest/doctest.h>

#include <complex>
#include <cytnx.hpp>

#include "tci/cytnx_typed_tensor.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/cytnx_typed_tensor_impl.h"

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
}

TEST_CASE("CytnxTensor - TCI API Integration") {
  using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using Elem = tci::tensor_traits<Tensor>::elem_t;
  using ContextHandle = tci::tensor_traits<Tensor>::context_handle_t;

  ContextHandle ctx = -1;  // CPU

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

  ContextHandle ctx = -1;  // CPU

  SUBCASE("for_each with std::sqrt") {
    Tensor tensor;
    tci::allocate(ctx, {3, 3}, tensor);

    // Fill with constant values
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{4.0, 0.0};
      }
    }

    // Apply sqrt to each element
    tci::for_each(ctx, tensor, [](Elem& elem) {
      elem = std::sqrt(elem);
    });

    // Verify all elements are now 2.0
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Elem result = tci::get_elem(ctx, tensor, {static_cast<cytnx::cytnx_uint64>(i),
                                                    static_cast<cytnx::cytnx_uint64>(j)});
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
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{1.0, 2.0};
      }
    }

    // Add constant to each element
    Elem addend{3.0, 4.0};
    tci::for_each(ctx, tensor, [addend](Elem& elem) {
      elem = elem + addend;
    });

    // Verify all elements are now (4.0, 6.0)
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tci::get_elem(ctx, tensor, {static_cast<cytnx::cytnx_uint64>(i),
                                                    static_cast<cytnx::cytnx_uint64>(j)});
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
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{8.0, 0.0};
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
        Elem result = tci::get_elem(ctx, tensor, {static_cast<cytnx::cytnx_uint64>(i),
                                                    static_cast<cytnx::cytnx_uint64>(j)});
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
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{val, val};
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
        Elem result = tci::get_elem(ctx, tensor, {static_cast<cytnx::cytnx_uint64>(i),
                                                    static_cast<cytnx::cytnx_uint64>(j)});
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
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{2.0, 0.0};
      }
    }

    // Invert each element: elem = 1/elem
    tci::for_each(ctx, tensor, [](Elem& elem) {
      elem = Elem{1.0, 0.0} / elem;
    });

    // Verify all elements are now 0.5
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Elem result = tci::get_elem(ctx, tensor, {static_cast<cytnx::cytnx_uint64>(i),
                                                    static_cast<cytnx::cytnx_uint64>(j)});
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
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{1.5, 2.5};
      }
    }

    // Accumulate sum using const for_each
    Elem sum{0.0, 0.0};
    tci::for_each(ctx, static_cast<const Tensor&>(tensor),
                  [&sum](const Elem& elem) {
      sum = sum + elem;
    });

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
        tensor.backend.at<Elem>({static_cast<cytnx::cytnx_uint64>(i),
                                  static_cast<cytnx::cytnx_uint64>(j)}) = Elem{1.0, 0.0};
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
        Elem result = tci::get_elem(ctx, tensor, {static_cast<cytnx::cytnx_uint64>(i),
                                                    static_cast<cytnx::cytnx_uint64>(j)});
        CHECK(result.real() == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(result.imag() == doctest::Approx(0.0).epsilon(1e-10));
      }
    }
  }
}