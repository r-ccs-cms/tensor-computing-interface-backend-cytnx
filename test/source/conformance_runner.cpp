#include <doctest/doctest.h>
#include <tci/tci.h>
#include <tcict/tcict.h>

#include <cytnx.hpp>

// Bridge macro: wraps a tcict conformance test function as a doctest TEST_CASE.
// If the conformance test throws tcict::assertion_error, doctest reports it as FAIL.
#define TCICT_DOCTEST_CASE(category, test_func, TenT)                                              \
  TEST_CASE("TCICT: " category " - " #test_func) {                                                \
    tcict::tci_test_fixture<TenT> fix;                                                             \
    tcict::tests::test_func<TenT>(fix);                                                            \
  }

using CplxTensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// --- Construction ---
TCICT_DOCTEST_CASE("construction", test_zeros, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_eye, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_random_inplace, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_random_outofplace, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_copy_inplace, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_copy_outofplace, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_copy_independence, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_copy_single_element, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_copy_large, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_assign_from_range_row_major, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_assign_from_range_column_major, CplxTensor)

// --- Read-only getters ---
TCICT_DOCTEST_CASE("getters", test_order, CplxTensor)
TCICT_DOCTEST_CASE("getters", test_shape, CplxTensor)
TCICT_DOCTEST_CASE("getters", test_size, CplxTensor)
TCICT_DOCTEST_CASE("getters", test_size_bytes, CplxTensor)
TCICT_DOCTEST_CASE("getters", test_set_get_elem, CplxTensor)
TCICT_DOCTEST_CASE("getters", test_size_bytes_2x2, CplxTensor)

// --- Miscellaneous ---
TCICT_DOCTEST_CASE("miscellaneous", test_close_identical, CplxTensor)
TCICT_DOCTEST_CASE("miscellaneous", test_close_different, CplxTensor)

// --- Linear algebra ---
TCICT_DOCTEST_CASE("linear_algebra", test_norm_identity, CplxTensor)
