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
using RealTensor = tci::CytnxTensor<cytnx::cytnx_double>;

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

// --- Construction (allocate, clear, move) ---
TCICT_DOCTEST_CASE("construction", test_allocate_3d, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_allocate_2d, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_allocate_1d, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_clear_basic, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_clear_empty, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_clear_and_reallocate, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_move_inplace, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_move_outofplace, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_move_empty, CplxTensor)
TCICT_DOCTEST_CASE("construction", test_move_preserves_values, CplxTensor)

// --- I/O ---
TCICT_DOCTEST_CASE("io", test_save_load_roundtrip, CplxTensor)
TCICT_DOCTEST_CASE("io", test_load_data_integrity, CplxTensor)

// --- Tensor manipulation ---
TCICT_DOCTEST_CASE("manipulation", test_shrink_inplace, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_shrink_outofplace, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_shrink_complex_values, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_real_extraction, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_imag_extraction, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_real_imag_inplace, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_cplx_conj_inplace, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_cplx_conj_outofplace, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_to_cplx_outofplace, RealTensor)
TCICT_DOCTEST_CASE("manipulation", test_to_cplx_inplace, RealTensor)
TCICT_DOCTEST_CASE("manipulation", test_to_cplx_complex_to_complex, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_doubling, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_summation, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_capture, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_const, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_inversion, CplxTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_with_coors, RealTensor)
TCICT_DOCTEST_CASE("manipulation", test_for_each_with_coors_const, RealTensor)

// --- Linear algebra ---
TCICT_DOCTEST_CASE("linear_algebra", test_norm_identity, CplxTensor)
TCICT_DOCTEST_CASE("linear_algebra", test_linear_combine_uniform, CplxTensor)
TCICT_DOCTEST_CASE("linear_algebra", test_linear_combine_weighted, CplxTensor)
TCICT_DOCTEST_CASE("linear_algebra", test_linear_combine_single, CplxTensor)
TCICT_DOCTEST_CASE("linear_algebra", test_normalize_inplace, CplxTensor)
TCICT_DOCTEST_CASE("linear_algebra", test_normalize_outofplace, CplxTensor)
TCICT_DOCTEST_CASE("linear_algebra", test_normalize_edge_cases, CplxTensor)
