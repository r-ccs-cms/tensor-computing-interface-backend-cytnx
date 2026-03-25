#include <doctest/doctest.h>
#include <tci/tci.h>
#include <tcict/tcict.h>

// Bridge macro: wraps a tcict conformance test function as a doctest TEST_CASE.
// If the conformance test throws tcict::assertion_error, doctest reports it as FAIL.
#define TCICT_DOCTEST_CASE(category, test_func, TenT)                                              \
  TEST_CASE("TCICT: " category " - " #test_func) {                                                \
    tcict::tci_test_fixture<TenT> fix;                                                             \
    tcict::tests::test_func<TenT>(fix);                                                            \
  }

// Conformance test registrations will be added here as tests are migrated.
