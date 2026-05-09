// Wires TCICT conformance tests into doctest via the bulk-registration
// adapter. Each `TCICT_DOCTEST_REGISTER_*` line expands into one `TEST_CASE`
// per applicable conformance test; see the tcict adapter header for the
// suite categorization.
//
// The backend's TCI header (`<tci/tci.h>`) must be included before the
// adapter, because some test templates call non-dependent `tci::` functions
// (e.g., `tci::create_context` in the fixture constructor) that are resolved
// at the adapter's first parsing phase.

#include <doctest/doctest.h>
#define TCI_NO_DEPRECATED_API
#include <tci/tci.h>
#include <cytnx.hpp>

#include <tcict/adapters/doctest.h>

using CytnxRealF = tci::CytnxTensor<cytnx::cytnx_float>;
using CytnxRealD = tci::CytnxTensor<cytnx::cytnx_double>;
using CytnxCplxF = tci::CytnxTensor<cytnx::cytnx_complex64>;
using CytnxCplxD = tci::CytnxTensor<cytnx::cytnx_complex128>;

TCICT_DOCTEST_REGISTER_REAL(float, CytnxRealF)
TCICT_DOCTEST_REGISTER_REAL(double, CytnxRealD)
TCICT_DOCTEST_REGISTER_CPLX(cfloat, CytnxCplxF)
TCICT_DOCTEST_REGISTER_CPLX(cdouble, CytnxCplxD)
