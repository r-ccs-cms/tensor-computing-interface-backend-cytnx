// Explicit specializations for CytnxTensor functions
// These allow user code to use CytnxTensor without including cytnx_typed_tensor_impl.h

#include "tci/cytnx_typed_tensor.h"
#include "tci/cytnx_typed_tensor_impl.h"
#include "tci/tensor_traits.h"
#include <cytnx.hpp>
#include <random>

namespace tci {

  // Explicit template instantiations for commonly used types
  using ComplexElem = cytnx::cytnx_complex128;
  using RealElem = cytnx::cytnx_double;

  // Instantiate size
  template ten_size_t<CytnxTensor<ComplexElem>> size<ComplexElem>(
      context_handle_t<CytnxTensor<ComplexElem>>& ctx,
      const CytnxTensor<ComplexElem>& a);

  template ten_size_t<CytnxTensor<RealElem>> size<RealElem>(
      context_handle_t<CytnxTensor<RealElem>>& ctx,
      const CytnxTensor<RealElem>& a);

  // Instantiate random
  template void random<ComplexElem, std::mt19937&>(
      context_handle_t<CytnxTensor<ComplexElem>>& ctx,
      const shape_t<CytnxTensor<ComplexElem>>& shape,
      std::mt19937& gen,
      CytnxTensor<ComplexElem>& a);

  template void random<RealElem, std::mt19937&>(
      context_handle_t<CytnxTensor<RealElem>>& ctx,
      const shape_t<CytnxTensor<RealElem>>& shape,
      std::mt19937& gen,
      CytnxTensor<RealElem>& a);

  // Instantiate show
  template void show<ComplexElem>(
      context_handle_t<CytnxTensor<ComplexElem>>& ctx,
      const CytnxTensor<ComplexElem>& a);

  template void show<RealElem>(
      context_handle_t<CytnxTensor<RealElem>>& ctx,
      const CytnxTensor<RealElem>& a);

  // Instantiate trunc_svd
  template void trunc_svd<ComplexElem>(
      context_handle_t<CytnxTensor<ComplexElem>>& ctx,
      const CytnxTensor<ComplexElem>& a,
      const rank_t<CytnxTensor<ComplexElem>> num_of_bds_as_row,
      CytnxTensor<ComplexElem>& u,
      real_ten_t<CytnxTensor<ComplexElem>>& s_diag,
      CytnxTensor<ComplexElem>& v_dag,
      real_t<CytnxTensor<ComplexElem>>& trunc_err,
      const bond_dim_t<CytnxTensor<ComplexElem>> chi_max,
      const real_t<CytnxTensor<ComplexElem>> s_min);
  // TODO: This explicit instantiation is redundant since trunc_svd is fully
  // defined in cytnx_typed_tensor_impl.h (Consider removing this
  // to be consistent with svd/qr/lq which only use header-complete implementation.

  // Instantiate assign_from_container
  // Note: Cannot explicitly instantiate template with template template parameters,
  // so we instantiate specific combinations used in tests
  template void assign_from_container<ComplexElem, std::vector<std::complex<double>>::iterator, std::function<std::size_t(const elem_coors_t<CytnxTensor<ComplexElem>>&)>>(
      context_handle_t<CytnxTensor<ComplexElem>>& ctx,
      const shape_t<CytnxTensor<ComplexElem>>& shape,
      std::vector<std::complex<double>>::iterator init_elems_begin,
      std::function<std::size_t(const elem_coors_t<CytnxTensor<ComplexElem>>&)>&& coors2idx,
      CytnxTensor<ComplexElem>& a);

  template void assign_from_container<RealElem, std::vector<double>::iterator, std::function<std::size_t(const elem_coors_t<CytnxTensor<RealElem>>&)>>(
      context_handle_t<CytnxTensor<RealElem>>& ctx,
      const shape_t<CytnxTensor<RealElem>>& shape,
      std::vector<double>::iterator init_elems_begin,
      std::function<std::size_t(const elem_coors_t<CytnxTensor<RealElem>>&)>&& coors2idx,
      CytnxTensor<RealElem>& a);

}  // namespace tci
