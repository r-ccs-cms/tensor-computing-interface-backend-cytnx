#include "tci/io_operations.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>

namespace tci {

// Template specializations for I/O operations using Cytnx

template <>
void save(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    std::string &strg
) {
    // Save tensor to file using Cytnx
    a.Save(strg);
}

template <>
void load(
    context_handle_t<cytnx::Tensor> &ctx,
    std::string &&strg,
    cytnx::Tensor &a
) {
    // Load tensor from file using Cytnx
    a.Load(strg);
}

template <>
cytnx::Tensor load(
    context_handle_t<cytnx::Tensor> &ctx,
    std::string &&strg
) {
    cytnx::Tensor result;
    load(ctx, std::move(strg), result);
    return result;
}

} // namespace tci