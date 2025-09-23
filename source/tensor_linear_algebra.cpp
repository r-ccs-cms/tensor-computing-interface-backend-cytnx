#include "tci/tensor_linear_algebra.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>

namespace tci {

// Template specializations for linear algebra functions using Cytnx

template <>
real_t<cytnx::Tensor> norm(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    // Use Cytnx's built-in Norm function for Frobenius norm
    auto norm_result = cytnx::linalg::Norm(a);

    // Extract the scalar value with explicit cast
    if (norm_result.dtype() == cytnx::Type.Double) {
        return static_cast<double>(norm_result.at({0}).real());
    } else if (norm_result.dtype() == cytnx::Type.ComplexDouble) {
        return static_cast<double>(norm_result.at({0}).real()); // Norm should always be real
    } else {
        // Convert to double first
        auto converted = norm_result.astype(cytnx::Type.Double);
        return static_cast<double>(converted.at({0}).real());
    }
}

template <>
void contract(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const List<bond_label_t<cytnx::Tensor>> &bd_labs_a,
    const cytnx::Tensor &b,
    const List<bond_label_t<cytnx::Tensor>> &bd_labs_b,
    cytnx::Tensor &c,
    const List<bond_label_t<cytnx::Tensor>> &bd_labs_c
) {
    // Basic validation
    if (bd_labs_a.size() != a.shape().size() ||
        bd_labs_b.size() != b.shape().size()) {
        throw std::invalid_argument("contract: bond label count must match tensor rank");
    }

    // Convert bond labels to cytnx format for contraction
    std::vector<cytnx::cytnx_int64> cytnx_labs_a, cytnx_labs_b;
    cytnx_labs_a.reserve(bd_labs_a.size());
    cytnx_labs_b.reserve(bd_labs_b.size());

    for (const auto& label : bd_labs_a) {
        cytnx_labs_a.push_back(static_cast<cytnx::cytnx_int64>(label));
    }
    for (const auto& label : bd_labs_b) {
        cytnx_labs_b.push_back(static_cast<cytnx::cytnx_int64>(label));
    }

    // Perform contraction using Cytnx linalg API for tensors
    // Note: Cytnx::Contract mainly works with UniTensor, use basic multiplication for now
    throw std::runtime_error("contract: tensor contraction not yet implemented - Cytnx Contract API needs UniTensor");
}

template <>
void contract(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const std::string_view bd_labs_str_a,
    const cytnx::Tensor &b,
    const std::string_view bd_labs_str_b,
    cytnx::Tensor &c,
    const std::string_view bd_labs_str_c
) {
    // Convert string notation to bond label lists
    List<bond_label_t<cytnx::Tensor>> bd_labs_a, bd_labs_b, bd_labs_c;

    // Parse string labels (each character represents a bond)
    for (char ch : bd_labs_str_a) {
        bd_labs_a.push_back(static_cast<bond_label_t<cytnx::Tensor>>(ch));
    }
    for (char ch : bd_labs_str_b) {
        bd_labs_b.push_back(static_cast<bond_label_t<cytnx::Tensor>>(ch));
    }
    for (char ch : bd_labs_str_c) {
        bd_labs_c.push_back(static_cast<bond_label_t<cytnx::Tensor>>(ch));
    }

    // Use the label version of contract
    contract(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);
}

template <>
void svd(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const rank_t<cytnx::Tensor> num_of_bds_as_row,
    cytnx::Tensor &u,
    real_ten_t<cytnx::Tensor> &s_diag,
    cytnx::Tensor &v_dag
) {
    auto shape = a.shape();

    // Calculate row and column dimensions based on num_of_bds_as_row
    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row && i < shape.size(); ++i) {
        row_dim *= shape[i];
    }
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
        col_dim *= shape[i];
    }

    // Reshape tensor to matrix form for SVD
    cytnx::Tensor matrix = a.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    // Perform SVD using Cytnx
    auto svd_result = cytnx::linalg::Svd(matrix);

    // Extract results
    u = svd_result[0];           // U matrix
    s_diag = svd_result[1];      // Singular values (already real)
    v_dag = svd_result[2];       // V† matrix (already conjugate transpose)

    // Reshape U to match original bond structure
    std::vector<cytnx::cytnx_uint64> u_shape;
    for (cytnx::cytnx_uint64 i = 0; i < num_of_bds_as_row; ++i) {
        u_shape.push_back(shape[i]);
    }
    u_shape.push_back(s_diag.shape()[0]); // Add bond dimension
    u.reshape_(u_shape);

    // Reshape V† to match original bond structure
    std::vector<cytnx::cytnx_uint64> v_shape;
    v_shape.push_back(s_diag.shape()[0]); // Bond dimension first
    for (cytnx::cytnx_uint64 i = num_of_bds_as_row; i < shape.size(); ++i) {
        v_shape.push_back(shape[i]);
    }
    v_dag.reshape_(v_shape);
}

template <>
elem_t<cytnx::Tensor> normalize(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout
) {
    // Calculate norm
    auto original_norm = norm(ctx, inout);

    // Normalize by dividing by norm
    if (original_norm > 0.0) {
        inout = inout / cytnx::cytnx_complex128(original_norm, 0.0);
    }

    return cytnx::cytnx_complex128(original_norm, 0.0);
}

template <>
elem_t<cytnx::Tensor> normalize(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    cytnx::Tensor &out
) {
    out = in.clone();
    return normalize(ctx, out);
}

template <>
void scale(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout,
    const elem_t<cytnx::Tensor> s
) {
    inout = inout * s;
}

template <>
void scale(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    const elem_t<cytnx::Tensor> s,
    cytnx::Tensor &out
) {
    out = in * s;
}

template <>
void linear_combine(
    context_handle_t<cytnx::Tensor> &ctx,
    const List<cytnx::Tensor> &ins,
    cytnx::Tensor &out
) {
    if (ins.empty()) return;

    out = ins[0].clone();
    for (size_t i = 1; i < ins.size(); ++i) {
        out = out + ins[i];
    }
}

template <>
void linear_combine(
    context_handle_t<cytnx::Tensor> &ctx,
    const List<cytnx::Tensor> &ins,
    const List<elem_t<cytnx::Tensor>> &coefs,
    cytnx::Tensor &out
) {
    if (ins.empty() || coefs.empty()) return;

    out = ins[0] * coefs[0];
    for (size_t i = 1; i < std::min(ins.size(), coefs.size()); ++i) {
        out = out + (ins[i] * coefs[i]);
    }
}

template <>
void diag(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout
) {
    auto shape = inout.shape();

    if (shape.size() == 1) {
        // rank-1 → rank-2 diagonal matrix
        auto n = shape[0];
        cytnx::Tensor diag_matrix = cytnx::zeros({n, n}, inout.dtype(), ctx);

        // Set diagonal elements
        for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
            auto val = inout.at({i});
            diag_matrix.at({i, i}) = val;
        }

        inout = std::move(diag_matrix);
    } else if (shape.size() == 2 && shape[0] == shape[1]) {
        // rank-2 square matrix → rank-1 diagonal vector
        auto n = shape[0];
        cytnx::Tensor diag_vector = cytnx::zeros({n}, inout.dtype(), ctx);

        // Extract diagonal elements
        for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
            auto val = inout.at({i, i});
            diag_vector.at({i}) = val;
        }

        inout = std::move(diag_vector);
    } else {
        throw std::invalid_argument("diag: input must be rank-1 vector or square rank-2 matrix");
    }
}

template <>
void diag(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    cytnx::Tensor &out
) {
    out = in.clone();
    diag(ctx, out);
}

template <>
void trace(
    context_handle_t<cytnx::Tensor> &ctx,
    cytnx::Tensor &inout,
    const bond_idx_pairs_t<cytnx::Tensor> &bdidx_pairs
) {
    if (bdidx_pairs.empty()) {
        return; // No trace to perform
    }

    auto shape = inout.shape();

    // Validate bond index pairs
    for (const auto& [idx1, idx2] : bdidx_pairs) {
        if (idx1 >= shape.size() || idx2 >= shape.size()) {
            throw std::invalid_argument("trace: bond index out of range");
        }
        if (shape[idx1] != shape[idx2]) {
            throw std::invalid_argument("trace: bond dimensions must match for tracing");
        }
    }

    // For now, implement simple matrix trace (rank-2 case)
    if (shape.size() == 2 && bdidx_pairs.size() == 1) {
        auto [idx1, idx2] = bdidx_pairs[0];
        if ((idx1 == 0 && idx2 == 1) || (idx1 == 1 && idx2 == 0)) {
            // Standard matrix trace
            elem_t<cytnx::Tensor> trace_sum = 0.0;
            auto n = std::min(shape[0], shape[1]);

            for (cytnx::cytnx_uint64 i = 0; i < n; ++i) {
                auto elem = inout.at({i, i});
                trace_sum += cytnx::cytnx_complex128(
                    static_cast<double>(elem.real()),
                    static_cast<double>(elem.imag())
                );
            }

            // Create scalar tensor with trace result
            inout = cytnx::zeros({}, inout.dtype(), ctx);
            inout.at({}) = trace_sum;
            return;
        }
    }

    // General case: not yet implemented
    throw std::runtime_error("trace: general tensor trace not yet implemented - only 2D matrix trace supported");
}

template <>
void trace(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &in,
    const bond_idx_pairs_t<cytnx::Tensor> &bdidx_pairs,
    cytnx::Tensor &out
) {
    out = in.clone();
    trace(ctx, out, bdidx_pairs);
}

} // namespace tci