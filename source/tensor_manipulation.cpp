#include "tci/tensor_manipulation.h"

#include <cytnx.hpp>

#include "tci/cytnx_tensor_traits.h"

namespace tci {

  // Template specializations for tensor manipulation functions using Cytnx

  template <> void set_elem(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& a,
                            const elem_coors_t<cytnx::Tensor>& coors,
                            const elem_t<cytnx::Tensor> el) {
    // Convert coordinates to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
      cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }

    // Set element in Cytnx tensor
    a.at(cytnx_coors) = static_cast<cytnx::cytnx_complex128>(el);
  }

  template <> void reshape(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
                           const shape_t<cytnx::Tensor>& new_shape) {
    // Convert shape to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(new_shape.size());
    for (const auto& dim : new_shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    inout.reshape_(cytnx_shape);
  }

  template <> void transpose(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
                             const List<bond_idx_t<cytnx::Tensor>>& new_order) {
    // Convert to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
      cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }

    inout = inout.permute(cytnx_order);
  }

  template <> void reshape(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                           const shape_t<cytnx::Tensor>& new_shape, cytnx::Tensor& out) {
    out = in.clone();
    reshape(ctx, out, new_shape);
  }

  template <> void transpose(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                             const List<bond_idx_t<cytnx::Tensor>>& new_order, cytnx::Tensor& out) {
    // Convert to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
      cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }

    out = in.permute(cytnx_order);
  }

  template <> void cplx_conj(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout) {
    inout = cytnx::linalg::Conj(inout);
  }

  template <> void cplx_conj(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                             cytnx::Tensor& out) {
    out = cytnx::linalg::Conj(in);
  }

  template <> void to_cplx(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                           cplx_ten_t<cytnx::Tensor>& out) {
    if (in.dtype() == cytnx::Type.ComplexDouble || in.dtype() == cytnx::Type.ComplexFloat) {
      // Already complex, just copy
      out = in.clone();
    } else {
      // Convert real to complex
      out = in.astype(cytnx::Type.ComplexDouble);
    }
  }

  template <>
  cplx_ten_t<cytnx::Tensor> to_cplx(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in) {
    cplx_ten_t<cytnx::Tensor> result;
    to_cplx(ctx, in, result);
    return result;
  }

  template <> void real(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                        real_ten_t<cytnx::Tensor>& out) {
    if (in.dtype() == cytnx::Type.ComplexDouble || in.dtype() == cytnx::Type.ComplexFloat) {
      // Extract real part from complex tensor
      auto temp = in.clone();
      out = temp.real();
    } else {
      // Already real, just copy
      out = in.clone();
    }
  }

  template <>
  real_ten_t<cytnx::Tensor> real(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in) {
    real_ten_t<cytnx::Tensor> result;
    real(ctx, in, result);
    return result;
  }

  template <> void imag(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                        real_ten_t<cytnx::Tensor>& out) {
    if (in.dtype() == cytnx::Type.ComplexDouble || in.dtype() == cytnx::Type.ComplexFloat) {
      // Extract imaginary part from complex tensor
      auto temp = in.clone();
      out = temp.imag();
    } else {
      // Real tensor, imaginary part is zero
      out = cytnx::zeros(in.shape(), cytnx::Type.Double, ctx);
    }
  }

  template <>
  real_ten_t<cytnx::Tensor> imag(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in) {
    real_ten_t<cytnx::Tensor> result;
    imag(ctx, in, result);
    return result;
  }

  template <> void concatenate(context_handle_t<cytnx::Tensor>& ctx, const List<cytnx::Tensor>& ins,
                               const bond_idx_t<cytnx::Tensor> concat_bdidx, cytnx::Tensor& out) {
    if (ins.empty()) {
      throw std::invalid_argument("Cannot concatenate empty list of tensors");
    }

    // For now, implement basic concatenation for 2D tensors
    if (concat_bdidx == 0) {
      // Vertical stacking
      std::vector<cytnx::Tensor> cytnx_tensors(ins.begin(), ins.end());
      out = cytnx::algo::Vstack(cytnx_tensors);
    } else if (concat_bdidx == 1) {
      // Horizontal stacking
      std::vector<cytnx::Tensor> cytnx_tensors(ins.begin(), ins.end());
      out = cytnx::algo::Hstack(cytnx_tensors);
    } else {
      throw std::runtime_error("General N-dimensional concatenation not yet implemented");
    }
  }

  template <> void stack(context_handle_t<cytnx::Tensor>& ctx, const List<cytnx::Tensor>& ins,
                         const bond_idx_t<cytnx::Tensor> stack_bdidx, cytnx::Tensor& out) {
    if (ins.empty()) {
      throw std::invalid_argument("Cannot stack empty list of tensors");
    }

    // Implement stacking by creating a new dimension at stack_bdidx
    // First verify all tensors have the same shape
    const auto& first_shape = ins[0].shape();
    for (size_t i = 1; i < ins.size(); ++i) {
      if (ins[i].shape() != first_shape) {
        throw std::invalid_argument("All tensors must have the same shape for stacking");
      }
    }

    // Create new shape with additional dimension for stacking
    std::vector<cytnx::cytnx_uint64> new_shape;
    for (size_t i = 0; i < first_shape.size(); ++i) {
      if (i == static_cast<size_t>(stack_bdidx)) {
        new_shape.push_back(ins.size()); // Number of tensors to stack
      }
      new_shape.push_back(first_shape[i]);
    }
    // Handle case where stack_bdidx is at the end
    if (static_cast<size_t>(stack_bdidx) >= first_shape.size()) {
      new_shape.push_back(ins.size());
    }

    // Create output tensor
    out = cytnx::zeros(new_shape, ins[0].dtype(), ins[0].device());

    // Copy data from each tensor
    for (size_t tensor_idx = 0; tensor_idx < ins.size(); ++tensor_idx) {
      // Create index for where to place this tensor in the stacked result
      std::vector<cytnx::Accessor> accessors;
      size_t dim_idx = 0;

      for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i == static_cast<size_t>(stack_bdidx)) {
          accessors.push_back(cytnx::Accessor(static_cast<cytnx::cytnx_int64>(tensor_idx)));
        } else {
          accessors.push_back(cytnx::Accessor::all());
          dim_idx++;
        }
      }

      // Copy the tensor data to the appropriate slice
      out.get(accessors) = ins[tensor_idx];
    }
  }

  // Note: for_each functions require template specialization for specific function types
  // For now, implement basic versions that work with lambda functions

  template <> void for_each<cytnx::Tensor, std::function<void(elem_t<cytnx::Tensor>&)>>(
      context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
      std::function<void(elem_t<cytnx::Tensor>&)>&& f) {
    auto& storage = inout.storage();
    const auto total = storage.size();

    for (cytnx::cytnx_uint64 idx = 0; idx < total; ++idx) {
      auto& elem = storage.at<elem_t<cytnx::Tensor>>(idx);
      f(elem);
    }
  }

  template <> void for_each<cytnx::Tensor, std::function<void(const elem_t<cytnx::Tensor>&)>>(
      context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
      std::function<void(const elem_t<cytnx::Tensor>&)>&& f) {
    const auto& storage = in.storage();
    const auto total = storage.size();

    for (cytnx::cytnx_uint64 idx = 0; idx < total; ++idx) {
      const auto& elem = storage.at<elem_t<cytnx::Tensor>>(idx);
      f(elem);
    }
  }

  // Helper functions for complex operations
  namespace {

    void copy_original_data_recursive(const cytnx::Tensor& src, cytnx::Tensor& dst, std::size_t dim,
                                      std::vector<cytnx::cytnx_uint64> current_coords,
                                      const std::vector<cytnx::cytnx_uint64>& original_shape);

    void extract_elements_recursive(
        const cytnx::Tensor& src, cytnx::Tensor& dst, std::size_t dim,
        std::vector<cytnx::cytnx_uint64> src_coords, std::vector<cytnx::cytnx_uint64> dst_coords,
        const List<Pair<elem_coor_t<cytnx::Tensor>, elem_coor_t<cytnx::Tensor>>>& coor_pairs);

    void replace_elements_recursive(cytnx::Tensor& main_tensor, const cytnx::Tensor& sub_tensor,
                                    std::size_t dim, const elem_coors_t<cytnx::Tensor>& begin_pt,
                                    std::vector<cytnx::cytnx_uint64>& sub_coords,
                                    const std::vector<cytnx::cytnx_uint64>& sub_shape);


  }  // namespace

  // Advanced tensor manipulation functions

  template <> void expand(
      context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
      const Map<bond_idx_t<cytnx::Tensor>, bond_dim_t<cytnx::Tensor>>& bond_idx_increment_map) {
    auto original_shape = inout.shape();
    std::vector<cytnx::cytnx_uint64> new_shape(original_shape.begin(), original_shape.end());

    // Apply increments to shape
    for (const auto& [bond_idx, increment] : bond_idx_increment_map) {
      if (bond_idx >= new_shape.size()) {
        throw std::invalid_argument("Bond index out of range");
      }
      new_shape[bond_idx] += increment;
    }

    // Create new tensor with expanded shape, initialized to zero
    cytnx::Tensor expanded = cytnx::zeros(new_shape, inout.dtype(), ctx);

    // Copy original data to the beginning of each expanded dimension
    auto original_coords = original_shape;
    copy_original_data_recursive(inout, expanded, 0, {}, original_coords);

    inout = std::move(expanded);
  }

  template <> void expand(
      context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
      const Map<bond_idx_t<cytnx::Tensor>, bond_dim_t<cytnx::Tensor>>& bond_idx_increment_map,
      cytnx::Tensor& out) {
    out = in.clone();
    expand(ctx, out, bond_idx_increment_map);
  }

  template <> void extract_sub(
      context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
      const List<Pair<elem_coor_t<cytnx::Tensor>, elem_coor_t<cytnx::Tensor>>>& coor_pairs) {
    auto original_shape = inout.shape();

    if (coor_pairs.size() != original_shape.size()) {
      throw std::invalid_argument("Number of coordinate pairs must match tensor rank");
    }

    // Calculate new shape
    std::vector<cytnx::cytnx_uint64> new_shape;
    for (std::size_t i = 0; i < coor_pairs.size(); ++i) {
      auto [start, end] = coor_pairs[i];
      if (start >= end || end > original_shape[i]) {
        throw std::invalid_argument("Invalid coordinate range");
      }
      new_shape.push_back(end - start);
    }

    // Create result tensor
    cytnx::Tensor result = cytnx::zeros(new_shape, inout.dtype(), ctx);

    // Extract sub-tensor by copying elements
    extract_elements_recursive(inout, result, 0, {}, {}, coor_pairs);

    inout = std::move(result);
  }

  template <> void extract_sub(
      context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
      const List<Pair<elem_coor_t<cytnx::Tensor>, elem_coor_t<cytnx::Tensor>>>& coor_pairs,
      cytnx::Tensor& out) {
    out = in.clone();
    extract_sub(ctx, out, coor_pairs);
  }

  template <>
  void shrink(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
              const bond_idx_elem_coor_pair_map<cytnx::Tensor>& bd_idx_el_coor_pair_map) {
    auto original_shape = inout.shape();

    // Build coordinate pairs list from the map
    List<Pair<elem_coor_t<cytnx::Tensor>, elem_coor_t<cytnx::Tensor>>> coor_pairs;
    coor_pairs.resize(original_shape.size());

    // Initialize with full ranges
    for (std::size_t i = 0; i < original_shape.size(); ++i) {
      coor_pairs[i] = {0, original_shape[i]};
    }

    // Apply shrinking ranges from the map
    for (const auto& [bond_idx, coor_pair] : bd_idx_el_coor_pair_map) {
      if (bond_idx >= original_shape.size()) {
        throw std::invalid_argument("Bond index out of range");
      }
      coor_pairs[bond_idx] = coor_pair;
    }

    // Use extract_sub to perform the shrinking
    extract_sub(ctx, inout, coor_pairs);
  }

  template <> void shrink(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                          const bond_idx_elem_coor_pair_map<cytnx::Tensor>& bd_idx_el_coor_pair_map,
                          cytnx::Tensor& out) {
    out = in.clone();
    shrink(ctx, out, bd_idx_el_coor_pair_map);
  }

  template <> void replace_sub(context_handle_t<cytnx::Tensor>& ctx, cytnx::Tensor& inout,
                               const cytnx::Tensor& sub,
                               const elem_coors_t<cytnx::Tensor>& begin_pt) {
    auto main_shape = inout.shape();
    auto sub_shape = sub.shape();

    if (begin_pt.size() != main_shape.size() || sub_shape.size() != main_shape.size()) {
      throw std::invalid_argument("Dimension mismatch");
    }

    // Check bounds
    for (std::size_t i = 0; i < begin_pt.size(); ++i) {
      if (begin_pt[i] + sub_shape[i] > main_shape[i]) {
        throw std::invalid_argument("Sub-tensor exceeds bounds");
      }
    }

    // Replace elements recursively
    std::vector<cytnx::cytnx_uint64> sub_coords(sub_shape.size(), 0);
    replace_elements_recursive(inout, sub, 0, begin_pt, sub_coords, sub_shape);
  }

  template <> void replace_sub(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& in,
                               const cytnx::Tensor& sub,
                               const elem_coors_t<cytnx::Tensor>& begin_pt, cytnx::Tensor& out) {
    out = in.clone();
    replace_sub(ctx, out, sub, begin_pt);
  }


  // Implementation of helper functions
  namespace {

    void copy_original_data_recursive(const cytnx::Tensor& src, cytnx::Tensor& dst, std::size_t dim,
                                      std::vector<cytnx::cytnx_uint64> current_coords,
                                      const std::vector<cytnx::cytnx_uint64>& original_shape) {
      if (dim == original_shape.size()) {
        // Base case: copy element
        auto src_elem = src.at(current_coords);
        dst.at(current_coords) = src_elem;
        return;
      }

      for (cytnx::cytnx_uint64 i = 0; i < original_shape[dim]; ++i) {
        current_coords.push_back(i);
        copy_original_data_recursive(src, dst, dim + 1, current_coords, original_shape);
        current_coords.pop_back();
      }
    }

    void extract_elements_recursive(
        const cytnx::Tensor& src, cytnx::Tensor& dst, std::size_t dim,
        std::vector<cytnx::cytnx_uint64> src_coords, std::vector<cytnx::cytnx_uint64> dst_coords,
        const List<Pair<elem_coor_t<cytnx::Tensor>, elem_coor_t<cytnx::Tensor>>>& coor_pairs) {
      if (dim == coor_pairs.size()) {
        // Base case: copy element
        auto elem = src.at(src_coords);
        dst.at(dst_coords) = elem;
        return;
      }

      auto [start, end] = coor_pairs[dim];
      for (cytnx::cytnx_uint64 i = start; i < end; ++i) {
        src_coords.push_back(i);
        dst_coords.push_back(i - start);
        extract_elements_recursive(src, dst, dim + 1, src_coords, dst_coords, coor_pairs);
        src_coords.pop_back();
        dst_coords.pop_back();
      }
    }


    void replace_elements_recursive(cytnx::Tensor& main_tensor, const cytnx::Tensor& sub_tensor,
                                    std::size_t dim, const elem_coors_t<cytnx::Tensor>& begin_pt,
                                    std::vector<cytnx::cytnx_uint64>& sub_coords,
                                    const std::vector<cytnx::cytnx_uint64>& sub_shape) {
      if (dim == sub_shape.size()) {
        // Base case: copy element from sub to main
        std::vector<cytnx::cytnx_uint64> main_coords;
        for (std::size_t i = 0; i < begin_pt.size(); ++i) {
          main_coords.push_back(begin_pt[i] + sub_coords[i]);
        }
        auto elem = sub_tensor.at(sub_coords);
        main_tensor.at(main_coords) = elem;
        return;
      }

      for (cytnx::cytnx_uint64 i = 0; i < sub_shape[dim]; ++i) {
        sub_coords[dim] = i;
        replace_elements_recursive(main_tensor, sub_tensor, dim + 1, begin_pt, sub_coords,
                                   sub_shape);
      }
    }

  }  // anonymous namespace

}  // namespace tci