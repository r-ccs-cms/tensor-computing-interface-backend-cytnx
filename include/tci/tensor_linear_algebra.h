#pragma once

#include <algorithm>
#include <cytnx.hpp>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string_view>

#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"

namespace tci {

  // Linear algebra functions are implemented as CytnxTensor<ElemT> specializations
  // in cytnx_typed_tensor_impl.h. See spec/tci_function_signatures-tensor-linear-algebra.md
  // for API documentation.

  // Implementation details for contract (moved from .cpp for template visibility)
  namespace detail {
    // NCON analysis: determine contraction and permutation from labels
    template <typename TenT> struct NCONAnalysis {
      std::vector<cytnx::cytnx_uint64> contract_axes_a, contract_axes_b;
      std::vector<cytnx::cytnx_uint64> free_axes_a, free_axes_b;
      std::vector<cytnx::cytnx_uint64> output_permutation;

      NCONAnalysis(const List<bond_label_t<TenT>>& bd_labs_a,
                   const List<bond_label_t<TenT>>& bd_labs_b,
                   const List<bond_label_t<TenT>>& bd_labs_c) {
        analyze(bd_labs_a, bd_labs_b, bd_labs_c);
      }

    private:
      // Check for duplicate labels within a single tensor's label list
      void check_duplicate_labels(const List<bond_label_t<TenT>>& labels, const char* tensor_name) {
        std::set<bond_label_t<TenT>> seen;
        for (const auto& label : labels) {
          if (seen.count(label)) {
            std::ostringstream oss;
            oss << "contract: repeated label '" << label << "' within " << tensor_name
                << " (spec v1 prohibits repeating labels within a single operand; "
                << "use tci::trace first if needed)";
            throw std::invalid_argument(oss.str());
          }
          seen.insert(label);
        }
      }

      void analyze(const List<bond_label_t<TenT>>& bd_labs_a,
                   const List<bond_label_t<TenT>>& bd_labs_b,
                   const List<bond_label_t<TenT>>& bd_labs_c) {
        // Check for duplicate labels within each tensor
        // Spec v1 prohibits repeating labels within a single operand
        check_duplicate_labels(bd_labs_a, "first tensor");
        check_duplicate_labels(bd_labs_b, "second tensor");

        // Find contracted indices (appear in both a and b, not in c)
        std::set<bond_label_t<TenT>> labels_b(bd_labs_b.begin(), bd_labs_b.end());
        std::set<bond_label_t<TenT>> labels_c(bd_labs_c.begin(), bd_labs_c.end());

        // Process tensor a: for each contracted axis, find corresponding axis in b
        // This ensures contract_axes_a[i] and contract_axes_b[i] have the same label
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (labels_b.count(label) && !labels_c.count(label)) {
            // This axis will be contracted
            contract_axes_a.push_back(static_cast<cytnx::cytnx_uint64>(i));
            // Find corresponding axis in b with same label
            for (size_t j = 0; j < bd_labs_b.size(); ++j) {
              if (bd_labs_b[j] == label) {
                contract_axes_b.push_back(static_cast<cytnx::cytnx_uint64>(j));
                break;
              }
            }
          } else if (labels_c.count(label)) {
            free_axes_a.push_back(static_cast<cytnx::cytnx_uint64>(i));
          }
        }

        // Process tensor b: find free axes (not contracted)
        std::set<cytnx::cytnx_uint64> contracted_b_set(contract_axes_b.begin(),
                                                       contract_axes_b.end());
        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          if (contracted_b_set.count(i) == 0) {
            auto label = bd_labs_b[i];
            if (labels_c.count(label)) {
              free_axes_b.push_back(static_cast<cytnx::cytnx_uint64>(i));
            }
          }
        }

        // Calculate output permutation to match bd_labs_c order
        calculate_output_permutation(bd_labs_a, bd_labs_b, bd_labs_c);
      }

      void calculate_output_permutation(const List<bond_label_t<TenT>>& bd_labs_a,
                                        const List<bond_label_t<TenT>>& bd_labs_b,
                                        const List<bond_label_t<TenT>>& bd_labs_c) {
        output_permutation.clear();

        if (bd_labs_c.empty()) {
          return;  // No output reordering needed
        }

        // Build list of free axes in natural order (tensor a first, then tensor b)
        std::vector<bond_label_t<TenT>> natural_order;

        // Add free axes from tensor a
        for (size_t i = 0; i < bd_labs_a.size(); ++i) {
          auto label = bd_labs_a[i];
          if (std::find(contract_axes_a.begin(), contract_axes_a.end(), i)
              == contract_axes_a.end()) {
            natural_order.push_back(label);
          }
        }

        // Add free axes from tensor b
        for (size_t i = 0; i < bd_labs_b.size(); ++i) {
          auto label = bd_labs_b[i];
          if (std::find(contract_axes_b.begin(), contract_axes_b.end(), i)
              == contract_axes_b.end()) {
            natural_order.push_back(label);
          }
        }

        // Create permutation: for each position in bd_labs_c, find where it is in natural_order
        output_permutation.resize(bd_labs_c.size());
        for (size_t i = 0; i < bd_labs_c.size(); ++i) {
          auto desired_label = bd_labs_c[i];
          auto it = std::find(natural_order.begin(), natural_order.end(), desired_label);
          if (it != natural_order.end()) {
            size_t natural_pos = std::distance(natural_order.begin(), it);
            output_permutation[i] = static_cast<cytnx::cytnx_uint64>(natural_pos);
          } else {
            throw std::invalid_argument("contract: output label not found in free axes");
          }
        }
      }
    };
  }  // namespace detail

}  // namespace tci
