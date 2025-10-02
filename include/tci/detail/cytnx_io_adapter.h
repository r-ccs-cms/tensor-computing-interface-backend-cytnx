#pragma once

#include <cytnx.hpp>
#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"

namespace tci {
  template <typename ElemT> class CytnxTensor;
}

namespace tci::detail {

  namespace {

    template <typename Storage> using storage_decay_t = std::decay_t<Storage>;

    template <typename Storage> inline constexpr bool is_path_storage_v
        = std::is_same_v<storage_decay_t<Storage>, std::filesystem::path>
          || std::is_convertible_v<storage_decay_t<Storage>, std::string>;

    template <typename Storage> inline constexpr bool is_input_stream_v
        = std::is_base_of_v<std::istream, storage_decay_t<Storage>>;

    template <typename Storage> inline constexpr bool is_output_stream_v
        = std::is_base_of_v<std::ostream, storage_decay_t<Storage>> && !is_input_stream_v<Storage>;

    inline std::filesystem::path to_path(const std::filesystem::path& path) { return path; }

    inline std::filesystem::path to_path(std::filesystem::path&& path) { return std::move(path); }

    inline std::filesystem::path to_path(const std::string& path) {
      return std::filesystem::path(path);
    }

    inline std::filesystem::path to_path(std::string&& path) {
      return std::filesystem::path(std::move(path));
    }

    inline std::filesystem::path to_path(const char* path) { return std::filesystem::path(path); }

    template <typename Storage> inline std::filesystem::path to_path(Storage&& storage) {
      if constexpr (std::is_convertible_v<Storage, std::string>) {
        return std::filesystem::path(std::string(std::forward<Storage>(storage)));
      } else {
        static_assert(dependent_false_v<Storage>, "Unable to convert storage to filesystem path");
      }
    }

    inline void ensure_parent_exists(const std::filesystem::path& path) {
      const auto parent = path.parent_path();
      if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
          throw std::runtime_error("tci::save failed to create directories: " + ec.message());
        }
      }
    }

    inline void remove_file_silent(const std::filesystem::path& path) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }

    inline void rename_or_copy(const std::filesystem::path& from, const std::filesystem::path& to) {
      std::error_code ec;
      std::filesystem::remove(to, ec);
      ec.clear();
      std::filesystem::rename(from, to, ec);
      if (ec) {
        ec.clear();
        std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
          throw std::runtime_error("tci::save failed to finalize file move: " + ec.message());
        }
        remove_file_silent(from);
      }
    }

    inline std::filesystem::path make_temp_cytn_file() {
      auto base = std::filesystem::temp_directory_path()
                  / ("tci-io-"
                     + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
      base.replace_extension(".cytn");
      return base;
    }

    inline void copy_stream_to_file(std::istream& in, const std::filesystem::path& path) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out.is_open()) {
        throw std::runtime_error("tci::load failed to open temporary file for writing");
      }
      out << in.rdbuf();
      if (!out) {
        throw std::runtime_error("tci::load failed while writing to temporary file");
      }
      out.close();
    }

    inline void copy_file_to_stream(const std::filesystem::path& path, std::ostream& out) {
      std::ifstream in(path, std::ios::binary);
      if (!in.is_open()) {
        throw std::runtime_error("tci::save failed to open temporary file for reading");
      }
      out << in.rdbuf();
      if (!out) {
        throw std::runtime_error("tci::save failed while streaming tensor contents");
      }
    }

    inline void load_from_path(context_handle_t<cytnx::Tensor>& ctx,
                               const std::filesystem::path& path, cytnx::Tensor& a) {
      (void)ctx;
      std::error_code ec;
      if (!std::filesystem::exists(path, ec)) {
        throw std::runtime_error("tci::load could not find file: " + path.string());
      }

      auto loaded = cytnx::Tensor::Load(path.string());
      a = std::move(loaded);
    }

    inline cytnx::Tensor load_from_path(context_handle_t<cytnx::Tensor>& ctx,
                                        const std::filesystem::path& path) {
      cytnx::Tensor result;
      load_from_path(ctx, path, result);
      return result;
    }

    inline void save_to_path(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                             const std::filesystem::path& path) {
      (void)ctx;
      ensure_parent_exists(path);

      if (path.extension() == ".cytn") {
        auto base = path;
        base.replace_extension();
        a.Save(base.string());
      } else {
        a.Save(path.string());
        const auto produced = path.string() + ".cytn";
        rename_or_copy(produced, path);
      }
    }

  }  // namespace

  // Path-like storage (std::string, const char*, std::filesystem::path, ...)
  template <typename Storage>
  struct storage_adapter<cytnx::Tensor, Storage, std::enable_if_t<is_path_storage_v<Storage>>> {
    template <typename StorageLike>
    static void load(context_handle_t<cytnx::Tensor>& ctx, StorageLike&& strg, cytnx::Tensor& a) {
      load_from_path(ctx, to_path(std::forward<StorageLike>(strg)), a);
    }

    template <typename StorageLike>
    static cytnx::Tensor load_out(context_handle_t<cytnx::Tensor>& ctx, StorageLike&& strg) {
      return load_from_path(ctx, to_path(std::forward<StorageLike>(strg)));
    }

    template <typename StorageLike> static void save(context_handle_t<cytnx::Tensor>& ctx,
                                                     const cytnx::Tensor& a, StorageLike& strg) {
      save_to_path(ctx, a, to_path(strg));
    }
  };

  // Input stream storage (std::istream, std::ifstream, std::istringstream, ...)
  template <typename Storage>
  struct storage_adapter<cytnx::Tensor, Storage, std::enable_if_t<is_input_stream_v<Storage>>> {
    template <typename StorageLike>
    static void load(context_handle_t<cytnx::Tensor>& ctx, StorageLike&& strg, cytnx::Tensor& a) {
      auto tmp_path = make_temp_cytn_file();
      copy_stream_to_file(strg, tmp_path);
      try {
        load_from_path(ctx, tmp_path, a);
      } catch (...) {
        remove_file_silent(tmp_path);
        throw;
      }
      remove_file_silent(tmp_path);
    }

    template <typename StorageLike>
    static cytnx::Tensor load_out(context_handle_t<cytnx::Tensor>& ctx, StorageLike&& strg) {
      cytnx::Tensor result;
      load(ctx, std::forward<StorageLike>(strg), result);
      return result;
    }

    template <typename StorageLike>
    static void save(context_handle_t<cytnx::Tensor>&, const cytnx::Tensor&, StorageLike&) {
      static_assert(dependent_false_v<StorageLike>,
                    "tci::save does not support input stream storage types");
    }
  };

  // Output stream storage (std::ostream, std::ofstream, std::ostringstream, ...)
  template <typename Storage>
  struct storage_adapter<cytnx::Tensor, Storage, std::enable_if_t<is_output_stream_v<Storage>>> {
    template <typename StorageLike>
    static void load(context_handle_t<cytnx::Tensor>&, StorageLike&&, cytnx::Tensor&) {
      static_assert(dependent_false_v<StorageLike>,
                    "tci::load does not support output stream storage types");
    }

    template <typename StorageLike>
    static cytnx::Tensor load_out(context_handle_t<cytnx::Tensor>&, StorageLike&&) {
      static_assert(dependent_false_v<StorageLike>,
                    "tci::load does not support output stream storage types");
    }

    template <typename StorageLike> static void save(context_handle_t<cytnx::Tensor>& ctx,
                                                     const cytnx::Tensor& a, StorageLike& strg) {
      auto tmp_path = make_temp_cytn_file();
      save_to_path(ctx, a, tmp_path);
      try {
        copy_file_to_stream(tmp_path, strg);
      } catch (...) {
        remove_file_silent(tmp_path);
        throw;
      }
      remove_file_silent(tmp_path);
    }
  };

  // CytnxTensor storage adapters - thin adapters delegating to cytnx::Tensor backend
  // (Backend Unification Pattern)

  // Path-like storage for CytnxTensor
  template <typename ElemT, typename Storage>
  struct storage_adapter<CytnxTensor<ElemT>, Storage,
                         std::enable_if_t<is_path_storage_v<Storage>>> {
    template <typename StorageLike>
    static void load(context_handle_t<CytnxTensor<ElemT>>& ctx, StorageLike&& strg,
                     CytnxTensor<ElemT>& a) {
      // Delegate to backend (cytnx::Tensor) implementation
      context_handle_t<cytnx::Tensor> backend_ctx = ctx;
      storage_adapter<cytnx::Tensor, Storage>::load(backend_ctx, std::forward<StorageLike>(strg),
                                                     a.backend);
    }

    template <typename StorageLike>
    static CytnxTensor<ElemT> load_out(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                        StorageLike&& strg) {
      CytnxTensor<ElemT> result;
      load(ctx, std::forward<StorageLike>(strg), result);
      return result;
    }

    template <typename StorageLike>
    static void save(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                     StorageLike& strg) {
      // Delegate to backend (cytnx::Tensor) implementation
      context_handle_t<cytnx::Tensor> backend_ctx = ctx;
      storage_adapter<cytnx::Tensor, Storage>::save(backend_ctx, a.backend, strg);
    }
  };

  // Input stream storage for CytnxTensor
  template <typename ElemT, typename Storage>
  struct storage_adapter<CytnxTensor<ElemT>, Storage,
                         std::enable_if_t<is_input_stream_v<Storage>>> {
    template <typename StorageLike>
    static void load(context_handle_t<CytnxTensor<ElemT>>& ctx, StorageLike&& strg,
                     CytnxTensor<ElemT>& a) {
      // Delegate to backend (cytnx::Tensor) implementation
      context_handle_t<cytnx::Tensor> backend_ctx = ctx;
      storage_adapter<cytnx::Tensor, Storage>::load(backend_ctx, std::forward<StorageLike>(strg),
                                                     a.backend);
    }

    template <typename StorageLike>
    static CytnxTensor<ElemT> load_out(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                        StorageLike&& strg) {
      CytnxTensor<ElemT> result;
      load(ctx, std::forward<StorageLike>(strg), result);
      return result;
    }

    template <typename StorageLike>
    static void save(context_handle_t<CytnxTensor<ElemT>>&, const CytnxTensor<ElemT>&,
                     StorageLike&) {
      static_assert(dependent_false_v<StorageLike>,
                    "tci::save does not support input stream storage types");
    }
  };

  // Output stream storage for CytnxTensor
  template <typename ElemT, typename Storage>
  struct storage_adapter<CytnxTensor<ElemT>, Storage,
                         std::enable_if_t<is_output_stream_v<Storage>>> {
    template <typename StorageLike>
    static void load(context_handle_t<CytnxTensor<ElemT>>&, StorageLike&&, CytnxTensor<ElemT>&) {
      static_assert(dependent_false_v<StorageLike>,
                    "tci::load does not support output stream storage types");
    }

    template <typename StorageLike>
    static CytnxTensor<ElemT> load_out(context_handle_t<CytnxTensor<ElemT>>&, StorageLike&&) {
      static_assert(dependent_false_v<StorageLike>,
                    "tci::load does not support output stream storage types");
    }

    template <typename StorageLike>
    static void save(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& a,
                     StorageLike& strg) {
      // Delegate to backend (cytnx::Tensor) implementation
      context_handle_t<cytnx::Tensor> backend_ctx = ctx;
      storage_adapter<cytnx::Tensor, Storage>::save(backend_ctx, a.backend, strg);
    }
  };

}  // namespace tci::detail
