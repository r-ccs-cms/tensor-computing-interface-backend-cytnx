#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cmath>
#include <cytnx.hpp>
#include <filesystem>

TEST_CASE("tci::save file creation verification") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Save creates file on disk") {
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tci::eye(ctx, 2, tensor);

    std::string filepath = "/tmp/claude/test_tensor.cytnx";
    CHECK_NOTHROW(tci::save(ctx, tensor, filepath));
    CHECK(std::filesystem::exists(filepath));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::load error handling") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Load nonexistent file throws with message") {
    std::string filepath = "/tmp/claude/nonexistent_tensor.cytnx";
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    CHECK_THROWS_WITH(tci::load(ctx, filepath, tensor), doctest::Contains("could not find file"));
  }

  SUBCASE("Out-of-place load nonexistent file throws with message") {
    std::string filepath = "/tmp/claude/nonexistent_tensor.cytnx";
    CHECK_THROWS_WITH(tci::load<tci::CytnxTensor<cytnx::cytnx_complex128>>(ctx, filepath),
                      doctest::Contains("could not find file"));
  }

  tci::destroy_context(ctx);
}
