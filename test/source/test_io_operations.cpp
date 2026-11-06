#include <doctest/doctest.h>
#include <tcapi/tcapi.h>

#include <cmath>
#include <cytnx.hpp>
#include <filesystem>

TEST_CASE("tcapi::save file creation verification") {
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tcapi::create_context(ctx);

  SUBCASE("Save creates file on disk") {
    tcapi::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tcapi::eye(ctx, 2, tensor);

    std::string filepath = "/tmp/claude/test_tensor.cytnx";
    CHECK_NOTHROW(tcapi::save(ctx, tensor, filepath));
    CHECK(std::filesystem::exists(filepath));
  }

  tcapi::destroy_context(ctx);
}

TEST_CASE("tcapi::load error handling") {
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tcapi::create_context(ctx);

  SUBCASE("Load nonexistent file throws with message") {
    std::string filepath = "/tmp/claude/nonexistent_tensor.cytnx";
    tcapi::CytnxTensor<cytnx::cytnx_complex128> tensor;
    CHECK_THROWS_WITH(tcapi::load(ctx, filepath, tensor), doctest::Contains("could not find file"));
  }

  SUBCASE("Out-of-place load nonexistent file throws with message") {
    std::string filepath = "/tmp/claude/nonexistent_tensor.cytnx";
    CHECK_THROWS_WITH(tcapi::load<tcapi::CytnxTensor<cytnx::cytnx_complex128>>(ctx, filepath),
                      doctest::Contains("could not find file"));
  }

  tcapi::destroy_context(ctx);
}
