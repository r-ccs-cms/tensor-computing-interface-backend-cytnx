#include <doctest/doctest.h>
#include <tci/tci.h>

#include <complex>
#include <vector>

TEST_CASE("TCI_VERBOSE Environment Variable Support") {
  using namespace tci::debug;

  SUBCASE("Default verbose level (no environment variable)") {
    CHECK(get_verbose_level() >= 0);
    CHECK(get_verbose_level() <= 2);
  }

  SUBCASE("Verbose level functions") {
    int level = get_verbose_level();

    CHECK(is_verbose(0) == (level >= 0));
    CHECK(is_verbose(1) == (level >= 1));
    CHECK(is_verbose(2) == (level >= 2));
    CHECK(is_verbose(3) == false);
  }

  SUBCASE("Timer functionality") {
    CHECK_NOTHROW({
      Timer t("test_timer");
    });
  }

  SUBCASE("Function entry logging") {
    CHECK_NOTHROW(print_function_entry("test_function"));
    CHECK_NOTHROW(print_function_entry("test_function", "additional_info"));
  }
}

TEST_CASE("Integration with TCI_VERBOSE") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using namespace tci;

  context_handle_t<Ten> ctx;
  create_context(ctx);

  SUBCASE("Verbose instrumentation in miscellaneous functions") {
    Ten a = eye<Ten>(ctx, 2);

    CHECK_NOTHROW(show(ctx, a));

    Ten b = eye<Ten>(ctx, 2);
    CHECK(eq(ctx, a, b, 1e-10));

    Ten c;
    context_handle_t<Ten> ctx2;
    create_context(ctx2);
    CHECK_NOTHROW(convert(ctx, a, ctx2, c));

    std::vector<std::complex<double>> container(4);
    auto row_major_map = [](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
      return coors[0] * 2 + coors[1];
    };
    CHECK_NOTHROW(to_container(ctx, a, container.begin(), row_major_map));

    destroy_context(ctx2);
  }

  destroy_context(ctx);
}
