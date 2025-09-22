#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace doctest {
namespace detail {

struct TestCase {
  const char* name;
  const char* file;
  int line;
  void (*func)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& failure_count() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(void (*func)(), const char* name, const char* file, int line) {
    registry().push_back({name, file, line, func});
  }
};

inline void report_failure(const char* expr, const char* file, int line, const char* message = nullptr) {
  std::cerr << file << ":" << line << ": CHECK(" << expr << ") failed";
  if (message != nullptr) {
    std::cerr << " - " << message;
  }
  std::cerr << std::endl;
  ++failure_count();
}

inline void check(bool result, const char* expr, const char* file, int line) {
  if (!result) {
    report_failure(expr, file, line);
  }
}

inline void report_exception(const char* expr, const char* file, int line, const char* info) {
  report_failure(expr, file, line, info);
}

inline int run_all_tests() {
  int total = 0;
  for (const auto& test : registry()) {
    ++total;
    try {
      test.func();
    } catch (const std::exception& ex) {
      std::cerr << test.file << ":" << test.line << ": uncaught exception in test '" << test.name
                << "': " << ex.what() << std::endl;
      ++failure_count();
    } catch (...) {
      std::cerr << test.file << ":" << test.line << ": uncaught unknown exception in test '"
                << test.name << "'" << std::endl;
      ++failure_count();
    }
  }

  std::cout << "[doctest-lite] " << total << " test case(s), " << failure_count() << " failure(s)"
            << std::endl;
  return failure_count();
}

}  // namespace detail

class Context {
 public:
  int run() { return detail::run_all_tests(); }
};

}  // namespace doctest

#define DOCTEST_CAT_IMPL(s1, s2) s1##s2
#define DOCTEST_CAT(s1, s2) DOCTEST_CAT_IMPL(s1, s2)

#define DOCTEST_ANON_FUNC DOCTEST_CAT(doctest_test_case_, __LINE__)
#define DOCTEST_ANON_REG DOCTEST_CAT(doctest_registrar_, __LINE__)

#define TEST_CASE(name)                                                                                                  \
  static void DOCTEST_ANON_FUNC();                                                                                       \
  static ::doctest::detail::Registrar DOCTEST_ANON_REG(DOCTEST_ANON_FUNC, name, __FILE__, __LINE__);                     \
  static void DOCTEST_ANON_FUNC()

#define SUBCASE(name) for (int DOCTEST_CAT(_doctest_subcase_, __LINE__) = 0; DOCTEST_CAT(_doctest_subcase_, __LINE__)++ == 0;)

#define CHECK(expr) ::doctest::detail::check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

#define CHECK_NOTHROW(expr)                                                                                              \
  do {                                                                                                                   \
    try {                                                                                                                \
      expr;                                                                                                             \
    } catch (const std::exception& ex) {                                                                                \
      ::doctest::detail::report_exception(#expr, __FILE__, __LINE__, ex.what());                                       \
    } catch (...) {                                                                                                      \
      ::doctest::detail::report_exception(#expr, __FILE__, __LINE__, "unknown exception");                             \
    }                                                                                                                    \
  } while (false)

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
namespace doctest {
inline int run_all_tests() { return detail::run_all_tests(); }
}  // namespace doctest

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return doctest::run_all_tests();
}
#endif

