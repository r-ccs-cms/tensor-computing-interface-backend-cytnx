#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace tcapi {
  namespace debug {

    /**
     * @brief Get the current TCAPI_VERBOSE level from environment variable
     *
     * @return int Verbose level: 0 = no output, 1 = function calls, 2 = timing info
     */
    int get_verbose_level();

    /**
     * @brief Check if verbose output is enabled
     *
     * @param level Minimum level required for output
     * @return bool True if verbose output should be shown
     */
    bool is_verbose(int level = 1);

    /**
     * @brief Print function entry information
     *
     * @param function_name Name of the function being called
     * @param tensor_info Optional tensor information string
     */
    void print_function_entry(const std::string& function_name,
                              const std::string& tensor_info = "");

    /**
     * @brief RAII timer for measuring function execution time
     */
    class Timer {
    public:
      explicit Timer(const std::string& name);
      ~Timer();

    private:
      std::string name_;
      std::chrono::high_resolution_clock::time_point start_time_;
    };

/**
 * @brief Macro for automatic function timing when TCAPI_VERBOSE >= 2
 */
#define TCAPI_TIME_FUNCTION(name)             \
  tcapi::debug::Timer timer(name);            \
  if (tcapi::debug::is_verbose(1)) {          \
    tcapi::debug::print_function_entry(name); \
  }

/**
 * @brief Macro for timing with tensor information
 */
#define TCAPI_TIME_FUNCTION_WITH_INFO(name, info)   \
  tcapi::debug::Timer timer(name);                  \
  if (tcapi::debug::is_verbose(1)) {                \
    tcapi::debug::print_function_entry(name, info); \
  }

  }  // namespace debug
}  // namespace tcapi