#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace tci {
namespace debug {

/**
 * @brief Get the current TCI_VERBOSE level from environment variable
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
void print_function_entry(const std::string& function_name, const std::string& tensor_info = "");

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
 * @brief Macro for automatic function timing when TCI_VERBOSE >= 2
 */
#define TCI_TIME_FUNCTION(name) \
    tci::debug::Timer timer(name); \
    if (tci::debug::is_verbose(1)) { \
        tci::debug::print_function_entry(name); \
    }

/**
 * @brief Macro for timing with tensor information
 */
#define TCI_TIME_FUNCTION_WITH_INFO(name, info) \
    tci::debug::Timer timer(name); \
    if (tci::debug::is_verbose(1)) { \
        tci::debug::print_function_entry(name, info); \
    }

} // namespace debug
} // namespace tci