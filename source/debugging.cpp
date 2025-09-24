#include "tci/debugging.h"
#include <cstdlib>
#include <iomanip>

namespace tci {
namespace debug {

int get_verbose_level() {
    static int verbose_level = -1;  // Cache the result

    if (verbose_level == -1) {
        const char* env_var = std::getenv("TCI_VERBOSE");
        if (env_var) {
            verbose_level = std::atoi(env_var);
            // Clamp to valid range [0, 2]
            if (verbose_level < 0) verbose_level = 0;
            if (verbose_level > 2) verbose_level = 2;
        } else {
            verbose_level = 0;  // Default: no verbose output
        }
    }

    return verbose_level;
}

bool is_verbose(int level) {
    return get_verbose_level() >= level;
}

void print_function_entry(const std::string& function_name, const std::string& tensor_info) {
    if (!is_verbose(1)) return;

    std::cout << "[TCI] " << function_name;
    if (!tensor_info.empty()) {
        std::cout << " - " << tensor_info;
    }
    std::cout << std::endl;
}

Timer::Timer(const std::string& name)
    : name_(name), start_time_(std::chrono::high_resolution_clock::now()) {
}

Timer::~Timer() {
    if (is_verbose(2)) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);

        std::cout << "[TCI] " << name_ << " took "
                  << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms" << std::endl;
    }
}

} // namespace debug
} // namespace tci