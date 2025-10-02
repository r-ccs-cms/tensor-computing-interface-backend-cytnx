#pragma once

namespace tci {

  // Forward declaration
  template <typename ContextHandleT> void create_context(ContextHandleT& ctx);

  /**
   * @brief Context handle for Cytnx backend
   *
   * This class wraps an integer device ID and restricts direct assignment.
   * Only create_context() is allowed to initialize the value, preventing
   * accidental misuse like `ctx = -1;`
   */
  class CytnxContextHandle {
  private:
    int value_;

    // Only create_context can call this
    void set_value(int v) { value_ = v; }

    // Friend declaration to allow create_context access
    template <typename T>
    friend void create_context(T& ctx);

  public:
    // Default constructor (uninitialized state marker)
    CytnxContextHandle() : value_(-999) {}

    // Delete construction and assignment from int
    CytnxContextHandle(int) = delete;
    CytnxContextHandle& operator=(int) = delete;

    // Allow copy/move operations
    CytnxContextHandle(const CytnxContextHandle&) = default;
    CytnxContextHandle& operator=(const CytnxContextHandle&) = default;
    CytnxContextHandle(CytnxContextHandle&&) = default;
    CytnxContextHandle& operator=(CytnxContextHandle&&) = default;

    // Implicit conversion to int (for Cytnx API compatibility)
    operator int() const { return value_; }

    // Explicit getter
    int get() const { return value_; }
  };

}  // namespace tci
