#pragma once

#include "tci/tensor_traits.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/read_only_getters.h"
#include "tci/construction_destruction.h"
#include "tci/io_operations.h"
#include "tci/tensor_manipulation.h"
#include "tci/tensor_linear_algebra.h"
#include "tci/miscellaneous.h"
#include "tci/debugging.h"

/**
 * @file tci.h
 * @brief Tensor Computing Interface (TCI) - Universal interface for tensor computing
 *
 * This header provides a universal interface for tensor computations that can be
 * implemented by various underlying tensor libraries. The current implementation
 * uses Cytnx as the backend tensor library.
 *
 * The interface is designed using C++ template genericity to abstract different
 * tensor library implementations while providing a consistent API.
 */

namespace tci {
    // All TCI functions and types are defined in their respective headers
    // This main header serves as a convenient include point for all TCI functionality
}