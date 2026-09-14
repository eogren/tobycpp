#ifndef SAFETENSORS_TENSOR_TYPES_HPP
#define SAFETENSORS_TENSOR_TYPES_HPP
#include <cstdint>

namespace toby::tensors {
enum class DeviceType : std::uint8_t { CPU, GPU };
enum class DataType : std::uint8_t { F32, U16 };
} // namespace toby::tensors
#endif
