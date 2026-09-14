#ifndef SAFETENSORS_ARENA_HPP
#define SAFETENSORS_ARENA_HPP

#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace toby::tensors {
struct MemcpyInfo {
    const void* src{};
    std::size_t src_offset{};
    std::size_t new_offset{};
    std::size_t size{};
    DeviceType src_device = DeviceType::CPU;
};

class Arena {
public:
    static std::unique_ptr<Arena> alloc_anonymous(std::size_t size, DeviceType device_type,
                                                  std::optional<std::string_view> name = {});

    virtual ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    [[nodiscard]] std::string_view name() const { return name_; }

    /**
     * Return a span representing all allocated memory in this arena. The device type
     * will match the device type of the arena.
     */
    [[nodiscard]] std::span<const std::byte> byte_span() const;

    [[nodiscard]] DeviceType device() const { return device_; }

    /** Allocate a 64-byte aligned chunk of the given size and return a pointer to it.
     * If the arena is too full will throw an exception.
     */
    [[nodiscard]] std::span<std::byte> alloc(std::size_t len);

    /** Copy the given pointer into a 64-byte aligned chunk of this arena and return
     * a pointer to it.
     * If the arena is too full will throw an exception.
     */
    [[nodiscard]] std::span<std::byte> alloc_from_cpu_ptr(std::span<const std::byte> in);

    /**
     * Trigger a set of memcpys into this arena.
     */
    // Source ranges must be valid; destination ranges must already be allocated.
    void bulk_memcpy(const std::vector<MemcpyInfo>& copies);

protected:
    Arena(std::string_view name, DeviceType device, void* base, std::size_t size)
        : name_(name), device_(device), base_(base), size_(size) {}

    Arena(Arena&&) noexcept = default;
    Arena& operator=(Arena&&) noexcept = default;

    void* base() { return base_; }

    [[nodiscard]] std::size_t used() const { return used_; }

    /**
        Copy bytes from the given src & device into `offset` in this arena. Throws if `offset+size`
       points into unallocated memory.
    */
    void memcpy(std::span<const std::byte> src, DeviceType device, std::size_t offset);

private:
    std::string name_;
    DeviceType device_;
    void* base_;
    std::size_t size_;
    std::size_t used_{0};
};
} // namespace toby::tensors

#endif
