#include "cpu_arena.hpp"

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/file.hpp"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <sys/mman.h>
#include <system_error>
#include <vector>

namespace {
constexpr int file_mode_to_mode(toby::tensors::FileMode mode) {
    switch (mode) {
    case toby::tensors::FileMode::Read:
        return PROT_READ;
    }
}
} // namespace

namespace toby::tensors::detail {
ScopedMapping ScopedMapping::anonymous(size_t len) {
    return ScopedMapping{-1, PROT_READ | PROT_WRITE, len};
}

ScopedMapping ScopedMapping::from_fd(int fd, FileMode mode) {
    return ScopedMapping{fd, file_mode_to_mode(mode), file_length(fd)};
};

ScopedMapping::ScopedMapping(int fd, int prot, std::size_t len) : len_(len) {
    if (len == 0) {
        return;
    }

    int flags = MAP_PRIVATE;
    if (fd == -1) {
        flags |= MAP_ANONYMOUS;
    }

    void* data = mmap(nullptr, len, prot, flags, fd, 0);
    if (data == MAP_FAILED) {
        throw std::system_error{errno, std::generic_category(), "mmap"};
    }

    addr_ = data;
}

ScopedMapping::~ScopedMapping() {
    if (addr_ != nullptr) {
        munmap(addr_, len_);
    }
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
void CpuArena::bulk_memcpy(const std::vector<MemcpyInfo>& copies) {
    for (const auto& copy : copies) {
        auto* dest = static_cast<std::byte*>(base()) + copy.new_offset;
        const auto* src = static_cast<const std::byte*>(copy.src) + copy.src_offset;

        std::memcpy(dest, src, copy.size);
    }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
} // namespace toby::tensors::detail
