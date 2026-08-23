#include "toby/safetensors/file.hpp"

#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <filesystem>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>

using toby::tensors::FileMode;

namespace {
constexpr int flags_to_mode(FileMode mode) {
    switch (mode) {
    case FileMode::Read:
        return O_RDONLY;
    default:
        throw std::invalid_argument{"Unhandled mode"};
    }
}

toby::tensors::Fd open_file_internal(const std::filesystem::path& file, int mode) {
    const auto* const path = file.c_str();
    const int fd = open(path, mode); // NOLINT(cppcoreguidelines-pro-type-vararg)
    if (fd == -1) {
        auto e = errno;
        throw std::filesystem::filesystem_error{"open", file,
                                                std::error_code{e, std::system_category()}};
    }

    return toby::tensors::Fd{fd};
}
} // namespace

namespace toby::tensors {
Fd::~Fd() {
    if (fd_ != -1) {
        close(fd_);
    }
}

Fd::Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

Fd open_file(const std::filesystem::path& file, FileMode mode) {
    return open_file_internal(file, flags_to_mode(mode));
}

std::size_t file_length(int fd) {
    struct stat info {};

    const int ret = fstat(fd, &info);
    if (ret == -1) {
        throw std::system_error{errno, std::generic_category(), "fstat"};
    }

    if ((info.st_mode & S_IFREG) == 0) {
        throw std::invalid_argument{"fd is not pointing at a file"};
    }

    if (info.st_size < 0) {
        throw std::out_of_range{"file size < 0"};
    }

    return static_cast<std::size_t>(info.st_size);
}
} // namespace toby::tensors
