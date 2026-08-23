#ifndef CORE_SAFETENSORS_FILE_HPP
#define CORE_SAFETENSORS_FILE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace toby::tensors {

/**
    Check the length of the file fd refers to. Will throw
    an exception if fd is not actually a file
*/
[[nodiscard]] std::size_t file_length(int fd);

class Fd {
public:
    explicit Fd(int fd) : fd_(fd) {}

    Fd(Fd&& other) noexcept;
    Fd& operator=(Fd&&) noexcept;

    ~Fd();

    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;

    [[nodiscard]] int fd() const {
        if (fd_ == -1) {
            throw std::runtime_error("Fd no longer valid");
        }

        return fd_;
    }

private:
    int fd_;
};

enum class FileMode : std::uint8_t {
    Read,
};

Fd open_file(const std::filesystem::path& file, FileMode mode = FileMode::Read);
} // namespace toby::tensors

#endif
