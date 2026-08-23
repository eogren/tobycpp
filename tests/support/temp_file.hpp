#ifndef TESTS_SUPPORT_TEMP_FILE_HPP
#define TESTS_SUPPORT_TEMP_FILE_HPP

// A self-deleting file in the system temp directory, for tests that need a real
// path or fd rather than a buffer.
//
// The standard library has no such thing: temp_directory_path() only names the
// directory, tmpfile() hands back a FILE* with no path, and tmpnam() is racy.
// mkstemp() is the primitive that creates the file atomically, so that is what
// this wraps.
//
// Prefer a checked-in fixture under tests/fixtures/ when the bytes are stable
// and worth reading. Reach for this when the bytes are one-off -- truncated
// headers, bogus lengths, garbage payloads -- and would only clutter the tree.

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace toby::test {

class TempFile {
public:
    explicit TempFile(std::span<const std::byte> contents, std::string_view stem = "toby_test") {
        // mkstemp() rewrites the trailing XXXXXX in place, so the template has
        // to live in modifiable storage.
        auto tmpl =
            (std::filesystem::temp_directory_path() / std::string{stem}).string() + ".XXXXXX";

        const int fd = ::mkstemp(tmpl.data());
        if (fd == -1) {
            throw std::system_error{errno, std::generic_category(), "mkstemp"};
        }
        path_ = tmpl;

        try {
            write_all(fd, contents);
        } catch (...) {
            ::close(fd);
            throw;
        }
        ::close(fd);
    }

    explicit TempFile(std::string_view contents, std::string_view stem = "toby_test")
        : TempFile(std::as_bytes(std::span{contents.data(), contents.size()}), stem) {}

    ~TempFile() {
        if (!path_.empty()) {
            std::error_code ec; // a destructor must not throw; a leftover temp file is not fatal
            std::filesystem::remove(path_, ec);
        }
    }

    TempFile(const TempFile&) = delete;

    TempFile& operator=(const TempFile&) = delete;

    TempFile(TempFile&& other) noexcept : path_(std::exchange(other.path_, {})) {}

    TempFile& operator=(TempFile&& other) noexcept {
        if (this != &other) {
            if (!path_.empty()) {
                std::error_code ec;
                std::filesystem::remove(path_, ec);
            }
            path_ = std::exchange(other.path_, {});
        }
        return *this;
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    // Opens the file read-only. The caller owns the descriptor -- hand it to
    // something that closes it (Storage does) or close it yourself.
    [[nodiscard]] int open_read() const {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): open() is variadic by declaration.
        const int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::system_error{errno, std::generic_category(), "open"};
        }
        return fd;
    }

private:
    static void write_all(int fd, std::span<const std::byte> contents) {
        while (!contents.empty()) {
            const ssize_t written = ::write(fd, contents.data(), contents.size());
            if (written < 0) {
                if (errno == EINTR) {
                    continue; // interrupted before any bytes moved; retry
                }
                throw std::system_error{errno, std::generic_category(), "write"};
            }
            contents = contents.subspan(static_cast<std::size_t>(written));
        }
    }

    std::filesystem::path path_;
};

} // namespace toby::test

#endif
