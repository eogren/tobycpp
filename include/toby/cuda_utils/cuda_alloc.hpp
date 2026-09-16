#ifndef CODA_CUDA_ALLOC_HPP
#define CODA_CUDA_ALLOC_HPP

#include <cstddef>
#include <utility>

namespace toby::cuda {
class CudaAlloc {
public:
    CudaAlloc(CudaAlloc&& other) noexcept
        : addr_(std::exchange(other.addr_, nullptr)), len_(std::exchange(other.len_, 0)) {}

    CudaAlloc& operator=(CudaAlloc&& other) noexcept;
    ~CudaAlloc();

    CudaAlloc(const CudaAlloc&) = delete;
    CudaAlloc& operator=(const CudaAlloc&) = delete;

    [[nodiscard]] void* addr() const { return addr_; }

    [[nodiscard]] std::size_t size() const { return len_; }

    static CudaAlloc anonymous(std::size_t len);

private:
    CudaAlloc(void* addr, std::size_t len) noexcept : addr_(addr), len_(len) {}

    void* addr_ = nullptr;
    std::size_t len_ = 0;
};
} // namespace toby::cuda

#endif
