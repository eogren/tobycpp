// cuda_probe: prove the CUDA runtime is wired up, and print what this GPU is.
//
//   $ cmake --preset clang-cuda && cmake --build --preset clang-cuda
//   $ ./build/clang-cuda/bin/cuda_probe
//
// Built only when -DTOBY_ENABLE_CUDA=ON. It is a build/driver diagnostic, not a
// test: it answers "does libcudart load, does the driver answer, does a
// host->device->host round trip come back with the same bytes" -- the questions
// worth separating out before any engine code is on the stack. Compute is
// deliberately absent; there is no kernel here, so it needs no nvcc.

#include <cstddef>
#include <cstdio>
#include <exception>
// cuda_runtime.h is the whole runtime API surface including the pieces that only
// mean something inside a .cu file. The two headers under it are what host code
// actually needs: driver_types.h for cudaError_t/cudaDeviceProp, and
// cuda_runtime_api.h for the calls. cuda_runtime.h itself stays for its C++
// overloads -- the templated cudaMalloc<T> is what spares us a reinterpret_cast.
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <print>
#include <string_view>
#include <vector>

namespace {

// Every cudaXxx() call returns a status, and ignoring one does not fail loudly --
// it defers the failure to some later, unrelated call, because errors are sticky
// per-thread. Check at the call site.
bool ok(const cudaError_t status, const std::string_view what) {
    if (status == cudaSuccess) {
        return true;
    }
    std::println(stderr, "{} failed: {} ({})", what, cudaGetErrorName(status),
                 cudaGetErrorString(status));
    return false;
}

constexpr double bytes_per_mib = 1024.0 * 1024.0;

void print_device(const int index) {
    cudaDeviceProp prop{};
    if (!ok(cudaGetDeviceProperties(&prop, index), "cudaGetDeviceProperties")) {
        return;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (ok(cudaSetDevice(index), "cudaSetDevice")) {
        ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    }

    std::println("device {}: {}", index, static_cast<const char*>(prop.name));
    std::println("  compute capability : {}.{}", prop.major, prop.minor);
    std::println("  SMs                : {}", prop.multiProcessorCount);
    std::println("  global memory      : {:.0f} MiB ({:.0f} MiB free)",
                 static_cast<double>(total_bytes) / bytes_per_mib,
                 static_cast<double>(free_bytes) / bytes_per_mib);
    std::println("  shared mem / block : {} B", prop.sharedMemPerBlock);
    std::println("  warp size          : {}", prop.warpSize);
    std::println("  unified addressing : {}", prop.unifiedAddressing != 0);
    std::println("  async engines      : {}", prop.asyncEngineCount);
}

// Host -> device -> host with a different destination buffer, so a no-op copy
// cannot pass by accident.
bool round_trip() {
    constexpr std::size_t element_count = 1024;
    constexpr std::size_t byte_count = element_count * sizeof(float);

    std::vector<float> host(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        host[i] = static_cast<float>(i) * 0.5F;
    }

    float* device = nullptr;
    if (!ok(cudaMalloc(&device, byte_count), "cudaMalloc")) {
        return false;
    }

    std::vector<float> back(element_count, 0.0F);
    const bool copied =
        ok(cudaMemcpy(device, host.data(), byte_count, cudaMemcpyHostToDevice), "cudaMemcpy H2D") &&
        ok(cudaMemcpy(back.data(), device, byte_count, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    ok(cudaFree(device), "cudaFree");

    if (!copied) {
        return false;
    }
    if (back != host) {
        std::println(stderr, "round trip: bytes came back different");
        return false;
    }
    std::println("round trip: {} floats ok", element_count);
    return true;
}

int run() {
    int runtime_version = 0;
    int driver_version = 0;
    ok(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    ok(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    std::println("runtime {}.{}, driver supports up to {}.{}", runtime_version / 1000,
                 runtime_version % 1000 / 10, driver_version / 1000, driver_version % 1000 / 10);

    int count = 0;
    if (!ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) {
        return 1;
    }
    if (count == 0) {
        std::println(stderr, "no CUDA devices visible");
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        print_device(i);
    }

    return round_trip() ? 0 : 1;
}

} // namespace

// std::vector and std::println can both throw, and an exception escaping main is
// a std::terminate with no message. Catch here so a failure prints something.
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        // std::fputs, not std::println: the handler must not itself be able to
        // throw, and an allocating formatter can.
        std::fputs("cuda_probe: ", stderr);
        std::fputs(e.what(), stderr);
        std::fputs("\n", stderr);
    } catch (...) {
        std::fputs("cuda_probe: unknown exception\n", stderr);
    }
    return 1;
}
