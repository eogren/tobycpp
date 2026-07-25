# Defines an INTERFACE target `toby_simd` that sets the target CPU architecture
# for first-party code, so SIMD (intrinsics / autovectorization / a portable
# wrapper lib) actually gets its full vector width. Link it into first-party
# targets (PRIVATE). Third-party deps deliberately do not get these flags.
#
# Why only -march for now (no std::simd wiring):
#   The C++26 std::simd (<simd>) is not in libc++ yet, and libc++'s
#   std::experimental::simd is a stub -- even `a + b` on native_simd<float>
#   fails to compile (measured 2026-07). So the way to vectorize here is
#   intrinsics / autovectorization / a header-only wrapper (Highway, Kretz
#   std-simd) -- all of which just need the right -march. When libc++ ships a
#   usable <simd>, re-add the std::simd plumbing here (the
#   _LIBCPP_ENABLE_EXPERIMENTAL macro etc.) and drop this note.
#
# -march=native builds for THIS host and is non-portable to older CPUs. Override
# with e.g. -DTOBY_SIMD_ARCH=x86-64-v3 for a portable AVX2 baseline, or
# -DTOBY_SIMD_ARCH= (empty) to disable arch tuning entirely.

add_library(toby_simd INTERFACE)
add_library(toby::simd ALIAS toby_simd)

if(TOBY_SIMD_ARCH)
  target_compile_options(toby_simd INTERFACE
    $<$<CXX_COMPILER_ID:Clang>:-march=${TOBY_SIMD_ARCH}>)
  message(STATUS "toby: SIMD codegen -march=${TOBY_SIMD_ARCH}")
else()
  message(STATUS "toby: SIMD codegen arch tuning disabled")
endif()
