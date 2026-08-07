cmake_minimum_required(VERSION 3.25)

project(
  toby
  VERSION 0.1.0
  DESCRIPTION "A C++ inference server, built from scratch for learning."
  LANGUAGES
    CXX
)

# ---------------------------------------------------------------------------
# Guardrails: in-source builds are almost always a mistake.
# ---------------------------------------------------------------------------
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
    "In-source builds are not allowed. Use a preset, e.g. "
    "`cmake --preset clang-debug` which builds into ./build/."
  )
endif()

# ---------------------------------------------------------------------------
# Language standard: C++23, required, no compiler extensions.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json (clang-tidy, editors, tooling need it).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a Debug build if the user (or a preset) didn't specify one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# Put all binaries in one predictable place.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(TOBY_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(TOBY_ENABLE_CLANG_TIDY "Run clang-tidy during the build" OFF)
option(TOBY_BUILD_TESTS "Build the unit test suite" ON)
option(TOBY_BUILD_TOOLS "Build developer tools in tools/" ON)

# SIMD codegen: set the target CPU arch so vectorization gets full width.
# (std::simd/experimental::simd is NOT wired -- unusable on libc++ 19; see cmake/Simd.cmake.)
set(
  TOBY_SIMD_ARCH
  "native"
  CACHE STRING
  "Target CPU arch for SIMD codegen via -march=. 'native' = this build host; empty disables."
)

# Sanitizers (mutually compose except TSan, which conflicts with ASan).
option(TOBY_SANITIZE_ADDRESS "Enable AddressSanitizer" OFF)
option(TOBY_SANITIZE_UNDEFINED "Enable UndefinedBehaviorSanitizer" OFF)
option(TOBY_SANITIZE_THREAD "Enable ThreadSanitizer" OFF)

# libc++ hardening level -- complements the sanitizers rather than duplicating
# them (see cmake/Hardening.cmake). Set per-preset in CMakePresets.json.

# ---------------------------------------------------------------------------
# Shared interface targets: warnings + sanitizers.
# Every first-party target links these; third-party code does not, so we stay
# strict on our own code without drowning in warnings from dependencies.
# ---------------------------------------------------------------------------
include(CompilerWarnings)
include(Sanitizers)
include(StaticAnalysis)
include(Simd)

# Must precede Dependencies and add_subdirectory(): hardening is applied with
# add_compile_definitions() at directory scope so that first-party code and the
# FetchContent dependencies agree on _LIBCPP_HARDENING_MODE.
include(Hardening)

# ---------------------------------------------------------------------------
# Third-party dependencies (fetched from source; see cmake/Dependencies.cmake).
# ---------------------------------------------------------------------------
include(Dependencies)

# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(TOBY_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(TOBY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
