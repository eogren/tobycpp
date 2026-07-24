cmake_minimum_required(VERSION 3.25)

project(toby
  VERSION 0.1.0
  DESCRIPTION "A C++ inference server, built from scratch for learning."
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Guardrails: in-source builds are almost always a mistake.
# ---------------------------------------------------------------------------
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed. Use a preset, e.g. "
    "`cmake --preset gcc-debug` which builds into ./build/.")
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
option(TOBY_ENABLE_CLANG_TIDY  "Run clang-tidy during the build"   OFF)
option(TOBY_BUILD_TESTS        "Build the unit test suite"         ON)

# Sanitizers (mutually compose except TSan, which conflicts with ASan).
option(TOBY_SANITIZE_ADDRESS   "Enable AddressSanitizer"           OFF)
option(TOBY_SANITIZE_UNDEFINED "Enable UndefinedBehaviorSanitizer" OFF)
option(TOBY_SANITIZE_THREAD    "Enable ThreadSanitizer"            OFF)

# ---------------------------------------------------------------------------
# Shared interface targets: warnings + sanitizers.
# Every first-party target links these; third-party code does not, so we stay
# strict on our own code without drowning in warnings from dependencies.
# ---------------------------------------------------------------------------
include(CompilerWarnings)
include(Sanitizers)
include(StaticAnalysis)

# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(TOBY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
