# Unit tests, powered by Catch2 v3 pulled in via FetchContent.

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1 # pin an exact release for reproducible builds
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# Catch2's CMake helpers provide catch_discover_tests() for CTest integration.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(toby_tests
  core/version_test.cpp
  tokenize/pretokenizer_test.cpp)

target_link_libraries(toby_tests
  PRIVATE
    toby::core
    toby::tokenize
    Catch2::Catch2WithMain
    toby::warnings
    toby::sanitizers
    toby::simd)

# Lint first-party test code too (but never Catch2's own sources).
toby_enable_tidy(toby_tests)

# Registers each TEST_CASE as an individual CTest test.
#
# TIMEOUT matters here: a pre-tokenizer that fails to advance its cursor hangs
# rather than returning a wrong answer, and without a timeout that wedges the
# whole run instead of reporting a failure. These are microsecond-scale unit
# tests, so 30s is pure headroom -- raise it only if a real test needs it.
catch_discover_tests(toby_tests
  PROPERTIES
    TIMEOUT 30)
