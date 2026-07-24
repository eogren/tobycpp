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
  core/version_test.cpp)

target_link_libraries(toby_tests
  PRIVATE
    toby::core
    Catch2::Catch2WithMain
    toby::warnings
    toby::sanitizers)

# Lint first-party test code too (but never Catch2's own sources).
toby_enable_tidy(toby_tests)

# Registers each TEST_CASE as an individual CTest test.
catch_discover_tests(toby_tests)
