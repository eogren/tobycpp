# Unit tests, powered by Catch2 v3 pulled in via FetchContent.

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1 # pin an exact release for reproducible builds
  GIT_SHALLOW    TRUE
  SYSTEM
)
FetchContent_MakeAvailable(Catch2)

# Catch2's CMake helpers provide catch_discover_tests() for CTest integration.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(toby_tests
  core/version_test.cpp
  tokenize/pretokenizer_test.cpp
  tokenize/utf8_test.cpp
  tokenize/vocab_test.cpp)

# Catch2 uses __COUNTER__ to create unique identifiers. Clang 22 diagnoses it
# under -Wpedantic at each first-party macro invocation even though Catch2 is a
# system include. Keep this test-only suppression after the warning bundle so
# the later, more specific flag is authoritative.
add_library(toby_test_dependency_warnings INTERFACE)
target_compile_options(toby_test_dependency_warnings
  INTERFACE
    $<$<AND:$<COMPILE_LANG_AND_ID:CXX,Clang>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,22>>:-Wno-c2y-extensions>)

target_link_libraries(toby_tests
  PRIVATE
    toby::core
    toby::tokenize
    Catch2::Catch2WithMain
    toby::warnings
    toby_test_dependency_warnings
    toby::sanitizers
    toby::simd)

# Fixture files are read at run time, and CTest runs tests from the build tree,
# so the source-tree path has to be compiled in. Baked as a define rather than
# resolved relative to argv[0]: that keeps the tests runnable straight from the
# binary too, not only under CTest.
target_compile_definitions(toby_tests
  PRIVATE
    TOBY_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# Loader tests intentionally exercise the private parsed representation until
# the decoded public Vocab exists.
target_include_directories(toby_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/tokenize)

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

# ---------------------------------------------------------------------------
# Differential test: the real pre-tokenizer vs. the GPT-2 regex.
#
# The C++ suite can only assert properties (losslessness, in-bounds views);
# exact chunk boundaries need an oracle, and the oracle is a regex with \p{L}
# classes and lookahead that std::regex cannot express. tools/pretokenizer_diff.py
# owns that reference. This just wires it to CTest so it runs with everything else.
#
# Optional by design: the script is a uv inline-script (it declares and fetches
# its own `regex` dependency), so a machine without uv -- or without network on
# the first run -- skips this test rather than failing the suite. The C++ tests
# still run.
# ---------------------------------------------------------------------------
find_program(TOBY_UV uv)

if(TOBY_UV AND TARGET pretokenize_dump)
  add_test(NAME pretokenizer_diff_gpt2
    COMMAND ${TOBY_UV} run --script
            ${CMAKE_SOURCE_DIR}/tools/pretokenizer_diff.py
            $<TARGET_FILE:pretokenize_dump>)

  # Longer than the unit tests: this shells out to Python, and uv may resolve and
  # cache `regex` on the very first run.
  #
  # LABELS lets you carve it out -- `ctest -L differential` to run only this,
  # `ctest -LE differential` to skip the Python round-trip in a tight edit loop.
  set_tests_properties(pretokenizer_diff_gpt2 PROPERTIES
    TIMEOUT 300
    LABELS "differential")
elseif(NOT TOBY_UV)
  message(STATUS
    "uv not found: skipping the pretokenizer differential test. "
    "Install uv to enable it (https://docs.astral.sh/uv/).")
endif()
