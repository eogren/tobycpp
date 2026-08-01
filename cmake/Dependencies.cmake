# =============================================================================
# Third-party dependencies.
#
# These are "plumbing": libraries we consume, not the engine we are writing to
# learn. We pull them with FetchContent so they build from source against our
# own standard library (libc++) -- that avoids the classic ABI mismatch you get
# from linking a libstdc++-built system package into libc++ code.
#
# Notes on staying clean:
#   * We fetch with the SYSTEM keyword (CMake >= 3.25) so a dependency's headers
#     are included as -isystem in our translation units. Our own code stays
#     warnings-as-errors; third-party headers don't trip it.
#   * EXCLUDE_FROM_ALL keeps a dependency's own tests/examples/install rules out
#     of our default build.
# =============================================================================

include(FetchContent)

# ---------------------------------------------------------------------------
# cpp-httplib -- the HTTP layer for toby_server.
#
# Single-header, thread-per-request, zero required dependencies. This is the
# same library llama.cpp's server is built on: more than enough for serving
# inference, where concurrency is gated by the engine's scheduler, not the
# socket layer. We can revisit an async framework later if we ever need cheap
# idle-connection fan-out or WebSockets.
#
# The optional OpenSSL/zlib/brotli integrations are opportunistic ("if
# available") by default, which makes the build non-deterministic across
# machines. We turn them OFF for a hermetic first cut; flip a flag below when
# you actually want TLS or compression.
# ---------------------------------------------------------------------------
set(TOBY_HTTPLIB_VERSION "v0.51.0" CACHE STRING "cpp-httplib git tag to fetch")

set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_TEST OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        ${TOBY_HTTPLIB_VERSION}
  GIT_SHALLOW    TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(httplib)

# ---------------------------------------------------------------------------
# nlohmann/json -- parsing tokenizer vocabulary files.
#
# Every vocab format we care about is JSON: GPT-2 ships vocab.json, and Llama 3
# / Qwen ship a single tokenizer.json holding vocab, merges and added tokens.
#
# Chosen for ergonomics, not speed. These files are large (GPT-2's vocab.json is
# ~1MB; a Llama 3 tokenizer.json is ~9MB) and nlohmann's DOM parser is not fast
# on them -- but this runs once at model load, next to reading weights off disk,
# so it is not the thing to optimize. If load time ever does matter, simdjson is
# a drop-in swap at the call sites in src/tokenize/vocab.cpp; the loader hands
# back its own types precisely so nothing else in the tree sees the JSON library.
#
# JSON_Install/JSON_BuildTests off: we consume headers only, and its test suite
# is large enough to notice in a clean configure.
# ---------------------------------------------------------------------------
set(TOBY_JSON_VERSION "v3.11.3" CACHE STRING "nlohmann/json git tag to fetch")

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        ${TOBY_JSON_VERSION}
  GIT_SHALLOW    TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# ICU4C -- UTF-8 traversal and Unicode character properties.
#
# The pre-tokenizer needs both halves of Unicode handling: walking UTF-8 while
# retaining byte offsets, and classifying each decoded code point (letters,
# numbers, whitespace, ...). ICU's `uc` component supplies those APIs and the
# Unicode data behind them; the higher-level i18n/io components are unnecessary.
#
# ICU4C is intentionally found as a system library instead of fetched here.
# Upstream does not provide a CMake build -- its supported Unix build uses
# Autotools -- so FetchContent_MakeAvailable() cannot integrate it like the
# CMake-native dependencies above. CMake's FindICU module provides the imported
# ICU::uc target and is the supported consumer-side integration.
#
# Tokenizer code should prefer ICU's stable C API. Besides being the smaller
# surface for this job, that avoids coupling our libc++ build to the C++ ABI of
# whichever compiler built the system ICU package.
# ---------------------------------------------------------------------------
if(APPLE)
  # Homebrew keeps ICU keg-only, so it is intentionally absent from the normal
  # system search paths. Preserve any user-supplied prefixes, then add the
  # Homebrew prefix as a convenience when Homebrew and its ICU formula exist.
  find_program(_toby_brew_exe NAMES brew)
  if(_toby_brew_exe)
    execute_process(
      COMMAND "${_toby_brew_exe}" --prefix icu4c
      RESULT_VARIABLE _toby_brew_icu_result
      OUTPUT_VARIABLE _toby_brew_icu_prefix
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_toby_brew_icu_result EQUAL 0 AND _toby_brew_icu_prefix)
      list(APPEND CMAKE_PREFIX_PATH "${_toby_brew_icu_prefix}")
    endif()
  endif()
endif()

find_package(ICU REQUIRED COMPONENTS uc)

# ---------------------------------------------------------------------------
# Boost.Unordered -- open-addressed hash containers.
#
# We want `boost::unordered_flat_map`, which is a genuinely different data
# structure from `std::unordered_map` rather than a tuned version of it. The
# standard's container is specified in a way that forces closed addressing:
# `bucket()`/`begin(n)` expose buckets, and reference stability across rehash is
# guaranteed. That mandates a node per element, so a lookup is a modulo, a
# pointer chase to the bucket head, and then a linked-list walk with a cache miss
# per hop. Boost's flat map stores elements inline in one array with SIMD-scanned
# metadata; the common lookup is one cache line. The cost you accept is that it
# invalidates references on rehash and its `value_type` is `pair<K, V>`, not
# `pair<const K, V>` -- which is precisely why it cannot be spelled as a
# conforming `std::unordered_map` and has to be a separate type.
#
# Whether that wins here is an open question, not a given: the flat map's edge is
# largest on lookup-heavy workloads with small keys, and vocab keys are short
# `std::string`s that mostly fit in the SSO buffer, so the element array stays
# dense. Measure before believing it.
#
# ON SIZE: this archive is ~104MB compressed and ~700MB extracted -- for a
# header-only map. `BOOST_INCLUDE_LIBRARIES` below limits what gets *configured*
# (only Unordered and its transitive dependencies get add_subdirectory'd, so the
# configure stays quick), but the download is all-or-nothing. If that ever grates,
# `ankerl::unordered_dense` is a single header with comparable benchmarks and the
# same open-addressing design; the swap is local to whoever includes the map.
#
# We use the release archive rather than the git superproject on purpose: the
# superproject is ~200 submodules, and a shallow clone of it is neither shallow
# nor quick. The archive is pinned by SHA256, which also makes this the only
# dependency here that is genuinely reproducible -- a git tag can be moved.
# ---------------------------------------------------------------------------
set(TOBY_BOOST_VERSION "1.91.0-1" CACHE STRING "Boost release archive to fetch")
set(TOBY_BOOST_SHA256
    "cc5dc5006ecbdf0051f90979be31b4eee5987d9ae14ae9fb9c03cfa43fa3cdad"
    CACHE STRING "SHA256 of the Boost CMake release archive")

# The list of Boost libraries to build. Transitive dependencies (Assert, Config,
# ContainerHash, Core, Mp11, ThrowException, ...) are resolved automatically --
# do NOT list them by hand. Keep this list minimal: every entry is another
# subdirectory in every configure.
set(BOOST_INCLUDE_LIBRARIES unordered)
set(BOOST_ENABLE_CMAKE ON)

# Share the tarball and the extracted tree across presets.
#
# By default FetchContent puts everything under `${CMAKE_BINARY_DIR}/_deps`, which
# is per-preset -- so clang-debug, clang-asan, clang-tsan and clang-tidy would each
# pull 104MB and each keep their own 700MB copy. Pointing SOURCE_DIR and DOWNLOAD_DIR
# at one shared location under build/ fixes both: the download step hashes the
# existing archive and skips the fetch, and every preset extracts to the same tree.
#
# Note this deliberately does NOT move the *binary* dir (which is what setting
# FETCHCONTENT_BASE_DIR globally would do). Boost.Unordered is header-only, so its
# binary dir is empty today -- but sharing build trees across presets that disagree
# about sanitizers and hardening is exactly the kind of thing that produces a
# baffling link error two months from now. Source is shared; build stays per-preset.
#
# The one caveat: configuring two presets *concurrently* can race on the extract.
# Configure them one at a time, or delete build/_shared and retry.
set(TOBY_DEPS_CACHE ${CMAKE_SOURCE_DIR}/build/_shared)

FetchContent_Declare(Boost
  URL https://github.com/boostorg/boost/releases/download/boost-${TOBY_BOOST_VERSION}/boost-${TOBY_BOOST_VERSION}-cmake.tar.xz
  URL_HASH SHA256=${TOBY_BOOST_SHA256}
  DOWNLOAD_DIR ${TOBY_DEPS_CACHE}/downloads
  SOURCE_DIR   ${TOBY_DEPS_CACHE}/boost-src
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(Boost)
