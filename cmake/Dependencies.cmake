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
