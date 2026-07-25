# Wires clang-tidy into the build when TOBY_ENABLE_CLANG_TIDY is ON.
# We do NOT set CMAKE_CXX_CLANG_TIDY globally, because that would also lint
# third-party sources pulled in via FetchContent. Instead, targets opt in
# explicitly with toby_enable_tidy(<target>).

if(TOBY_ENABLE_CLANG_TIDY)
  # clang-tidy must match the compiler's major version. A mismatched tidy parses
  # C++23 and the libc++ headers with a different frontend, so it reports bogus
  # diagnostics or -- worse -- silently misses real ones and leaves the build
  # green. We do not list older versions as fallbacks: for a Clang 20 toolchain
  # they are never the answer, and finding one would only hide the problem.
  #
  # Plain `clang-tidy` stays in the list because several distros (Arch, Homebrew,
  # Nix) ship the current version under the unversioned name. It still has to
  # pass the version gate below.
  string(REGEX MATCH "^[0-9]+" _toby_cxx_major "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(TOBY_CLANG_TIDY_EXE NAMES clang-tidy-${_toby_cxx_major} clang-tidy)

  if(TOBY_CLANG_TIDY_EXE)
    execute_process(
      COMMAND "${TOBY_CLANG_TIDY_EXE}" --version
      OUTPUT_VARIABLE _toby_tidy_version_out
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "version ([0-9]+)" _toby_tidy_match "${_toby_tidy_version_out}")
    set(_toby_tidy_major "${CMAKE_MATCH_1}")

    if(NOT _toby_tidy_major STREQUAL _toby_cxx_major)
      message(FATAL_ERROR
        "toby: clang-tidy version skew. Found ${TOBY_CLANG_TIDY_EXE} (major "
        "'${_toby_tidy_major}') but the compiler is Clang ${_toby_cxx_major}. "
        "A mismatched clang-tidy misses real findings without failing. Install "
        "clang-tidy-${_toby_cxx_major}, or point TOBY_CLANG_TIDY_EXE at it.")
    endif()

    message(STATUS "toby: clang-tidy enabled -> ${TOBY_CLANG_TIDY_EXE} (v${_toby_tidy_major})")
    set(TOBY_CLANG_TIDY_COMMAND
      "${TOBY_CLANG_TIDY_EXE}"
      "--extra-arg-before=-Wno-unknown-warning-option"
      CACHE STRING "clang-tidy invocation for first-party targets" FORCE)
  else()
    # Fatal, not a warning: this preset exists only to run clang-tidy. Quietly
    # building without it produces a green build that analyzed nothing.
    message(FATAL_ERROR
      "toby: TOBY_ENABLE_CLANG_TIDY=ON but no clang-tidy-${_toby_cxx_major} was found.")
  endif()
else()
  set(TOBY_CLANG_TIDY_COMMAND "" CACHE STRING "" FORCE)
endif()

# Helper: opt a first-party target in to clang-tidy (no-op unless configured).
function(toby_enable_tidy target)
  if(TOBY_CLANG_TIDY_COMMAND)
    set_target_properties(${target} PROPERTIES
      CXX_CLANG_TIDY "${TOBY_CLANG_TIDY_COMMAND}")
  endif()
endfunction()
