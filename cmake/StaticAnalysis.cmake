# Wires clang-tidy into the build when TOBY_ENABLE_CLANG_TIDY is ON.
# We do NOT set CMAKE_CXX_CLANG_TIDY globally, because that would also lint
# third-party sources pulled in via FetchContent. Instead, targets opt in
# explicitly with toby_enable_tidy(<target>).

if(TOBY_ENABLE_CLANG_TIDY)
  find_program(TOBY_CLANG_TIDY_EXE NAMES clang-tidy-19 clang-tidy-18 clang-tidy)
  if(TOBY_CLANG_TIDY_EXE)
    message(STATUS "toby: clang-tidy enabled -> ${TOBY_CLANG_TIDY_EXE}")
    set(TOBY_CLANG_TIDY_COMMAND
      "${TOBY_CLANG_TIDY_EXE}"
      "--extra-arg-before=-Wno-unknown-warning-option"
      CACHE STRING "clang-tidy invocation for first-party targets" FORCE)
  else()
    message(WARNING "toby: TOBY_ENABLE_CLANG_TIDY=ON but clang-tidy was not found.")
    set(TOBY_CLANG_TIDY_COMMAND "" CACHE STRING "" FORCE)
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
