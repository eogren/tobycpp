# libc++ hardening: turns undefined behavior in the standard library into a
# deterministic trap instead of a silently wrong answer.
#
# This is NOT redundant with the sanitizers. ASan checks the bounds of the
# underlying *allocation*; hardening checks the bounds of the *view*. Reading
# past the end of a subspan that sits in the middle of a live buffer is UB that
# ASan+UBSan miss completely and hardening catches:
#
#   std::vector<int> v{1,2,3,4,5,6,7,8};
#   auto mid = std::span<const int>{v}.subspan(2, 0);   // empty, interior
#   mid[0];   // ASan+UBSan: prints 3, exits 0.  FAST/EXTENSIVE/DEBUG: trap.
#
# Modes (see https://libcxx.llvm.org/Hardening.html):
#   none       no checks.
#   fast       low-overhead checks for UB with high security impact. ABI-neutral;
#              this is the mode intended to ship in production.
#   extensive  fast + more checks, still no complexity-class changes. ABI-neutral.
#   debug      everything, including O(n) checks. Emits a readable assertion
#              message; the cheaper modes just __builtin_trap (SIGILL, no text).
#
# Applied with add_compile_definitions() at directory scope rather than via an
# INTERFACE target, deliberately: every TU in the build -- ours and the
# FetchContent dependencies -- must agree on this macro. Compiling toby against
# hardened libc++ headers while Catch2 sees unhardened ones is an ODR violation
# waiting to happen. Include this module before any add_subdirectory().

set(
  TOBY_HARDENING
  "extensive"
  CACHE STRING
  "libc++ hardening level: none | fast | extensive | debug"
)
set_property(
  CACHE
    TOBY_HARDENING
  PROPERTY
    STRINGS
      none
      fast
      extensive
      debug
)

string(TOLOWER "${TOBY_HARDENING}" _toby_hardening)

if(_toby_hardening STREQUAL "none")
  set(_toby_hardening_macro _LIBCPP_HARDENING_MODE_NONE)
elseif(_toby_hardening STREQUAL "fast")
  set(_toby_hardening_macro _LIBCPP_HARDENING_MODE_FAST)
elseif(_toby_hardening STREQUAL "extensive")
  set(_toby_hardening_macro _LIBCPP_HARDENING_MODE_EXTENSIVE)
elseif(_toby_hardening STREQUAL "debug")
  set(_toby_hardening_macro _LIBCPP_HARDENING_MODE_DEBUG)
else()
  message(
    FATAL_ERROR
    "toby: TOBY_HARDENING must be one of none|fast|extensive|debug, "
    "got '${TOBY_HARDENING}'."
  )
endif()

# The macro is libc++-specific; libstdc++ ignores it silently, which would make
# this look enabled while doing nothing. Say so rather than pretend.
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  if(NOT _toby_hardening STREQUAL "none")
    message(
      WARNING
      "toby: TOBY_HARDENING=${_toby_hardening} requested, but _LIBCPP_HARDENING_MODE "
      "only affects libc++. No hardening will be applied."
    )
  endif()
else()
  message(STATUS "toby: libc++ hardening -> ${_toby_hardening}")
  add_compile_definitions(_LIBCPP_HARDENING_MODE=${_toby_hardening_macro})
endif()
